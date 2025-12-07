"""
optimizer.py

Hedging optimization module for the AI Hedging Agent.

This module provides:
- Greek-matching optimization using convex objectives
- Multi-scenario hedging optimization
- Budget and position constraints
"""

from __future__ import annotations

from typing import Dict, Sequence, Optional

import numpy as np
import cvxpy as cp

from .pricing import BlackScholesParams, black_scholes_price, OptionType
from .greeks import all_greeks
from .stress_testing import StressScenario, stress_test_option_black_scholes


def optimize_greeks(
    portfolio_params: Sequence[BlackScholesParams],
    portfolio_types: Sequence[OptionType],
    hedge_params: Sequence[BlackScholesParams],
    hedge_types: Sequence[OptionType],
    target_greeks: Dict[str, float],
    portfolio_quantities: Optional[Sequence[float]] = None,
    max_position: float = 100.0,
    budget: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Optimize hedge quantities to match target Greeks.

    Parameters
    ----------
    portfolio_params : list of BlackScholesParams
        Instruments in the base portfolio.
    portfolio_types : list of {"call", "put"}
        Option types for the base portfolio.
    hedge_params : list of BlackScholesParams
        Instruments available for hedging.
    hedge_types : list of {"call", "put"}
        Option types for the hedge instruments.
    target_greeks : dict
        Desired target Greek values, e.g. {"delta": 0, "gamma": 0}.
        Missing keys default to zero.
    portfolio_quantities : list of float, optional
        Signed quantities (positive = long, negative = short) for each
        portfolio instrument. If None, all quantities are assumed to be +1.
    max_position : float
        Maximum absolute position per hedge instrument.
    budget : float, optional
        Maximum net premium outlay allowed for the hedge.

    Returns
    -------
    dict with keys:
        "x"             : optimal hedge quantities (numpy array)
        "greeks_before" : portfolio Greeks before hedging (numpy array)
        "greeks_after"  : portfolio Greeks after hedging (numpy array)
    """

    n_port = len(portfolio_params)
    if portfolio_quantities is None:
        q_port = np.ones(n_port)
    else:
        q_port = np.asarray(portfolio_quantities, dtype=float)
        if q_port.shape[0] != n_port:
            raise ValueError("portfolio_quantities must match portfolio_params length.")

    # 1) Compute portfolio Greeks (sum over instruments, with quantities)
    portfolio_g = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for w, p, t in zip(q_port, portfolio_params, portfolio_types):
        g = all_greeks(p, option_type=t)
        for k in portfolio_g:
            portfolio_g[k] += w * g[k]

    # 2) Compute hedge Greeks per unit and build matrix (5 x n_hedge)
    hedge_matrix_rows = []
    for p, t in zip(hedge_params, hedge_types):
        g = all_greeks(p, option_type=t)
        hedge_matrix_rows.append(
            [g["delta"], g["gamma"], g["vega"], g["theta"], g["rho"]]
        )

    hedge_matrix = np.array(hedge_matrix_rows).T  # shape: (5, n_hedge)
    n_hedge = hedge_matrix.shape[1]

    x = cp.Variable(n_hedge)  # hedge quantities

    # 3) Build target and portfolio Greek vectors
    greek_vector = np.array(
        [
            target_greeks.get("delta", 0.0),
            target_greeks.get("gamma", 0.0),
            target_greeks.get("vega", 0.0),
            target_greeks.get("theta", 0.0),
            target_greeks.get("rho", 0.0),
        ]
    )

    portfolio_vector = np.array(
        [
            portfolio_g["delta"],
            portfolio_g["gamma"],
            portfolio_g["vega"],
            portfolio_g["theta"],
            portfolio_g["rho"],
        ]
    )

    # Objective: minimize || target - (portfolio + hedge*x) ||_2
    objective = cp.Minimize(
        cp.norm(greek_vector - (portfolio_vector + hedge_matrix @ x), 2)
    )

    constraints = [cp.abs(x) <= max_position]

    # Optional budget constraint on net premium
    if budget is not None:
        hedge_costs = np.array(
            [black_scholes_price(p, t) for p, t in zip(hedge_params, hedge_types)]
        )
        constraints.append(hedge_costs @ x <= budget)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    x_opt = np.array(x.value, dtype=float)
    greeks_after = portfolio_vector + hedge_matrix @ x_opt

    return {
        "x": x_opt,
        "greeks_before": portfolio_vector,
        "greeks_after": greeks_after,
    }


def optimize_scenarios_worst_case(
    portfolio_params: Sequence[BlackScholesParams],
    portfolio_types: Sequence[OptionType],
    hedge_params: Sequence[BlackScholesParams],
    hedge_types: Sequence[OptionType],
    scenarios: Sequence[StressScenario],
    portfolio_quantities: Optional[Sequence[float]] = None,
    max_position: float = 100.0,
) -> Dict[str, np.ndarray]:
    """
    Hedge optimization based on worst-case scenario P&L.

    The objective is to find hedge quantities x that minimise the
    worst-case loss across a set of stress scenarios.

    Parameters
    ----------
    portfolio_params : list of BlackScholesParams
        Instruments in the base portfolio.
    portfolio_types : list of {"call", "put"}
        Option types for the base portfolio.
    hedge_params : list of BlackScholesParams
        Hedge instruments (unit notional).
    hedge_types : list of {"call", "put"}
        Option types for the hedge instruments.
    scenarios : list of StressScenario
        Stress scenarios for spot and volatility.
    portfolio_quantities : list of float, optional
        Signed quantities (positive = long, negative = short) for each
        portfolio instrument. If None, all quantities are assumed to be +1.
    max_position : float
        Maximum absolute position per hedge instrument.

    Returns
    -------
    dict with keys:
        "x"               : optimal hedge quantities (numpy array)
        "worst_case_loss" : optimal worst-case loss (float)
        "scenario_losses" : array of losses per scenario at optimum
    """

    n_port = len(portfolio_params)
    if portfolio_quantities is None:
        q_port = np.ones(n_port)
    else:
        q_port = np.asarray(portfolio_quantities, dtype=float)
        if q_port.shape[0] != n_port:
            raise ValueError("portfolio_quantities must match portfolio_params length.")

    n_hedge = len(hedge_params)
    x = cp.Variable(n_hedge)

    scenario_loss_exprs = []

    for sc in scenarios:
        # Portfolio P&L under scenario sc (sum over instruments with quantities)
        pnl_portfolio = 0.0
        for w, p, t in zip(q_port, portfolio_params, portfolio_types):
            res = stress_test_option_black_scholes(p, t, [sc])[0]
            pnl_portfolio += w * res["pnl"]

        # Hedge P&L per unit under scenario sc
        pnl_hedges_per_unit = []
        for p, t in zip(hedge_params, hedge_types):
            res = stress_test_option_black_scholes(p, t, [sc])[0]
            pnl_hedges_per_unit.append(res["pnl"])

        pnl_hedges_per_unit = np.array(pnl_hedges_per_unit)

        # Total loss in scenario sc = - (pnl_portfolio + pnl_hedges_per_unit @ x)
        scenario_loss_exprs.append(-(pnl_portfolio + pnl_hedges_per_unit @ x))

    # Stack scenario losses into a vector expression
    scenario_losses_expr = cp.hstack(scenario_loss_exprs)

    # Objective: minimise worst-case loss across scenarios
    objective = cp.Minimize(cp.max(scenario_losses_expr))

    constraints = [cp.abs(x) <= max_position]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    x_opt = np.array(x.value, dtype=float)
    scenario_losses_val = np.array(scenario_losses_expr.value, dtype=float)
    worst_case_loss = float(np.max(scenario_losses_val))

    return {
        "x": x_opt,
        "worst_case_loss": worst_case_loss,
        "scenario_losses": scenario_losses_val,
    }
