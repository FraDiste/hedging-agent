"""
stress_testing.py

Scenario and stress-testing utilities for the AI Hedging Agent.

This module provides:
- Simple price and volatility shocks for single Black–Scholes options
- Linear stress tests for simple portfolios (e.g. cash, stocks, futures)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

import numpy as np

from .pricing import BlackScholesParams, OptionType, black_scholes_price


@dataclass
class StressScenario:
    """
    Container for a single stress scenario.

    Attributes
    ----------
    name : str
        Name or label of the scenario (e.g. "Spot -10%, Vol +20%").
    spot_multiplier : float
        Multiplicative factor applied to the current spot (e.g. 0.9 for -10%).
    vol_multiplier : float
        Multiplicative factor applied to the current volatility (e.g. 1.2 for +20%).
    """

    name: str
    spot_multiplier: float = 1.0
    vol_multiplier: float = 1.0


def stress_test_option_black_scholes(
    params: BlackScholesParams,
    option_type: OptionType,
    scenarios: Sequence[StressScenario],
) -> List[Dict[str, float]]:
    """
    Perform stress testing on a single European option using Black–Scholes.

    The function returns, for each scenario, the shocked price and the
    resulting P&L relative to the base case.

    Parameters
    ----------
    params : BlackScholesParams
        Base-case parameters.
    option_type : {"call", "put"}
        Type of the option.
    scenarios : Sequence[StressScenario]
        List of scenarios with spot and volatility multipliers.

    Returns
    -------
    List[Dict[str, float]]
        A list of dictionaries, each containing:
        "name", "spot", "volatility", "price_base", "price_shocked", "pnl".
    """
    # Base-case price
    base_price = black_scholes_price(params, option_type=option_type)

    results: List[Dict[str, float]] = []

    for sc in scenarios:
        shocked_params = BlackScholesParams(
            spot=params.spot * sc.spot_multiplier,
            strike=params.strike,
            maturity=params.maturity,
            risk_free_rate=params.risk_free_rate,
            volatility=params.volatility * sc.vol_multiplier,
            dividend_yield=params.dividend_yield,
        )

        shocked_price = black_scholes_price(shocked_params, option_type=option_type)
        pnl = shocked_price - base_price

        results.append(
            {
                "name": sc.name,
                "spot": shocked_params.spot,
                "volatility": shocked_params.volatility,
                "price_base": base_price,
                "price_shocked": shocked_price,
                "pnl": pnl,
            }
        )

    return results


def stress_test_linear_portfolio(
    spot_vector: np.ndarray,
    position_vector: np.ndarray,
    spot_shocks: np.ndarray,
) -> np.ndarray:
    """
    Perform a simple linear stress test on a portfolio of spot instruments.

    This function assumes a portfolio of linear instruments (e.g. stocks,
    futures) where the P&L under a price shock is proportional to the
    change in the underlying spot price.

    Parameters
    ----------
    spot_vector : np.ndarray
        Current spot prices of each asset (shape: [n_assets]).
    position_vector : np.ndarray
        Position in each asset, expressed in units of the underlying
        (positive for long, negative for short).
    spot_shocks : np.ndarray
        Relative shocks to apply to each asset's spot price
        (shape: [n_scenarios, n_assets]), e.g. -0.1 for -10%.

    Returns
    -------
    np.ndarray
        P&L for each scenario (shape: [n_scenarios]).
    """
    spot_vector = np.asarray(spot_vector, dtype=float)
    position_vector = np.asarray(position_vector, dtype=float)
    spot_shocks = np.asarray(spot_shocks, dtype=float)

    if spot_shocks.ndim == 1:
        spot_shocks = spot_shocks.reshape(1, -1)

    if spot_vector.shape != position_vector.shape:
        raise ValueError("spot_vector and position_vector must have the same shape.")

    if spot_shocks.shape[1] != spot_vector.shape[0]:
        raise ValueError(
            "spot_shocks must have shape (n_scenarios, n_assets) "
            "with n_assets equal to len(spot_vector)."
        )

    # Base portfolio value
    base_value = np.sum(spot_vector * position_vector)

    # Shocked spot prices for each scenario
    shocked_spots = spot_vector * (1.0 + spot_shocks)

    # Portfolio value per scenario
    portfolio_values = shocked_spots @ position_vector

    # P&L = shocked value - base value
    pnl_scenarios = portfolio_values - base_value
    return pnl_scenarios
