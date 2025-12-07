"""
risk_metrics.py

Risk metrics for the AI Hedging Agent.

This module provides:
- Historical / Monte Carlo Value-at-Risk (VaR)
- Parametric (normal) VaR
- Expected Shortfall (ES)
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


def value_at_risk(
    pnl: np.ndarray,
    alpha: float = 0.99,
    method: Literal["historical", "mc"] = "historical",
) -> float:
    """
    Compute Value-at-Risk (VaR) from a distribution of P&L values.

    Parameters
    ----------
    pnl : np.ndarray
        Array of profit-and-loss values (negative values = losses).
    alpha : float, default 0.99
        Confidence level. For example, alpha = 0.99 corresponds to
        the 1% worst losses.
    method : {"historical", "mc"}, default "historical"
        Label describing the origin of the P&L distribution.

    Returns
    -------
    float
        VaR at confidence level alpha (reported as a positive number).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1:
        raise ValueError("pnl must be a one-dimensional array.")

    # e.g. alpha=0.99 -> 1% quantile of losses
    var_quantile = np.quantile(pnl, 1 - alpha)

    # VaR is usually reported as positive loss
    return float(-var_quantile)


def parametric_var(
    mean: float,
    std: float,
    alpha: float = 0.99,
) -> float:
    """
    Compute parametric (normal) Value-at-Risk.

    Parameters
    ----------
    mean : float
        Expected P&L.
    std : float
        Standard deviation of P&L.
    alpha : float, default 0.99
        Confidence level.

    Returns
    -------
    float
        Parametric VaR (positive number).
    """
    if std < 0:
        raise ValueError("Standard deviation must be non-negative.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    # For a normal distribution: VaR_alpha = - (mean + z_{1-alpha} * std)
    # where z_{1-alpha} is the critical value of the standard normal.
    from scipy.stats import norm

    z = norm.ppf(1 - alpha)
    var_value = -(mean + z * std)
    return float(max(var_value, 0.0))


def expected_shortfall(
    pnl: np.ndarray,
    alpha: float = 0.99,
) -> float:
    """
    Compute Expected Shortfall (ES), also known as Conditional VaR.

    Parameters
    ----------
    pnl : np.ndarray
        Array of profit-and-loss values (negative values = losses).
    alpha : float, default 0.99
        Confidence level.

    Returns
    -------
    float
        Expected Shortfall at level alpha (positive number).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1:
        raise ValueError("pnl must be a one-dimensional array.")

    var_level = np.quantile(pnl, 1 - alpha)

    tail_losses = pnl[pnl <= var_level]  # losses worse than or equal to VaR
    if tail_losses.size == 0:
        return 0.0

    es_value = -tail_losses.mean()
    return float(es_value)

def var_parametric(
    pnl: np.ndarray,
    alpha: float = 0.99,
) -> float:
    """
    Convenience wrapper for parametric VaR using empirical mean and std.

    Parameters
    ----------
    pnl : np.ndarray
        Array of profit-and-loss values.
    alpha : float, default 0.99
        Confidence level.

    Returns
    -------
    float
        Parametric VaR (positive number) estimated from the
        empirical mean and standard deviation of pnl.
    """
    pnl = np.asarray(pnl, dtype=float)
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1))
    return parametric_var(mean, std, alpha=alpha)
