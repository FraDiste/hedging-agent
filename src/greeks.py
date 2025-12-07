"""
greeks.py

Black–Scholes Greeks for the AI Hedging Agent.

This module provides functions to compute the standard Greeks
(Delta, Gamma, Vega, Theta, Rho) for European options under the
Black–Scholes framework with continuous dividend yield.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Literal

import numpy as np
from scipy.stats import norm

from .pricing import BlackScholesParams, OptionType, _d1_d2, black_scholes_price


def delta(params: BlackScholesParams, option_type: OptionType = "call") -> float:
    """
    Compute the Delta of a European option.

    Returns
    -------
    float
        Delta of the option with respect to the underlying spot price.
    """
    option_type = option_type.lower()
    d1, _ = _d1_d2(params)

    q = params.dividend_yield
    T = params.maturity
    discount_factor_q = np.exp(-q * T)

    if option_type == "call":
        value = discount_factor_q * norm.cdf(d1)
    elif option_type == "put":
        value = -discount_factor_q * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")

    return float(value)


def gamma(params: BlackScholesParams) -> float:
    """
    Compute the Gamma of a European option (same for calls and puts).

    Returns
    -------
    float
        Gamma of the option.
    """
    d1, _ = _d1_d2(params)

    S = params.spot
    T = params.maturity
    q = params.dividend_yield
    sigma = params.volatility

    discount_factor_q = np.exp(-q * T)

    value = (
        discount_factor_q
        * norm.pdf(d1)
        / (S * sigma * np.sqrt(T))
    )
    return float(value)


def vega(params: BlackScholesParams) -> float:
    """
    Compute the Vega of a European option.

    Returns
    -------
    float
        Vega of the option with respect to volatility.
        Note: returned in price units per 1.0 change in volatility
        (e.g. multiply by 0.01 for per 1% change).
    """
    d1, _ = _d1_d2(params)

    S = params.spot
    T = params.maturity
    q = params.dividend_yield

    discount_factor_q = np.exp(-q * T)

    value = discount_factor_q * S * norm.pdf(d1) * np.sqrt(T)
    return float(value)


def theta(params: BlackScholesParams, option_type: OptionType = "call") -> float:
    """
    Compute the Theta of a European option (per year).

    Returns
    -------
    float
        Theta of the option with respect to the passage of time.
    """
    option_type = option_type.lower()
    d1, d2 = _d1_d2(params)

    S = params.spot
    K = params.strike
    T = params.maturity
    r = params.risk_free_rate
    q = params.dividend_yield
    sigma = params.volatility

    discount_factor_q = np.exp(-q * T)
    discount_factor_r = np.exp(-r * T)

    first_term = (
        -discount_factor_q
        * S
        * norm.pdf(d1)
        * sigma
        / (2 * np.sqrt(T))
    )

    if option_type == "call":
        second_term = r * discount_factor_r * K * norm.cdf(d2)
        third_term = -q * discount_factor_q * S * norm.cdf(d1)
    elif option_type == "put":
        second_term = -r * discount_factor_r * K * norm.cdf(-d2)
        third_term = q * discount_factor_q * S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")

    value = first_term - second_term + third_term
    return float(value)


def rho(params: BlackScholesParams, option_type: OptionType = "call") -> float:
    """
    Compute the Rho of a European option.

    Returns
    -------
    float
        Rho of the option with respect to the risk-free rate.
    """
    option_type = option_type.lower()
    _, d2 = _d1_d2(params)

    K = params.strike
    T = params.maturity
    r = params.risk_free_rate

    discount_factor_r = np.exp(-r * T)

    if option_type == "call":
        value = T * discount_factor_r * K * norm.cdf(d2)
    elif option_type == "put":
        value = -T * discount_factor_r * K * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")

    return float(value)


def all_greeks(
    params: BlackScholesParams,
    option_type: OptionType = "call",
) -> Dict[str, float]:
    """
    Convenience function that returns price and all major Greeks.

    Returns
    -------
    dict
        Dictionary with the keys:
        "price", "delta", "gamma", "vega", "theta", "rho".
    """
    return {
        "price": black_scholes_price(params, option_type=option_type),
        "delta": delta(params, option_type=option_type),
        "gamma": gamma(params),
        "vega": vega(params),
        "theta": theta(params, option_type=option_type),
        "rho": rho(params, option_type=option_type),
    }
