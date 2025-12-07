"""
pricing.py

Core pricing functions for the AI Hedging Agent.

This module implements the Black–Scholes model for European options
with continuous dividend yield. The functions are designed to be
numerically stable, reusable, and suitable for production use in a
larger quantitative finance system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy.stats import norm


OptionType = Literal["call", "put"]


@dataclass
class BlackScholesParams:
    """
    Container for Black–Scholes input parameters.

    Attributes
    ----------
    spot : float
        Current price of the underlying asset (S).
    strike : float
        Strike price of the option (K).
    maturity : float
        Time to maturity in years (T).
    risk_free_rate : float
        Continuously compounded risk-free interest rate (r).
    volatility : float
        Volatility of the underlying asset (sigma).
    dividend_yield : float
        Continuous dividend yield of the underlying asset (q).
    """

    spot: float
    strike: float
    maturity: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0


def _validate_bs_params(params: BlackScholesParams) -> None:
    """
    Validate Black–Scholes parameters and raise ValueError if invalid.
    """
    if params.spot <= 0:
        raise ValueError("Spot price must be strictly positive.")
    if params.strike <= 0:
        raise ValueError("Strike price must be strictly positive.")
    if params.maturity <= 0:
        raise ValueError("Time to maturity must be strictly positive.")
    if params.volatility <= 0:
        raise ValueError("Volatility must be strictly positive.")


def _d1_d2(params: BlackScholesParams) -> Tuple[float, float]:
    """
    Compute d1 and d2 for the Black–Scholes formula.

    Returns
    -------
    (d1, d2) : tuple of floats
    """
    _validate_bs_params(params)

    S = params.spot
    K = params.strike
    T = params.maturity
    r = params.risk_free_rate
    sigma = params.volatility
    q = params.dividend_yield

    # Use numpy.log and numpy.sqrt for numerical stability
    numerator = np.log(S / K) + (r - q + 0.5 * sigma**2) * T
    denominator = sigma * np.sqrt(T)

    d1 = numerator / denominator
    d2 = d1 - sigma * np.sqrt(T)
    return float(d1), float(d2)


def black_scholes_price(
    params: BlackScholesParams,
    option_type: OptionType = "call",
) -> float:
    """
    Compute the price of a European option using the Black–Scholes model.

    Parameters
    ----------
    params : BlackScholesParams
        Input parameters for the Black–Scholes model.
    option_type : {"call", "put"}, default "call"
        Type of the option.

    Returns
    -------
    float
        Theoretical Black–Scholes price of the option.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")

    d1, d2 = _d1_d2(params)

    S = params.spot
    K = params.strike
    T = params.maturity
    r = params.risk_free_rate
    q = params.dividend_yield

    discount_factor_r = np.exp(-r * T)
    discount_factor_q = np.exp(-q * T)

    if option_type == "call":
        price = (
            discount_factor_q * S * norm.cdf(d1)
            - discount_factor_r * K * norm.cdf(d2)
        )
    else:  # put
        price = (
            discount_factor_r * K * norm.cdf(-d2)
            - discount_factor_q * S * norm.cdf(-d1)
        )

    return float(price)


def black_scholes_call(params: BlackScholesParams) -> float:
    """
    Convenience wrapper for a European call option price.
    """
    return black_scholes_price(params, option_type="call")


def black_scholes_put(params: BlackScholesParams) -> float:
    """
    Convenience wrapper for a European put option price.
    """
    return black_scholes_price(params, option_type="put")
