"""
monte_carlo.py

Monte Carlo simulation utilities for the AI Hedging Agent.

This module provides:
- Geometric Brownian Motion (GBM) path simulation
- European option pricing via Monte Carlo
- Optional antithetic variates for variance reduction
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .pricing import BlackScholesParams, OptionType


def simulate_gbm_paths(
    params: BlackScholesParams,
    n_paths: int,
    n_steps: int,
    antithetic: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate GBM paths for the underlying asset.

    Parameters
    ----------
    params : BlackScholesParams
        Input parameters (spot, maturity, risk-free rate, volatility, dividend yield).
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps per path.
    antithetic : bool, default True
        If True, use antithetic variates for variance reduction.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps + 1) containing simulated price paths.
        The first column corresponds to time t=0 (initial spot).
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    rng = np.random.default_rng(seed)

    S0 = params.spot
    T = params.maturity
    r = params.risk_free_rate
    q = params.dividend_yield
    sigma = params.volatility

    dt = T / n_steps

    # For antithetic variates, simulate half the paths and mirror them
    base_paths = n_paths // 2 if antithetic else n_paths

    # Standard normal random numbers
    Z = rng.normal(size=(base_paths, n_steps))

    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)
        # If n_paths is odd, drop the last extra row
        if Z.shape[0] > n_paths:
            Z = Z[:n_paths, :]

    # Preallocate array for paths
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Iteratively build the paths
    for t in range(1, n_steps + 1):
        # log-Euler discretisation of GBM
        paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * Z[:, t - 1])

    return paths


def price_european_option_mc(
    params: BlackScholesParams,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    n_steps: int = 100,
    antithetic: bool = True,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Price a European option via Monte Carlo simulation under GBM.

    Parameters
    ----------
    params : BlackScholesParams
        Black–Scholes input parameters (used for GBM simulation).
    option_type : {"call", "put"}, default "call"
        Type of the option.
    n_paths : int, default 100_000
        Number of Monte Carlo paths.
    n_steps : int, default 100
        Number of time steps per path.
    antithetic : bool, default True
        Whether to use antithetic variates.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    price : float
        Monte Carlo estimate of the option price.
    std_error : float
        Standard error of the estimator.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")

    paths = simulate_gbm_paths(
        params=params,
        n_paths=n_paths,
        n_steps=n_steps,
        antithetic=antithetic,
        seed=seed,
    )

    S_T = paths[:, -1]
    K = params.strike
    r = params.risk_free_rate
    T = params.maturity

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discounted_payoffs = np.exp(-r * T) * payoffs

    price = float(discounted_payoffs.mean())
    std_error = float(discounted_payoffs.std(ddof=1) / np.sqrt(len(discounted_payoffs)))

    return price, std_error
