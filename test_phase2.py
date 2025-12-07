print("Running test_phase1...")

from src.pricing import BlackScholesParams, black_scholes_call
from src.monte_carlo import price_european_option_mc
from src.risk_metrics import value_at_risk, expected_shortfall, parametric_var

import numpy as np


print("Running test_phase2...")

params = BlackScholesParams(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    risk_free_rate=0.02,
    volatility=0.20,
    dividend_yield=0.0,
)

# 1) Compare Monte Carlo price vs Black–Scholes price
bs_price = black_scholes_call(params)
mc_price, mc_se = price_european_option_mc(
    params,
    option_type="call",
    n_paths=50_000,
    n_steps=100,
    antithetic=True,
    seed=42,
)

print(f"Black–Scholes call price: {bs_price:.6f}")
print(f"Monte Carlo call price : {mc_price:.6f} (std. error ~ {mc_se:.6f})")

# 2) Generate a fake P&L distribution and compute VaR/ES
rng = np.random.default_rng(123)
# Assume normally distributed P&L with mean 0 and std 100
pnl_samples = rng.normal(loc=0.0, scale=100.0, size=50_000)

var_99 = value_at_risk(pnl_samples, alpha=0.99)
es_99 = expected_shortfall(pnl_samples, alpha=0.99)

mean_pnl = float(pnl_samples.mean())
std_pnl = float(pnl_samples.std(ddof=1))
param_var_99 = parametric_var(mean_pnl, std_pnl, alpha=0.99)

print(f"Historical / MC VaR 99%: {var_99:.2f}")
print(f"Expected Shortfall 99% : {es_99:.2f}")
print(f"Parametric VaR 99%     : {param_var_99:.2f}")
