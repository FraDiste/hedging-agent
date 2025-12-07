from src.portfolio_parser import (
    parse_portfolio_text,
    build_bs_portfolio_from_parsed,
    CONTROLLED_SYNTAX_DOC,
)
from src.greeks import all_greeks
from src.pricing import BlackScholesParams

import numpy as np

print("Running test_phase6...")

portfolio_text = """
Long 2 call strike 100 maturity 1.0
Short 1 put strike 90 maturity 0.5
"""

print("Input text:")
print(portfolio_text)

# 1) Parse text
positions = parse_portfolio_text(portfolio_text)
print("\nParsed positions:")
for p in positions:
    print(p)

# 2) Build Black–Scholes portfolio
spot = 100.0
r = 0.02
sigma = 0.20
q = 0.0

params_list, types_list, quantities = build_bs_portfolio_from_parsed(
    positions,
    spot=spot,
    risk_free_rate=r,
    volatility=sigma,
    dividend_yield=q,
)

print("\nBuilt BS portfolio:")
for params, opt_type, qty in zip(params_list, types_list, quantities):
    print(f"qty={qty}, type={opt_type}, strike={params.strike}, T={params.maturity}")

# 3) Compute aggregated Greeks with quantities
agg_greeks = {"price": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

for params, opt_type, qty in zip(params_list, types_list, quantities):
    g = all_greeks(params, option_type=opt_type)
    for k in agg_greeks:
        agg_greeks[k] += qty * g[k]

print("\nAggregated Greeks (with quantities):")
for k, v in agg_greeks.items():
    print(f"{k}: {v:.6f}")
