from src.pricing import BlackScholesParams
from src.optimizer import optimize_greeks, optimize_scenarios_worst_case
from src.stress_testing import StressScenario, stress_test_option_black_scholes
from src.greeks import all_greeks

import numpy as np

print("Running test_phase4...")

# ============================
# 1) Greek-matching optimization
# ============================

# Base portfolio: one long ATM call
portfolio_params = [
    BlackScholesParams(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        risk_free_rate=0.02,
        volatility=0.20,
        dividend_yield=0.0,
    )
]
portfolio_types = ["call"]

# Hedge instruments: one OTM call and one put
hedge_params = [
    BlackScholesParams(
        spot=100.0,
        strike=110.0,
        maturity=1.0,
        risk_free_rate=0.02,
        volatility=0.20,
        dividend_yield=0.0,
    ),
    BlackScholesParams(
        spot=100.0,
        strike=90.0,
        maturity=1.0,
        risk_free_rate=0.02,
        volatility=0.20,
        dividend_yield=0.0,
    ),
]
hedge_types = ["call", "put"]

target_greeks = {"delta": 0.0, "gamma": 0.0}

result_greeks = optimize_greeks(
    portfolio_params=portfolio_params,
    portfolio_types=portfolio_types,
    hedge_params=hedge_params,
    hedge_types=hedge_types,
    target_greeks=target_greeks,
    max_position=10.0,
    budget=None,
)

x_opt = result_greeks["x"]
before = result_greeks["greeks_before"]
after = result_greeks["greeks_after"]

print("\n=== Greek-matching optimization ===")
print("Optimal hedge quantities (x):", x_opt)
print("Greeks before hedging:", before)
print("Greeks after hedging :", after)

# ============================
# 2) Scenario-based worst-case optimization
# ============================

scenarios = [
    StressScenario(name="Spot -10%", spot_multiplier=0.9, vol_multiplier=1.0),
    StressScenario(name="Spot +10%", spot_multiplier=1.1, vol_multiplier=1.0),
    StressScenario(name="Vol +20%", spot_multiplier=1.0, vol_multiplier=1.2),
    StressScenario(name="Spot -10%, Vol +20%", spot_multiplier=0.9, vol_multiplier=1.2),
]

result_scenarios = optimize_scenarios_worst_case(
    portfolio_params=portfolio_params,
    portfolio_types=portfolio_types,
    hedge_params=hedge_params,
    hedge_types=hedge_types,
    scenarios=scenarios,
    max_position=10.0,
)

x_opt_scen = result_scenarios["x"]
worst_loss = result_scenarios["worst_case_loss"]
scenario_losses = result_scenarios["scenario_losses"]

print("\n=== Scenario-based optimization (worst-case) ===")
print("Optimal hedge quantities (x):", x_opt_scen)
print("Worst-case loss:", worst_loss)

print("\nLoss per scenario (after hedging):")
for sc, loss in zip(scenarios, scenario_losses):
    print(f"{sc.name}: loss = {loss:.4f}")
