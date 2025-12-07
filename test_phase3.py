from src.pricing import BlackScholesParams
from src.stress_testing import (
    StressScenario,
    stress_test_option_black_scholes,
    stress_test_linear_portfolio,
)
import numpy as np

print("Running test_phase3...")

# === 1) Option stress test ===
params = BlackScholesParams(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    risk_free_rate=0.02,
    volatility=0.20,
    dividend_yield=0.0,
)

scenarios = [
    StressScenario(name="Spot -10%", spot_multiplier=0.9, vol_multiplier=1.0),
    StressScenario(name="Spot +10%", spot_multiplier=1.1, vol_multiplier=1.0),
    StressScenario(name="Vol +20%", spot_multiplier=1.0, vol_multiplier=1.2),
    StressScenario(name="Spot -10%, Vol +20%", spot_multiplier=0.9, vol_multiplier=1.2),
]

results = stress_test_option_black_scholes(
    params=params,
    option_type="call",
    scenarios=scenarios,
)

print("\nOption stress test (call):")
for res in results:
    print(
        f"{res['name']}: shocked price = {res['price_shocked']:.4f}, "
        f"PnL = {res['pnl']:.4f}"
    )

# === 2) Linear portfolio stress test ===
spot_vector = np.array([100.0, 50.0])       # e.g. two stocks
position_vector = np.array([10.0, -20.0])   # long 10 of first, short 20 of second

spot_shocks = np.array([
    [-0.10, -0.10],   # both -10%
    [-0.10,  0.05],   # first -10%, second +5%
    [ 0.05,  0.05],   # both +5%
])

pnl_scenarios = stress_test_linear_portfolio(
    spot_vector=spot_vector,
    position_vector=position_vector,
    spot_shocks=spot_shocks,
)

print("\nLinear portfolio stress test (P&L per scenario):")
for i, pnl in enumerate(pnl_scenarios):
    print(f"Scenario {i+1}: P&L = {pnl:.2f}")
