print("Running test_phase1...")

from src.pricing import BlackScholesParams, black_scholes_call, black_scholes_put
from src.greeks import all_greeks

params = BlackScholesParams(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    risk_free_rate=0.02,
    volatility=0.20,
    dividend_yield=0.0,
)

call_price = black_scholes_call(params)
put_price = black_scholes_put(params)
greeks_call = all_greeks(params, option_type="call")

print("Call price:", call_price)
print("Put price :", put_price)
print("Call greeks:", greeks_call)
