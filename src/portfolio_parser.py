from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math

from src.pricing import BlackScholesParams


CONTROLLED_SYNTAX_DOC = """
Controlled syntax for defining a portfolio.

Each line describes ONE position with the following grammar:

    <Side> <Quantity> <InstrumentType> [strike <K>] maturity <T>

Where:
- <Side>           = Long | Short
- <Quantity>       = positive number (can be fractional)
- <InstrumentType> = call | put | future | forward | digital_call | digital_put
- <K>              = strike (required for call, put, forward, digital_*)
- <T>              = time-to-maturity in years (e.g. 0.5, 1.0, 2.0)

Examples:
- Long 1 call strike 100 maturity 1.0
- Short 2 put strike 90 maturity 0.5
- Long 3 future maturity 0.75
- Short 1 forward strike 101 maturity 1.0
- Long 2 digital_call strike 95 maturity 0.5
- Short 1 digital_put strike 90 maturity 0.25

Notes:
- Futures are assumed to start today with zero initial value; we internally use the
  fair forward price as a reference level.
- Forwards and digital options currently receive only P&L and VaR/ES treatment
  (they are NOT used as hedge instruments in optimisation).
"""


class PortfolioParseError(Exception):
    """Raised when the portfolio text cannot be parsed."""


@dataclass
class ParsedPosition:
    side: str                # "Long" or "Short"
    quantity: float
    option_type: str         # "call", "put", "future", "forward", "digital_call", "digital_put"
    strike: float | None     # None for pure futures
    maturity_years: float


def parse_portfolio_text(text: str) -> List[ParsedPosition]:
    """
    Parse a multi-line text portfolio description into a list of ParsedPosition.
    """
    positions: List[ParsedPosition] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for i, line in enumerate(lines, start=1):
        tokens = line.split()
        if len(tokens) < 4:
            raise PortfolioParseError(
                f"Line {i}: too few tokens. Expected at least "
                "'<Side> <Quantity> <InstrumentType> ...'.\nGot: '{line}'"
            )

        side_raw = tokens[0].lower()
        if side_raw not in ("long", "short"):
            raise PortfolioParseError(
                f"Line {i}: invalid side '{tokens[0]}'. Use 'Long' or 'Short'."
            )
        side = "Long" if side_raw == "long" else "Short"

        try:
            quantity = float(tokens[1])
        except ValueError:
            raise PortfolioParseError(
                f"Line {i}: could not parse quantity '{tokens[1]}' as a number."
            )

        inst_type = tokens[2].lower()
        allowed_types = {"call", "put", "future", "forward", "digital_call", "digital_put"}
        if inst_type not in allowed_types:
            raise PortfolioParseError(
                f"Line {i}: unsupported instrument type '{tokens[2]}'. "
                f"Supported: {', '.join(sorted(allowed_types))}."
            )

        strike: float | None = None
        maturity: float | None = None

        # parse optional "strike K" and required "maturity T"
        idx = 3
        while idx < len(tokens):
            tok = tokens[idx].lower()
            if tok == "strike":
                if idx + 1 >= len(tokens):
                    raise PortfolioParseError(
                        f"Line {i}: 'strike' keyword without value."
                    )
                try:
                    strike = float(tokens[idx + 1])
                except ValueError:
                    raise PortfolioParseError(
                        f"Line {i}: could not parse strike '{tokens[idx + 1]}' as a number."
                    )
                idx += 2
            elif tok == "maturity":
                if idx + 1 >= len(tokens):
                    raise PortfolioParseError(
                        f"Line {i}: 'maturity' keyword without value."
                    )
                try:
                    maturity = float(tokens[idx + 1])
                except ValueError:
                    raise PortfolioParseError(
                        f"Line {i}: could not parse maturity '{tokens[idx + 1]}' as a number."
                    )
                idx += 2
            else:
                raise PortfolioParseError(
                    f"Line {i}: unexpected token '{tokens[idx]}'. "
                    "Expected 'strike' or 'maturity'."
                )

        if maturity is None:
            raise PortfolioParseError(
                f"Line {i}: maturity is missing. Use 'maturity <T>' (in years)."
            )

        # Instruments that REQUIRE a strike
        if inst_type in {"call", "put", "forward", "digital_call", "digital_put"} and strike is None:
            raise PortfolioParseError(
                f"Line {i}: instrument type '{inst_type}' requires 'strike <K>'."
            )

        # Futures can omit strike: we will derive a fair forward internally
        pos = ParsedPosition(
            side=side,
            quantity=quantity,
            option_type=inst_type,
            strike=strike,
            maturity_years=maturity,
        )
        positions.append(pos)

    return positions


def build_bs_portfolio_from_parsed(
    parsed_positions: List[ParsedPosition],
    spot: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
) -> Tuple[list[BlackScholesParams], list[str], list[float]]:
    """
    Convert parsed positions into:
      - list of BlackScholesParams
      - list of instrument types
      - list of signed quantities (long > 0, short < 0)

    For futures, we internally set the 'strike' to the fair forward price
    S0 * exp((r - q) * T). For forwards and digitals we keep the user strike.
    """
    params_list: list[BlackScholesParams] = []
    types: list[str] = []
    quantities: list[float] = []

    for pos in parsed_positions:
        signed_quantity = pos.quantity if pos.side == "Long" else -pos.quantity
        inst_type = pos.option_type.lower()
        T = pos.maturity_years

        if inst_type == "future":
            # use fair forward as reference level
            fair_forward = spot * math.exp((risk_free_rate - dividend_yield) * T)
            strike = fair_forward
        else:
            strike = pos.strike if pos.strike is not None else 0.0

        params = BlackScholesParams(
            spot=spot,
            strike=strike,
            maturity=T,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
        )
        params_list.append(params)
        types.append(inst_type)
        quantities.append(signed_quantity)

    return params_list, types, quantities
