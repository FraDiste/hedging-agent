import streamlit as st
import numpy as np
import pandas as pd
from openai import OpenAI
import os
import sys
import plotly.express as px

# Add project root to Python path so that "src" can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

client = OpenAI()

from src.pricing import BlackScholesParams, black_scholes_price
from src.greeks import all_greeks
from src.monte_carlo import simulate_gbm_paths
from src.risk_metrics import value_at_risk, expected_shortfall
from src.stress_testing import StressScenario, stress_test_option_black_scholes
from src.optimizer import optimize_greeks, optimize_scenarios_worst_case
from src.portfolio_parser import (
    parse_portfolio_text,
    build_bs_portfolio_from_parsed,
    PortfolioParseError,
    CONTROLLED_SYNTAX_DOC,
)

def ask_ai(
    user_question: str,
    context: str,
    focus: str = "general",
    detail_level: str = "Short summary",
    language: str = "English",
) -> str:
    """
    Call the OpenAI model to get an educational explanation.

    - user_question: what the user (or UI button) is asking
    - context: structured snapshot of the current portfolio / risk engine
    - focus: "general", "greeks", "risk_metrics", "hedging", "global_overview", "experiments"
    - detail_level: "Short summary" or "Deep dive"
    - language: "English" or "Italiano"
    """
    # Small helper for verbosity instructions
    if detail_level == "Short summary":
        detail_instruction = (
            "Keep the explanation concise: at most 3–4 short paragraphs. "
            "Prioritise intuition over formulas."
        )
    else:
        detail_instruction = (
            "You may go into more depth and use formulas or numerical examples "
            "when they genuinely help understanding, but still keep the answer readable."
        )

    system_prompt = (
        "You are an educational assistant specialised in derivatives, "
        "quantitative risk management and hedging.\n"
        "You ALWAYS base your explanations on the provided numerical context. "
        "If something is not contained in the context, say that you cannot know.\n"
        "You explain concepts step by step, using intuition and simple analogies. "
        "You avoid giving trading or investment recommendations.\n"
        "Your role is purely pedagogical: you interpret results from a risk engine, "
        "you do NOT suggest trades.\n"
        f"You must answer in {language}. {detail_instruction}"
    )

    user_prompt = (
        f"FOCUS: {focus}\n"
        f"DETAIL_LEVEL: {detail_level}\n"
        f"LANGUAGE: {language}\n\n"
        "Here is the current portfolio context from the risk engine:\n"
        f"{context}\n\n"
        "User question:\n"
        f"{user_question}"
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content

def safe_ask_ai(
    user_question: str,
    context: str,
    focus: str = "general",
    detail_level: str = "Short summary",
    language: str = "English",
) -> str:
    """
    Wrapper robusto per chiamare il modello OpenAI con:
    - lingua scelta (English / Deutsch),
    - livello di dettaglio (short / deep),
    - focus tematico (greeks, risk_metrics, hedging, experiments, ...),
    - gestione elegante degli errori.
    """

    # ----- Istruzione di lingua -----
    if language == "Deutsch":
        language_instruction = (
            "Antworten Sie auf Deutsch, klar, strukturiert und gut verständlich."
        )
    elif language == "English":
        language_instruction = (
            "Answer in clear, well-structured English, suitable for a master student."
        )
    else:
        language_instruction = f"Answer in {language}, clearly and well-structured."

    # ----- Istruzione di livello di dettaglio -----
    if detail_level == "Deep dive":
        detail_instruction = (
            "Provide a deep dive explanation: define key concepts, connect them to "
            "the numerical values in the context, and use equations or simple examples "
            "when helpful. The answer can be relatively long."
        )
    else:  # "Short summary"
        detail_instruction = (
            "Provide a concise summary: focus on the key ideas and how they relate to "
            "the numerical values in the context. Use short paragraphs or bullet points."
        )

    # ----- Istruzione di focus tematico -----
    if focus == "greeks":
        focus_instruction = (
            "Focus mainly on interpreting the portfolio Greeks (Delta, Gamma, Vega, "
            "Theta, Rho) and what they mean for directional risk, convexity, volatility "
            "exposure, time decay and interest-rate sensitivity."
        )
    elif focus == "risk_metrics":
        focus_instruction = (
            "Focus mainly on interpreting the Monte Carlo risk metrics VaR_99 and ES_99 "
            "and what they say about tail risk and the P&L distribution."
        )
    elif focus == "hedging":
        focus_instruction = (
            "Focus mainly on explaining the hedging layer(s): Greek-matching hedge and "
            "scenario-based worst-case hedge, and how they change the risk profile."
        )
    elif focus == "experiments":
        focus_instruction = (
            "Focus on suggesting experiments the user can perform in this app to better "
            "understand derivatives risk and hedging."
        )
    elif focus == "global_overview":
        focus_instruction = (
            "Provide a global overview of the portfolio risk profile, combining Greeks, "
            "VaR/ES and hedging results into a coherent story."
        )
    else:
        focus_instruction = (
            "Provide a general explanation using all relevant information from the "
            "context and the user question."
        )

    system_prompt = (
        "You are an educational assistant specialised in derivatives, quantitative "
        "risk management and hedging.\n"
        "You ALWAYS base your explanations on the numerical context provided. "
        "You never invent numbers or trades.\n"
        "You strictly avoid any investment or trading advice. You only explain, "
        "interpret and suggest safe educational experiments.\n"
        f"{language_instruction} {detail_instruction} {focus_instruction}"
    )

    user_content = (
        "PORTFOLIO_CONTEXT:\n"
        f"{context}\n\n"
        "USER_QUESTION:\n"
        f"{user_question}\n\n"
        "Remember: do NOT give investment advice or trading recommendations."
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content

    except Exception as e:
        # Fallback amichevole in caso di errore API
        return (
            "I’m sorry, I could not contact the AI explanation service right now. "
            "This is a technical error, not a problem with your portfolio.\n\n"
            f"Technical details (for debugging): {type(e).__name__}: {e}"
        )


def safe_ask_ai(
    user_question: str,
    context: str,
    focus: str = "general",
    detail_level: str = "Short summary",
    language: str = "English",
) -> str:
    """
    Wrapper around ask_ai that catches API errors and returns
    a friendly message instead of crashing the app.
    """
    try:
        return ask_ai(
            user_question=user_question,
            context=context,
            focus=focus,
            detail_level=detail_level,
            language=language,
        )
    except Exception as e:
        return (
            "The AI explanation service is temporarily unavailable. "
            "You can still use the quantitative risk engine; only the natural-language "
            f"explanations are affected.\n\n(Technical detail: {e})"
        )



# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def portfolio_greeks(
    portfolio_params: list[BlackScholesParams],
    portfolio_types: list[str],
    quantities: list[float],
) -> dict[str, float]:
    """
    Aggregate Greeks for the given portfolio, using signed quantities.

    - For vanilla options (call/put), we use closed-form Greeks from all_greeks.
    - For futures and forwards, we approximate:
        Delta = 1, all other Greeks = 0
      (the sign is carried by the position quantity).
    - For digital options we currently do NOT include Greeks, because in this
      prototype they are primarily used to enrich the P&L distribution and tail risk.
    """
    agg = {"price": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    for w, p, t in zip(quantities, portfolio_params, portfolio_types):
        t = t.lower()
        if t in ("call", "put"):
            g = all_greeks(p, option_type=t)
        elif t in ("future", "forward"):
            g = {"price": 0.0, "delta": 1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        else:
            # digital options and any other unsupported type: skip Greeks contribution
            continue

        for k in agg:
            agg[k] += w * g[k]
    return agg


def portfolio_mc_pnl_distribution(
    portfolio_params: list[BlackScholesParams],
    portfolio_types: list[str],
    quantities: list[float],
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: int | None = 123,
) -> np.ndarray:
    """
    Compute a Monte Carlo distribution of portfolio P&L, using quantities.

    Supported types:
    - call / put        : standard European options (Black–Scholes)
    - future            : payoff S_T - F0 (approx. zero initial value)
    - forward           : payoff S_T - K     (approx. zero initial value)
    - digital_call      : payoff 1{S_T > K}  (cash-or-nothing)
    - digital_put       : payoff 1{S_T < K}  (cash-or-nothing)

    For simplicity in this prototype, all non-vanilla instruments are assumed to
    start at (approximately) zero value, so we set current_price = 0 for those.
    This mostly affects the P&L level but not the tail shape we care about for VaR/ES.
    """
    if len(portfolio_params) == 0:
        return np.array([])

    base = portfolio_params[0]

    paths = simulate_gbm_paths(
        params=base,
        n_paths=n_paths,
        n_steps=n_steps,
        antithetic=True,
        seed=seed,
    )
    S_T = paths[:, -1]
    T = base.maturity
    r = base.risk_free_rate

    pnl_paths = np.zeros_like(S_T)

    for w, p, t in zip(quantities, portfolio_params, portfolio_types):
        t = t.lower()

        if t == "call":
            payoff = np.maximum(S_T - p.strike, 0.0)
            discounted_payoff = np.exp(-r * T) * payoff
            current_price = black_scholes_price(p, "call")

        elif t == "put":
            payoff = np.maximum(p.strike - S_T, 0.0)
            discounted_payoff = np.exp(-r * T) * payoff
            current_price = black_scholes_price(p, "put")

        elif t == "future":
            # payoff based on reference forward level (stored in p.strike)
            payoff = S_T - p.strike
            discounted_payoff = payoff  # futures are marked-to-market; ignore discount for simplicity
            current_price = 0.0

        elif t == "forward":
            # payoff vs. contract strike K = p.strike
            payoff = S_T - p.strike
            discounted_payoff = np.exp(-r * T) * payoff
            current_price = 0.0  # fair forwards start at ~0; we ignore any small PV

        elif t == "digital_call":
            payoff = (S_T > p.strike).astype(float)
            discounted_payoff = np.exp(-r * T) * payoff
            current_price = 0.0

        elif t == "digital_put":
            payoff = (S_T < p.strike).astype(float)
            discounted_payoff = np.exp(-r * T) * payoff
            current_price = 0.0

        else:
            # unsupported instrument types are ignored in P&L
            continue

        pnl_paths += w * (discounted_payoff - current_price)

    return pnl_paths


def build_default_scenarios() -> list[StressScenario]:
    return [
        StressScenario(name="Spot -10%", spot_multiplier=0.9, vol_multiplier=1.0),
        StressScenario(name="Spot +10%", spot_multiplier=1.1, vol_multiplier=1.0),
        StressScenario(name="Vol +20%", spot_multiplier=1.0, vol_multiplier=1.2),
        StressScenario(name="Spot -10%, Vol +20%", spot_multiplier=0.9, vol_multiplier=1.2),
    ]


def scenario_losses_for_portfolio(
    portfolio_params: list[BlackScholesParams],
    portfolio_types: list[str],
    quantities: list[float],
    scenarios: list[StressScenario],
) -> dict[str, float]:
    """
    Compute scenario P&L (without hedge) for a portfolio.
    Currently uses the stress_test_option_black_scholes function, and is
    therefore reliable for vanilla call/put instruments. Non-vanilla types
    should be treated with care or excluded from scenario-based optimisation.
    """
    losses: dict[str, float] = {}

    for sc in scenarios:
        pnl_portfolio = 0.0
        for w, p, t in zip(quantities, portfolio_params, portfolio_types):
            res = stress_test_option_black_scholes(p, t, [sc])[0]
            pnl_portfolio += w * res["pnl"]

        losses[sc.name] = -pnl_portfolio  # loss = - P&L
    return losses


# ---------------------------------------------------------------------
# Streamlit App Layout
# ---------------------------------------------------------------------


st.set_page_config(
    page_title="AI Hedging Agent",
    page_icon="📉",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
    <style>
    div[data-testid="stMarkdown"] {
        border-radius: 10px;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    [data-testid="stSidebar"] {
        min-width: 200px;
        max-width: 200px;
    }
    input, textarea, select {
        border-radius: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .stRadio > div {
        gap: 3px;
    }

    .stRadio label {
        padding: 2px 6px;          /* meno padding verticale */
        margin-top: 0px;
        margin-bottom: 0px;
        border-radius: 8px;
        display: inline-block;
    }

    .stRadio label:hover {
        background-color: #111827;
        color: #E5E7EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Sidebar branding
with st.sidebar:
    # Logo e titolo
    st.image("static/whu_logo.png", width=120)
    st.markdown("### AI Hedging Agent")

    # Separatore sottile
    st.markdown(
        """
        <hr style="height:1px;border:none;background-color:#3C4A63;margin:10px 0 16px 0;">
        """,
        unsafe_allow_html=True,
    )

    # Box "Navigation" elegante
    st.markdown(
        """
        <div style="
            padding:10px 12px 6px 12px;
            border-radius:12px;
            background-color:#1B2433;
            border:1px solid #2E3A4D;
            margin-bottom:4px;
        ">
            <p style="margin:0 0 4px 0; font-size:0.78rem; letter-spacing:0.14em;
                      text-transform:uppercase; color:#94A3B8;">
                Navigation
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Menu con icone
    mode = st.radio(
        label="",
        options=[
            "📘  Project overview & theory",
            "📊  Risk & Hedging Engine",
            "🤖  Finance Education Chatbot",
        ],
        index=0,
    )




# Custom thicker separator
st.markdown(
    """
    <hr style="height:3px;border:none;background-color:#2F3B52;margin-top:0.5rem;margin-bottom:2rem;">
    """,
    unsafe_allow_html=True,
)


# ===========================================================
# MODE 0: PROJECT OVERVIEW & THEORY
# ===========================================================

if mode.startswith("📘"):
    # Project overview & theory

    st.title("AI Hedging Agent")

    st.markdown(
        """
#### Derivatives-focused quantitative risk management system

This application is a **derivatives-focused quantitative risk management platform**.  
It combines a **derivatives pricing & hedging engine** with a **finance education assistant**.

Use the navigation on the left to move between the theoretical overview,  
the risk & hedging engine, and the educational chatbot.
"""
    )

    st.markdown(
        "<hr style='margin-top:0.75rem; margin-bottom:1.5rem; border:0; "
        "border-top:2px solid #1f2937;'>",
        unsafe_allow_html=True,
    )
    # Project overview & theory
    st.subheader("Project overview & theoretical background")

    st.markdown(r"""
This project is centered on **financial derivatives** and their quantitative treatment.  
It provides a full workflow for:

- pricing derivatives (options, futures, forwards, digital options),
- computing **Greeks** for sensitivity analysis,
- generating **Monte Carlo P&L distributions**,
- evaluating **tail risk** (VaR & Expected Shortfall),
- performing **stress testing** on market parameters,
- and computing **optimal hedges** through convex optimisation.

At the same time, the application integrates a natural-language assistant designed  
to explain derivatives-related risk concepts in a clear, educational way.

---

### Theoretical foundations

**Black–Scholes model**

The underlying asset price $S_t$ is modelled as a Geometric Brownian Motion:
""")

    st.latex(r"dS_t = \mu S_t \, dt + \sigma S_t \, dW_t")

    st.markdown(r"""
where $\mu$ is the drift, $\sigma$ the volatility and $W_t$ a standard Brownian motion.
Under the risk-neutral measure, the drift becomes the risk-free rate $r$, and we obtain
closed-form prices for European calls and puts.

**Greeks**

Greeks are partial derivatives of the option price with respect to underlying parameters:

- **Delta**: sensitivity to the underlying price  
- **Gamma**: curvature of the price with respect to the underlying  
- **Vega**: sensitivity to volatility  
- **Theta**: sensitivity to the passage of time  
- **Rho**: sensitivity to the risk-free rate  

The application aggregates Greeks **at portfolio level** by summing the contributions
of each position, multiplied by its quantity and sign (long/short).

**Monte Carlo P&L distribution**

For a given portfolio, we simulate many future paths for the underlying price using GBM,
compute the discounted payoff of each instrument along each path, and subtract the current
instrument prices when relevant. This yields a distribution of **profit and loss (P&L)**
for the portfolio.

From this empirical distribution we estimate:

- **VaR 99%**: the 1st percentile of the P&L distribution (a loss threshold)  
- **ES 99%**: the average of P&L values below the VaR threshold (tail risk)

**Stress testing**

In addition to Monte Carlo, we apply deterministic shocks to:

- the spot price (e.g. *Spot –10%*, *Spot +10%*),  
- the volatility (e.g. *Vol +20%*),  
- or both simultaneously.

For each scenario we recompute option prices and obtain the **P&L under stress**.

**Hedging via convex optimisation**

The engine introduces a small set of standard hedge instruments (e.g. OTM call and put).
We then solve two optimisation problems:

1. **Greek-matching hedge**  
   - Decision variables: hedge quantities  
   - Objective: minimise the squared distance between target Greeks (e.g. Delta=0, Gamma=0)
     and the portfolio Greeks after adding the hedge  
   - Constraints: position limits and optional budget constraint  

2. **Scenario-based worst-case hedge**  
   - Decision variables: hedge quantities  
   - Objective: minimise the **maximum loss** across all predefined stress scenarios  
   - This is formulated as a convex optimisation problem by introducing an auxiliary
     variable representing the worst-case loss.

All optimisations are implemented with `cvxpy`, which allows us to keep the formulation close
to the mathematical problem.

---

### Role of AI in this prototype

The **AI component** is not used to generate trading signals.  
Instead, it plays a **pedagogical role**:

- it receives a structured summary of the current portfolio, Greeks, VaR/ES and hedge results,  
- it generates natural-language explanations tailored to non-expert users,  
- it explicitly avoids any investment recommendations.

This design keeps the project aligned with **ethical and responsible use of AI in finance**.
""")



# ===========================================================
# MODE 1: RISK & HEDGING ENGINE
# ===========================================================

elif mode.startswith("📊"):
    # Risk & Hedging Engine

    st.title("Risk & Hedging Engine")

    st.markdown(
        """
This section implements the **core derivatives risk engine** of the AI Hedging Agent.  
It prices and aggregates a portfolio of **derivative instruments**, computes **Greeks**,  
simulates **Monte Carlo P&L**, and runs **hedging optimisation** in stress scenarios.
"""
    )

    st.markdown(
        "<hr style='margin-top:0.75rem; margin-bottom:1.5rem; border:0; "
        "border-top:2px solid #1f2937;'>",
        unsafe_allow_html=True,
    )

    # NEW: quick guide for non-expert users
    st.markdown(
        """
**How to read this page**

1. Define your **derivatives portfolio** in the text box below (calls, puts, futures, forwards, digitals).  
2. Set the current **market inputs**: spot price, risk-free rate and volatility.  
3. Look at the **portfolio Greeks** and the **Monte Carlo VaR / ES** to understand sensitivities and tail risk.  
4. Inspect the **P&L distribution histogram** to see the full range of simulated outcomes.  
5. Optionally run the **Greek-matching hedge** and the **scenario-based hedge** to see how the risk profile changes.  
6. Switch to the **Finance Education Chatbot** tab if you want a plain-English explanation of the results.
"""
    )

    st.markdown(
        """
This section demonstrates the core quantitative capabilities of the AI Hedging Agent:

1. Portfolio input via a simple text syntax  
2. Pricing and Greeks  
3. Monte Carlo P&L distribution, VaR and Expected Shortfall  
4. Greek-matching hedge optimisation  
5. Scenario-based worst-case hedging
"""
    )


    # -------------------- Portfolio input --------------------
    st.markdown("### 1. Portfolio input (controlled syntax)")

    # Example portfolios for quick demos
    example_portfolios = {
        "Custom (free text)": """Long 1 call strike 100 maturity 1.0
Short 1 put strike 90 maturity 0.5""",
        "Directional bullish (long calls + future)": """Long 2 call strike 100 maturity 1.0
Long 1 future maturity 1.0""",
        "Short volatility (short straddle)": """Short 1 call strike 100 maturity 0.5
Short 1 put strike 100 maturity 0.5""",
        "Forward + hedge with options": """Long 1 forward strike 100 maturity 1.0
Short 1 call strike 105 maturity 1.0
Short 1 put strike 95 maturity 1.0""",
        "Digital payoff around strike": """Long 2 digital_call strike 100 maturity 0.5
Long 2 digital_put strike 100 maturity 0.5""",
    }

    selected_example = st.selectbox(
        "Load example portfolio:",
        options=list(example_portfolios.keys()),
        index=0,
        help="Select a template or keep 'Custom' to type your own positions.",
    )

    default_text = example_portfolios[selected_example]

    portfolio_text = st.text_area(
        "Enter your derivatives portfolio by following the same order of the example:",
        value=default_text,
        height=140,
        help=CONTROLLED_SYNTAX_DOC,
    )

    col_mkt1, col_mkt2, col_mkt3 = st.columns(3)
    with col_mkt1:
        spot = st.number_input(
            "Spot price",
            value=100.0,
            min_value=0.01,
            help="Current underlying price used for all derivatives in the portfolio.",
        )
    with col_mkt2:
        r = st.number_input(
            "Risk-free rate (continuous)",
            value=0.02,
            format="%.4f",
            help="Annualised continuously-compounded risk-free rate (e.g. 0.02 = 2% per year).",
        )
    with col_mkt3:
        sigma = st.number_input(
            "Volatility",
            value=0.20,
            min_value=0.0001,
            format="%.4f",
            help="Annualised volatility of the underlying (e.g. 0.20 = 20% per year).",
        )


    q = 0.0  # dividend yield, keep fixed for now

    try:
        parsed_positions = parse_portfolio_text(portfolio_text)
        if not parsed_positions:
            st.warning("No positions parsed. Please enter at least one line.")
            st.stop()

        portfolio_params, portfolio_types, quantities = build_bs_portfolio_from_parsed(
            parsed_positions,
            spot=spot,
            risk_free_rate=r,
            volatility=sigma,
            dividend_yield=q,
        )
    except PortfolioParseError as e:
        st.error(f"Error parsing portfolio:\n\n{e}")
        st.stop()

    df_portfolio = pd.DataFrame(
        [
            {
                "Side": pos.side,
                "Quantity": pos.quantity,
                "Type": pos.option_type.upper(),
                "Strike": pos.strike,
                "Maturity (years)": pos.maturity_years,
            }
            for pos in parsed_positions
        ]
    )
    st.dataframe(df_portfolio, use_container_width=True)

    non_option_types = sorted(
        {t for t in portfolio_types if t.lower() not in ("call", "put")}
    )

    # -------------------------------------------------------
    # 2. Greeks and VaR/ES
    # -------------------------------------------------------

    st.markdown("### 2. Portfolio Greeks and risk metrics")

    greeks_before = portfolio_greeks(portfolio_params, portfolio_types, quantities)

    g_price = round(greeks_before["price"], 4)
    g_delta = round(greeks_before["delta"], 4)
    g_gamma = round(greeks_before["gamma"], 6)
    g_vega = round(greeks_before["vega"], 4)
    g_theta = round(greeks_before["theta"], 4)
    g_rho = round(greeks_before["rho"], 4)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="
                background-color:#1E293B;
                border-radius:14px;
                padding:18px 18px 8px 18px;
                color:#E2E8F0;
                box-shadow:0 0 12px rgba(15,23,42,0.6);
            ">
            <h4 style="margin-top:0; margin-bottom:10px;">Greeks summary (approx.)</h4>
            <p style="margin-bottom:4px;"><b>Price:</b> {g_price:.4f}</p>
            <p style="margin-bottom:4px;"><b>Delta:</b> {g_delta:.4f}</p>
            <p style="margin-bottom:4px;"><b>Gamma:</b> {g_gamma:.6f}</p>
            <p style="margin-bottom:4px;"><b>Vega:</b> {g_vega:.4f}</p>
            <p style="margin-bottom:4px;"><b>Theta:</b> {g_theta:.4f}</p>
            <p style="margin-bottom:4px;"><b>Rho:</b> {g_rho:.4f}</p>
            <p style="margin-top:10px; font-size:0.8rem; color:#94A3B8;">
                Futures and forwards contribute mainly to Delta; digital options are
                included in P&L but not in the Greek figures.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        pnl_paths = portfolio_mc_pnl_distribution(
            portfolio_params, portfolio_types, quantities
        )
        st.markdown("**Monte Carlo VaR / ES (99%)**")
        if pnl_paths.size > 0:
            var_99 = value_at_risk(pnl_paths, alpha=0.99)
            es_99 = expected_shortfall(pnl_paths, alpha=0.99)

            var_99_r = round(var_99, 2)
            es_99_r = round(es_99, 2)

            st.markdown(
                f"""
                <div style="
                    background-color:#1E293B;
                    border-radius:14px;
                    padding:18px 18px 8px 18px;
                    color:#E2E8F0;
                    box-shadow:0 0 12px rgba(15,23,42,0.6);
                ">
                <h4 style="margin-top:0; margin-bottom:10px;">Risk summary (99%)</h4>
                <p style="margin-bottom:4px;"><b>VaR 99%:</b> {var_99_r:.2f}</p>
                <p style="margin-bottom:4px;"><b>ES 99%:</b> {es_99_r:.2f}</p>
                <p style="margin-top:10px; font-size:0.85rem; color:#94A3B8;">
                    Based on the Monte Carlo P&amp;L distribution for the current portfolio.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.session_state["last_portfolio_greeks"] = greeks_before
            st.session_state["last_var_99"] = float(var_99)
            st.session_state["last_es_99"] = float(es_99)
        else:
            st.info("No P&L paths available.")

    if pnl_paths.size > 0:
        st.markdown("**P&L distribution (Monte Carlo)**")
        fig = px.histogram(
            pnl_paths,
            nbins=40,
            title="Portfolio P&L distribution (Monte Carlo)",
            labels={"value": "P&L"},
        )
        fig.update_layout(
            bargap=0.05,
            template="simple_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Educational explanation (static, no API cost)
        with st.expander("How to interpret these risk metrics?"):
            st.markdown(
                """
- **Delta** measures how sensitive the portfolio value is to small moves in the underlying price  
  (approximate change in value for a 1-unit move in the underlying).

- **Gamma** measures how quickly Delta itself changes when the underlying moves  
  (large Gamma = Delta can change a lot in a short time).

- **Vega** measures sensitivity to volatility: how much the portfolio value changes if implied volatility moves.

- **Theta** is the time-decay of the portfolio: how much value you lose (or gain) per day just because time passes.

- **Rho** measures sensitivity to the risk-free rate.

- **VaR 99%** is a loss threshold: under the model and simulation assumptions, in 99% of cases
  you expect not to lose more than this amount.

- **ES 99%** (Expected Shortfall) is the **average** loss in the worst 1% of cases;  
  it is more conservative and looks at the whole tail of the loss distribution.
"""
            )

    # -------------------------------------------------------
    # 3. Greek-matching optimisation
    # -------------------------------------------------------

    st.markdown("### 3. Greek-matching hedge optimisation")

    if non_option_types:
        st.info(
            "Greek-matching optimisation is available only for pure vanilla option "
            "portfolios (calls and puts). Your portfolio also contains: "
            + ", ".join(non_option_types)
            + ". Please remove these instruments if you want to run this optimisation."
        )
    else:
        st.markdown(
            """
We now add two hedge instruments (one OTM call and one OTM put) and solve a
convex optimisation problem that minimises the distance between the current
Greeks and a target of **zero Delta and Gamma**.
"""
        )

        hedge_params = [
            BlackScholesParams(
                spot=spot,
                strike=spot * 1.10,
                maturity=1.0,
                risk_free_rate=r,
                volatility=sigma,
                dividend_yield=q,
            ),
            BlackScholesParams(
                spot=spot,
                strike=spot * 0.90,
                maturity=1.0,
                risk_free_rate=r,
                volatility=sigma,
                dividend_yield=q,
            ),
        ]
        hedge_types = ["call", "put"]

        if st.button("Run Greek-matching optimisation"):
            target_greeks = {"delta": 0.0, "gamma": 0.0}

            opt_result = optimize_greeks(
                portfolio_params=portfolio_params,
                portfolio_types=portfolio_types,
                hedge_params=hedge_params,
                hedge_types=hedge_types,
                target_greeks=target_greeks,
                portfolio_quantities=quantities,
                max_position=50.0,
                budget=None,
            )

            x_opt = opt_result["x"]
            greeks_after = opt_result["greeks_after"]

            st.markdown("**Optimal hedge quantities (per unit notional):**")
            hedge_desc = [
                f"Hedge 1: CALL K={hedge_params[0].strike:.2f}",
                f"Hedge 2: PUT  K={hedge_params[1].strike:.2f}",
            ]
            st.write(
                {
                    hedge_desc[i]: float(round(x_opt[i], 4))
                    for i in range(len(x_opt))
                }
            )

            st.markdown("**Greeks after hedging (approx. neutral):**")
            st.write(
                {
                    "Delta": round(greeks_after[0], 4),
                    "Gamma": round(greeks_after[1], 6),
                    "Vega": round(greeks_after[2], 4),
                    "Theta": round(greeks_after[3], 4),
                    "Rho": round(greeks_after[4], 4),
                }
            )

            st.session_state["last_greek_hedge"] = x_opt.tolist()
            st.session_state["last_greeks_after_hedge"] = greeks_after.tolist()

    # -------------------------------------------------------
    # 4. Scenario-based worst-case optimisation
    # -------------------------------------------------------

    st.markdown("### 4. Scenario-based worst-case hedging")

    if non_option_types:
        st.info(
            "Scenario-based worst-case optimisation is currently implemented only "
            "for portfolios of vanilla options (calls and puts). To run this step, "
            "please restrict the portfolio to calls and puts only."
        )
    else:
        st.markdown(
            """
We now consider several stress scenarios (spot down/up, volatility spike,
combined move) and compute a hedge that **minimises the worst-case loss**
across all scenarios.
"""
        )

        scenarios = build_default_scenarios()

        if st.button("Run scenario-based optimisation"):
            scen_result = optimize_scenarios_worst_case(
                portfolio_params=portfolio_params,
                portfolio_types=portfolio_types,
                hedge_params=hedge_params,
                hedge_types=hedge_types,
                scenarios=scenarios,
                portfolio_quantities=quantities,
                max_position=50.0,
            )

            x_opt_scen = scen_result["x"]
            worst_loss = scen_result["worst_case_loss"]
            scenario_losses = scen_result["scenario_losses"]

            st.markdown("**Optimal hedge quantities (scenario-based):**")
            hedge_desc = [
                f"Hedge 1: CALL K={hedge_params[0].strike:.2f}",
                f"Hedge 2: PUT  K={hedge_params[1].strike:.2f}",
            ]
            st.write(
                {
                    hedge_desc[i]: float(round(x_opt_scen[i], 4))
                    for i in range(len(x_opt_scen))
                }
            )

            st.markdown(
                f"""
                <div style="
                    background-color:#1E293B;
                    border-radius:14px;
                    padding:18px 18px 10px 18px;
                    color:#E2E8F0;
                    margin-top:10px;
                    box-shadow:0 0 12px rgba(15,23,42,0.6);
                ">
                <h4 style="margin-top:0; margin-bottom:10px;">Worst-case scenario loss</h4>
                <p style="font-size:1.1rem; margin-bottom:6px;">
                    <b>{worst_loss:.4f}</b>
                </p>
                <p style="margin-top:6px; font-size:0.85rem; color:#94A3B8;">
                    This is the loss in the most adverse stress scenario, given the current
                    scenario-based hedge.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Loss per scenario (after hedging):**")
            loss_table = pd.DataFrame(
                {
                    "Scenario": [sc.name for sc in scenarios],
                    "Loss": [round(l, 4) for l in scenario_losses],
                }
            )
            st.dataframe(loss_table, use_container_width=True)

            st.session_state["last_scenario_hedge"] = x_opt_scen.tolist()
            st.session_state["last_worst_case_loss"] = float(worst_loss)
            st.session_state["last_scenario_losses"] = [
                float(l) for l in scenario_losses
            ]

# ===========================================================
# MODE 2: FINANCE EDUCATION CHATBOT
# ===========================================================

elif mode.startswith("🤖"):
    # Finance Education Chatbot

    st.title("Finance Education Chatbot")

    st.markdown(
        """
This chatbot is designed to **explain derivatives risk and hedging results**  
generated by the AI Hedging Agent.

It focuses on:
- option and derivative pricing,
- portfolio **Greeks** (Delta, Gamma, Vega, Theta, Rho),
- **VaR / ES** from Monte Carlo simulation,
- **stress tests** and **hedging strategies**.

It is purely **educational** and does *not* provide investment advice.
"""
    )

    st.markdown(
        "<hr style='margin-top:0.75rem; margin-bottom:1.5rem; border:0; "
        "border-top:2px solid #1f2937;'>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------
    # Global explanation settings (language + detail level)
    # -------------------------------------------------------
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        language = st.selectbox(
            "Explanation language",
            ["English", "Deutsch"],
            index=0,
        )
    with col_set2:
        detail_level = st.radio(
            "Detail level",
            ["Short summary", "Deep dive"],
            index=0,
            horizontal=True,
        )

    # -------------------------------------------------------
    # Build structured context string from session state
    # -------------------------------------------------------
    context_lines: list[str] = []

    if "last_portfolio_greeks" in st.session_state:
        context_lines.append(
            f"GREEKS_BEFORE = {st.session_state['last_portfolio_greeks']}"
        )
    if "last_var_99" in st.session_state and "last_es_99" in st.session_state:
        context_lines.append(
            f"RISK_METRICS = {{'VaR_99': {st.session_state['last_var_99']}, "
            f"'ES_99': {st.session_state['last_es_99']}}}"
        )
    if "last_greek_hedge" in st.session_state:
        context_lines.append(
            f"GREEK_HEDGE_QUANTITIES = {st.session_state['last_greek_hedge']}"
        )
    if "last_greeks_after_hedge" in st.session_state:
        context_lines.append(
            f"GREEKS_AFTER_HEDGE = {st.session_state['last_greeks_after_hedge']}"
        )
    if "last_scenario_hedge" in st.session_state:
        context_lines.append(
            f"SCENARIO_HEDGE_QUANTITIES = {st.session_state['last_scenario_hedge']}"
        )
    if "last_worst_case_loss" in st.session_state:
        context_lines.append(
            f"WORST_CASE_SCENARIO_LOSS = {st.session_state['last_worst_case_loss']}"
        )
    if "last_scenario_losses" in st.session_state:
        context_lines.append(
            f"SCENARIO_LOSSES_VECTOR = {st.session_state['last_scenario_losses']}"
        )

    context_text = (
        "\n".join(context_lines) if context_lines else "No portfolio context is available yet."
    )

    # -------------------------------------------------------
    # Quick questions (one-click) - sopra la text area
    # -------------------------------------------------------
    st.markdown("**Quick questions (one-click):**")

    col_qa1, col_qa2, col_qa3, col_qa4 = st.columns(4)

    with col_qa1:
        explain_greeks_button = st.button("Explain my Greeks")

    with col_qa2:
        explain_var_es_button = st.button("Explain my VaR & ES")

    with col_qa3:
        explain_hedges_button = st.button("Explain my hedges")

    with col_qa4:
        explain_portfolio_button = st.button("Explain my portfolio")

    st.markdown("")
    experiments_button = st.button("Suggest experiments with this portfolio")

    st.markdown("---")

    # -------------------------------------------------------
    # Domanda libera dell'utente (sotto i bottoni)
    # -------------------------------------------------------
    user_question = st.text_area(
        "Your question:",
        height=120,
        placeholder="For example: 'What does my Delta mean?' or 'Why is my VaR so large?'",
    )

    ask_button = st.button("Ask the assistant", type="primary")

    # -------------------------------------------------------
    # 1) Domanda manuale dell'utente
    # -------------------------------------------------------
    if ask_button and user_question.strip():
        with st.spinner("Thinking..."):
            answer = safe_ask_ai(
                user_question=user_question.strip(),
                context=context_text,
                focus="general",
                detail_level=detail_level,
                language=language,
            )

        st.markdown("### Assistant's explanation")
        st.write(answer)

    # -------------------------------------------------------
    # 2) Bottoni automatici (quick questions + experiments)
    # -------------------------------------------------------
    elif (
        explain_portfolio_button
        or explain_greeks_button
        or explain_var_es_button
        or explain_hedges_button
        or experiments_button
    ):
        # Costruiamo la "domanda" automatica specifica
        if explain_portfolio_button:
            auto_question = (
                "Please provide a structured explanation of this portfolio, including:\n"
                "- interpretation of Delta, Gamma, Vega, Theta and Rho,\n"
                "- interpretation of the VaR_99 and ES_99 figures,\n"
                "- how the Greek-matching hedge changes the risk profile,\n"
                "- how the scenario-based hedge affects the worst-case loss,\n"
                "- intuitive analogies for a non-expert.\n"
                "Do NOT give any investment advice."
            )
            focus = "global_overview"
            title = "Portfolio risk explanation"

        elif explain_greeks_button:
            auto_question = (
                "Please explain ONLY the portfolio Greeks "
                "(Delta, Gamma, Vega, Theta, Rho) in intuitive terms so that a non-expert "
                "can understand directional risk, convexity, volatility exposure, time-decay "
                "and interest-rate sensitivity. Do NOT give any investment advice."
            )
            focus = "greeks"
            title = "Explanation of portfolio Greeks"

        elif explain_var_es_button:
            auto_question = (
                "Please explain ONLY the Monte Carlo risk measures VaR_99 and ES_99. "
                "Explain what they mean, how to interpret their values, and what the shape "
                "of the P&L distribution implies about tail risk. Use intuitive language and "
                "simple analogies. Do NOT give any investment advice."
            )
            focus = "risk_metrics"
            title = "Explanation of VaR and Expected Shortfall"

        elif explain_hedges_button:
            auto_question = (
                "Please explain the two hedging layers of this agent:\n"
                "- the Greek-matching hedge (how it moves Delta/Gamma closer to 0),\n"
                "- the scenario-based worst-case hedge (how it reduces the worst-case loss).\n"
                "Describe in intuitive terms how these hedges change the risk profile, "
                "without giving any investment advice."
            )
            focus = "hedging"
            title = "Explanation of hedging strategies"

        else:  # experiments_button
            auto_question = (
                "Suggest 3 concrete experiments the user could run in this app, such as:\n"
                "- changing volatility,\n"
                "- changing strikes or maturities,\n"
                "- modifying hedge sizes.\n"
                "For each experiment, explain in a few lines what risk aspect it illustrates. "
                "Do NOT give trading recommendations; focus only on learning."
            )
            focus = "experiments"
            title = "Suggested experiments with this portfolio"

        with st.spinner("Thinking..."):
            answer = safe_ask_ai(
                user_question=auto_question,
                context=context_text,
                focus=focus,
                detail_level=detail_level,
                language=language,
            )

        st.markdown(f"### {title}")
        st.write(answer)
