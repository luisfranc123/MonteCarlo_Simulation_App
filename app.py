# =================================================================
# Page configuration imports
# =================================================================
import streamlit as st
import pandas as pd
import numpy as np

from data import (yf_tickers, assets, build_portfolio, 
                  fetch_portfolio_data, fetch_dividend_yields)
from simulation import run_simulation
from analytics import compute_all_statistics, compute_asset_statistics
from charts import (distribution_charts, statistics_table, 
                    sharpe_comparison_chart)

# =================================================================
# Page configuration 
# =================================================================
st.set_page_config(
    page_title = "Monte Carlo Portfolio Simulator", 
    layout = 'wide', 
)
# -----------------------------------------------------------------
# Arin Brand Style 
# -----------------------------------------------------------------
st.markdown("""
<style>
  /* ── Arin Risk Advisors brand palette ── */
  :root{
    --bg: #0D2948; /* Deep Navy Blue — main background */
    --surf: #0a2038; /* Darker navy — card/surface background */
    --surf2: #071828; /* Deepest navy — sidebar background */
    --acc: #1B8FFB; /* Sky Blue — accents, links, highlights */
    --txt: #FFFFFF; /* White — primary text on navy */
    --mut: #80807F; /* Neutral Gray — secondary / muted text */
    --border: #1a3a5c; /* Navy tint — borders and dividers */
    --black: #313131; /* Brand black — used in tables/badges */
  }
  .stApp{background:var(--bg);color:var(--txt);
         font-family:'Calibri','Georgia',serif}
  header[data-testid="stHeader"]{background:var(--bg)!important}
  h1,h2,h3,h4,h5,h6{font-family:'Segoe UI','Arial',sans-serif}
  section[data-testid="stSidebar"]{background:var(--surf2)!important}
  .kpi{background:var(--surf);border-radius:10px;padding:14px 18px;
       border:1px solid var(--border);text-align:center}
  .kpi .lbl{font-size:11px;color:var(--mut);text-transform:uppercase;
             letter-spacing:.06em;margin-bottom:4px;
             font-family:'Segoe UI','Arial',sans-serif}
  .kpi .val{font-size:22px;font-weight:700;color:var(--acc)}
  .kpi .sub{font-size:11px;color:var(--mut);margin-top:2px}
  .sh{font-size:17px;font-weight:700;color:var(--acc);
      font-family:'Segoe UI','Arial',sans-serif;
      border-bottom:1px solid var(--border);
      padding-bottom:6px;margin:18px 0 12px}
  /* ── Sidebar text ── */
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div,
  section[data-testid="stSidebar"] .stSlider span,
  section[data-testid="stSidebar"] .stNumberInput label,
  section[data-testid="stSidebar"] .stToggle label{
    color: #FFFFFF !important;
  }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------
# Header 
# -----------------------------------------------------------------
st.title("Monte Carlo Portfolio Simulator")
st.caption("Arin Risk Advisors, LLC — Portfolio Construction Analysis")
st.divider()
# -----------------------------------------------------------------
# Fetch Data 
# -----------------------------------------------------------------
with st.spinner("Fetching market data from yfinance..."):
    market_data = fetch_portfolio_data()
    dividend_yields = fetch_dividend_yields()

# show warning for any assets using fallback values
portfolio_base = build_portfolio(market_data)
failed_assets = portfolio_base[portfolio_base["source"] == "fallback"]["name"].tolist()

if failed_assets:
    st.warning(
        f"The following assets could not be fetched from yfinance "
        f"and are using Carson reference values: "
        f"**{', '.join(failed_assets)}**"
    )
# -----------------------------------------------------------------
# Sidebar controls 
# -----------------------------------------------------------------
st.sidebar.header("Simulation Controls")
n_simulations = st.number_input(
    label = "Number of simulations", 
    min_value = 500, 
    max_value = 5_000, 
    value = 1_000,
    step = 500, 
    help = "More simulations = more stable results but slower runtime",    
)
n_months = st.number_input(
    label = "Time horizon (months)", 
    min_value = 60, 
    max_value = 360, 
    value = 120, 
    step = 12, 
    help = "120 months = 10 years. Each step adds one year.",     
)
risk_free_rate = st.sidebar.number_input(
    label = "Risk-free rate",
    min_value = 0.0,
    max_value = 0.10,
    value = 0.0225,
    step = 0.0025,
    format = "%.4f",
    help = "Annualised risk-free rate used in Sharpe ratio calculation.",
)
# -----------------------------------------------------------------
# Sidebar — Options Overlay Toggles 
# -----------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("Options Overlay")
st.sidebar.caption(
    "Toggle the collar strategy on or off for each hedged case. "
    "When off, the case uses dynamic beta returns with no overlay."
)
run_volatility = st.sidebar.toggle(
    label = "Volatility Managed Case", 
    value = True, 
    help = "Apply collar to all assets.", 
)
run_equity = st.sidebar.toggle(
    label = "Equity Hedged Case", 
    value = True, 
    help = "Apply collar to equity assets only.", 
)
# -----------------------------------------------------------------
# Portfolio editor
# -----------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("Portfolio Editor")
st.sidebar.caption(
    "Edit wights, expected returns, and volatility. "
    "Weights should sum to 100%."
)
# Build the editable DataFrame from live data
editor_df = portfolio_base[[
    "name", "weight", "exp_return", "volatility", "source"
]].copy()

# Convert volatility percentages for readibility in the editor
editor_df["weight"] = editor_df["weight"]*100
editor_df["exp_return"] = editor_df["exp_return"]*100
editor_df["volatility"] = editor_df["volatility"]*100

# rename columns for display
editor_df.columns = [
    "Asset", "Weight (%)", "Exp. Return (%)", "Volatility (%)", "Source"
]

# st.data_editor renders an editable table.
edited_df = st.sidebar.data_editor(
    editor_df, 
    use_container_width = True, 
    hide_index = True, 
    disabled = ["Asset", "Source"], 
    column_config = {
        "Weight (%)": st.column_config.NumberColumn(
            format = "%.2f", 
            min_value = 0.0, 
            max_value = 100.0, 
        ), 
        "Exp. Return (%)": st.column_config.NumberColumn(
            format = "%.2f"
        ), 
        "Volatility (%)": st.column_config.NumberColumn(
            format = "%.2f", 
            min_value = 0.0, 
        ), 
    }, 
)

# check if weights sum to 100% and warn if not
total_weight = edited_df["Weight (%)"].sum()
if not np.isclose(total_weight, 100.0, atol = 1.0):
    st.sidebar.warning(
        f"Weights sum to {total_weight:.2f}%. "
        f"They will be normalized to 100% before the simulation runs."
    )
# -----------------------------------------------------------------
# Rebuild portfolio from edited values
# -----------------------------------------------------------------
edited_portfolio = portfolio_base.copy()
edited_portfolio["weight"] = edited_df["Weight (%)"]/100
edited_portfolio["exp_return"] = edited_df["Exp. Return (%)"]/100
edited_portfolio["volatility"] = edited_df["Volatility (%)"]/100
# -----------------------------------------------------------------
# Run simulation button and results
# -----------------------------------------------------------------
col_btn, col_info = st.columns([1, 3])

with col_btn:
    run_button = st.button(
    label = "▶  Run Simulation", 
    type = "primary", 
    use_container_width = True, 
    )

with col_info:
    st.caption(
    f"**{n_simulations:,}** simulations · "
    f"**{n_months//12}**-year horizon · "
    f"Risk-free rate **{risk_free_rate:.2%}**"
    )
# -----------------------------------------------------------------
# Results
# -----------------------------------------------------------------
if run_button:

    with st.spinner("Running Monte Carlo simulation..."):
        results = run_simulation(
            portfolio = edited_portfolio, 
            corr_matrix = market_data["corr_matrix"], 
            dividend_yields = dividend_yields, 
            n_simulations = n_simulations, 
            n_months = n_months, 
            risk_free_rate = risk_free_rate, 
            use_overlay = {
                "volatility": run_volatility, 
                "equity": run_equity, 
            }, 
        )
    # --- Distribution chart ---
    st.subheader("Return Distribution")
    st.plotly_chart(
        distribution_charts(results, edited_portfolio, risk_free_rate), 
        use_container_width = True, 
    )
    st.divider()

    # --- Statistics table ---
    st.subheader("Portfolio Statistics")
    st.plotly_chart(
        statistics_table(results, edited_portfolio, risk_free_rate), 
        use_container_width = True, 
    )
    # --- Fallback notice at the bottom ---
    if failed_assets:
        st.info(
            "Assets using Carson reference values are marked "
            "'fallback' in the portfolio editor."
        )
    