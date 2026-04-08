# =================================================================
# Plotly visualisations for the Monte Carlo simulation app.#1B8FFB
# 1. Libraries and depdndencies
# =================================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from analytics import (weighted_portfolio_returns,
                        compute_all_statistics)

case_colours = {
    "Base Case": "#FFFFFF",   
    "Dynamic Beta Case": "#1B8FFB",   
    "Volatility Managed Case": "#80807F",   
    "Equity Hedged Case": "#A8C8F0",   
}
case_order = [
    "Base Case",
    "Dynamic Beta Case",
    "Volatility Managed Case",
    "Equity Hedged Case",
]

case_styles = {
    "Base Case": "solid",
    "Dynamic Beta Case": "solid",
    "Volatility Managed Case": "dash",
    "Equity Hedged Case": "dot",
}

# =================================================================
# 2. The Distribution Chart
# =================================================================
def distribution_charts(simulation_results: dict,
                        portfolio: pd.DataFrame,
                        risk_free_rate: float) -> go.Figure:
    """
    Builds a Plotly figure showing relative frequency histograms
    for all four simulation cases overlaid on one chart.

    Each bar shows the fraction of simulated monthly returns
    that fell within that return range — no smoothing assumptions.
    """

    case_labels = {
        "base": "Base Case",
        "dynamic": "Dynamic Beta Case",
        "volatility": "Volatility Managed Case",
        "equity": "Equity Hedged Case",
    }
    
    # 2x2 grid — row 1: Base, Dynamic | row 2: Volatility, Equity
    positions = {
        "base": (1, 1),
        "dynamic": (1, 2),
        "volatility": (2, 1),
        "equity": (2, 2),
    }

    fig = make_subplots(
        rows = 2,
        cols = 2,
        subplot_titles = list(case_labels.values()),
        horizontal_spacing = 0.08,
        vertical_spacing = 0.14,
    )

    for case_key, case_label in case_labels.items():
        row, col = positions[case_key]

        returns_df = simulation_results[case_key]
        portfolio_returns = weighted_portfolio_returns(
            returns_df, portfolio
        )
        returns_array = portfolio_returns.values

        fig.add_trace(
            go.Histogram(
                x = returns_array,
                name = case_label,
                histnorm = "probability",
                opacity = 0.85,
                marker_color = case_colours[case_label],
                marker_line = dict(
                    color = case_colours[case_label],
                    width = 0.5,
                ),
                nbinsx = 25,
                showlegend = False,
                hovertemplate = (
                    f"<b>{case_label}</b><br>"
                    "Return: %{x}<br>"
                    "Frequency: %{y:.1%}"
                    "<extra></extra>"
                ),
            ),
            row=row, col = col,
        )

        # Zero return line on each subplot
        fig.add_vline(
            x = 0,
            line_width = 1,
            line_dash = "dash",
            line_color = "rgba(255,255,255,0.4)",
            row = row,
            col = col,
        )

    # Apply axis formatting to all subplots
    fig.update_xaxes(
        tickformat = ".1%",
        showgrid = True,
        gridcolor = "rgba(255,255,255,0.1)",
        color = "#FFFFFF",
        title_text = "Monthly Return",
        title_font = dict(size=11, color="#FFFFFF"),
    )
    fig.update_yaxes(
        tickformat = ".1%",
        showgrid = True,
        gridcolor = "rgba(255,255,255,0.1)",
        color = "#FFFFFF",
        title_text = "Relative Frequency",
        title_font = dict(size=11, color="#FFFFFF"),
    )

    # Subplot title colour — these sit in annotations
    for annotation in fig.layout.annotations:
        annotation.font.color = "#FFFFFF"
        annotation.font.size = 13

    fig.update_layout(
        title = dict(
            text = "Monthly Return Distribution — All Cases",
            font = dict(size=16, color="#FFFFFF"),
            x  = 0.5,
            xanchor = "center",
        ),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        height = 620,
        margin = dict(l=60, r=20, t=80, b=40),
    )

    return fig

# =================================================================
# 3. The Statistics Table
# =================================================================
def statistics_table(simulation_results: dict, 
                     portfolio: pd.DataFrame, 
                     risk_free_rate: float) -> go.Figure:
    """
    Builds a Plotly table showing summary statistics for all 
    four similation cases.

    Paramters: 
        - simulation_results: dict from run_simulation()
        - portfolio: pd.DataFrame from build_portfolio()
        - risk_free_rate: annualized risk-free rate

    Returns:
    go.Figure — a Plotly table figure
    """

    # Get the summary DataFrame from analytics.py
    summary = compute_all_statistics(
        simulation_results, portfolio, risk_free_rate
    )

    # --- Format each column for display ---
    cases = summary.index.tolist()

    ann_returns = [f"{v:.2%}" for v in summary["annualized_return"]]
    ann_stds = [f"{v:.2%}" for v in summary["annualized_std"]]
    sharpes = [f"{v:.3f}" for v in summary["sharpe"]]
    skews = [f"{v:.4f}" for v in summary["skewness"]]
    kurts = [f"{v:.4f}" for v in summary["kurtosis"]]

    # Colour each case row header to match the distribution chart
    header_colours = [case_colours[c] for c in cases]

    fig = go.Figure(data=[go.Table(
                # --- Column headers ---
        header = dict(
            values = ["<b>Case</b>",
                          "<b>Ann. Return</b>",
                          "<b>Ann. Std Dev</b>",
                          "<b>Sharpe Ratio</b>",
                          "<b>Skewness</b>",
                          "<b>Kurtosis</b>"],
            fill_color = "#1B8FFB",
            font = dict(color="white", size=12),
            align = "center",
            height = 32,
        ),

        # --- Table cells ---
        cells = dict(
            values = [cases,
                      ann_returns,
                      ann_stds,
                      sharpes,
                      skews,
                      kurts],
            fill_color = [
                ["#0a2038"]*4,                        
                ["#0a2038", "#071828"]*2,              
                ["#0a2038", "#071828"]*2,              
                ["#0a2038", "#071828"]*2,              
                ["#0a2038", "#071828"]*2,              
                ["#0a2038", "#071828"]*2,              
            ],
            font = dict(
                color = [["white"] * 4] + [["white"] * 4] * 5,
                size = 12,
            ),
            align = ["left"] + ["center"] * 5,
            height = 30,
        )
    )])

    fig.update_layout(
        title = dict(
            text = "Portfolio Statistics Summary",
            font = dict(size=16, color = "#FFFFFF"),
            x  = 0.5,
            xanchor = "center",
        ),
        margin = dict(l=0, r=0, t=40, b=0),
        height = 200,
        paper_bgcolor = "rgba(0,0,0,0)",             
        plot_bgcolor = "rgba(0,0,0,0)",
    )

    return fig

# =================================================================
# 4. The Sharpe ratio comparison chart
# =================================================================  
def sharpe_comparison_chart(simulation_results: dict,
                              portfolio: pd.DataFrame,
                              risk_free_rate: float) -> go.Figure:
    """
    Builds a grouped horizontal bar chart comparing per-asset
    Sharpe ratios across all four simulation cases.

    Parameters:

        - simulation_results: dict from run_simulation()
        - portfolio: pd.DataFrame from build_portfolio()
        - risk_free_rate: annualised risk-free rate

    Returns:
    
        go.Figure
    """
    from analytics import compute_asset_statistics

    asset_sharpes = compute_asset_statistics(
        simulation_results, portfolio, risk_free_rate
    )

    fig = go.Figure()

    for case_label in case_order:
        if case_label not in asset_sharpes.columns:
            continue

        fig.add_trace(go.Bar(
            name = case_label,
            y = asset_sharpes.index.tolist(),
            x = asset_sharpes[case_label].tolist(),
            orientation = "h", # horizontal bars
            marker_color = case_colours[case_label],
            opacity = 0.85,
            hovertemplate = (
                f"<b>{case_label}</b><br>"
                "%{y}<br>"
                "Sharpe: %{x:.3f}"
                "<extra></extra>"
            ),
        ))

    # Add a vertical line at zero Sharpe
    fig.add_vline(
        x = 0,
        line_width = 1,
        line_dash = "dash",
        line_color = "rgba(255,255,255,0.4)",
    )

    fig.update_layout(
        title = dict(
            text = "Per-Asset Sharpe Ratios by Case",
            font = dict(size=16, color = '#FFFFFF'),
            x = 0.5,
            xanchor = "center",
        ),
        xaxis = dict(
            title = "Sharpe Ratio",
            color = "#FFFFFF", 
            showgrid = True,
            gridcolor = "rgba(255, 255, 255, 0.1)",
            zeroline = True,
            zerolinecolor = "rgba(255,255,255,0.4)",
        ),
        yaxis = dict(
            title = "",
            color = "#FFFFFF", 
            autorange = "reversed",   # top asset first
        ),
        barmode = "group",        # side-by-side bars
        legend = dict(
            orientation = "h",
            font = dict(color = "#FFFFFF"), 
            yanchor = "bottom",
            y = 1.02,
            xanchor = "right",
            x = 1,
        ),
        plot_bgcolor = "rgba(0, 0, 0, 0)",
        paper_bgcolor = "rgba(0, 0, 0, 0)",
        height = 600,
        margin = dict(l=160, r=20, t=60, b=40),
    )

    return fig
    