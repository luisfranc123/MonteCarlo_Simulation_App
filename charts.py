# =================================================================
# Plotly visualisations for the Monte Carlo simulation app.#1B8FFB
# 1. Libraries and depdndencies
# =================================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    Builds a Plotly figure showing the kde return distribution 
    for all 4 simulations overlaid on one chart. 
    
    Parameters:    
        - simulation_results : dict from run_simulation() in simulation.py
        - portfolio: pd.DataFrame from build_portfolio() in data.py
        - risk_free_rate: annualised risk-free rate (for annotation)

    Returns
    
        go.Figure — a Plotly figure 
    
    """
    # Internal keys 
    case_labels = {
        "base": "Base Case", 
        "dynamic": "Dynamic Beta Case", 
        "volatility": "Volatility Managed Case",
        "equity": "Equity Hedged Case",
    }
    fig = go.Figure()

    for case_key, case_label in case_labels.items():
        returns_df = simulation_results[case_key]

        # get weighted portfolio retuns
        portfolio_returns = weighted_portfolio_returns(
            returns_df, portfolio
        )
    
        # Convert to numpy array for scipy
        returns_array = portfolio_returns.values
    
        # Build the kde
        kde = gaussian_kde(returns_array)
    
        # create x-grid
        x_min = returns_array.min() - 0.01
        x_max = returns_array.max() + 0.01
        x_vals = np.linspace(x_min, x_max, 300)
        
        # evaluate the kde at each x value 
        y_vals = kde(x_vals)
    
        # Add the trace to the figure
        fig.add_trace(go.Scatter(
            x = x_vals, 
            y = y_vals, 
            mode = "lines", 
            name = case_label, 
            line = dict(
                color = case_colours[case_label], 
                width = 2.5,
                dash = case_styles[case_label],
        ), 
        hovertemplate = (
            f"<b>{case_label}</b><br>"
            "Return: %{x:;2%}<br>"
            "Density: %{y:.2f}"
            "<extra></extra>"
        ), 
        ))

    # Add a vertical line at zero
    fig.add_vline(
        x = 0, 
        line_width = 1, 
        line_dash = "dash", 
        line_color = "rgba(255,255,255,0.4)", 
        annotation_text = "Zero Return", 
        annotation_position = "top right", 
    )

    # Layout
    fig.update_layout(
        title = dict(
            text = "Monthly Return Distribution — All Cases",
            font = dict(size = 16),
            x = 0.5,        
            xanchor = "center",
        ),
        xaxis = dict(
            title = "Monthly Return",
            tickformat = ".1%",    
            showgrid = True,
            gridcolor = "rgba(255, 255, 255, 0.1)",
            color = "#FFFFFF", 
        ),
        yaxis = dict(
            title = "Probability Density",
            showgrid = True,
            gridcolor = "rgba(255, 255, 255, 0.1)",
            color = "#FFFFFF",
        ),
        legend = dict(
            orientation = "h",         
            yanchor = "bottom",
            y = 1.02,        
            xanchor = "right",
            x = 1,
            font = dict(color = "#FFFFFF"), 
        ),
        hovermode = "x unified",     
        plot_bgcolor = "rgba(0, 0, 0, 0)",
        paper_bgcolor ="rgba(0, 0, 0, 0)",
        height = 500,
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
            font = dict(size=16),
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
    