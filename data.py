# =================================================================, 
# Fetches and prepares portfolio data from yfinance.
# =================================================================

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# -----------------------------------------------------------------
# ASSET DEFINITIONS
# Each asset has a display name, its Carson ticker, and the yfinance
# ticker we'll actually use to fetch data (may differ from Carson).
# -----------------------------------------------------------------

assets = [
    # --- Equity ---
    {"name": "US Large Cap", "carson": "SPY", "yf": "SPY", "is_equity": True, "weight": 0.4125},
    {"name": "US Mid Cap", "carson": "MDY", "yf": "MDY", "is_equity": True, "weight": 0.1238},
    {"name": "US Small Cap", "carson": "IWM", "yf": "IWM", "is_equity": True, "weight": 0.0825},
    {"name": "Developed Intl", "carson": "EFA", "yf": "EFA", "is_equity": True, "weight": 0.0619},
    {"name": "Emerging Intl", "carson": "VWO", "yf": "VWO", "is_equity": True, "weight": 0.0619},
    {"name": "Commodity", "carson": "DBC", "yf": "DBC", "is_equity": True, "weight": 0.0206},
    {"name": "REIT", "carson": "VNQ", "yf": "VNQ", "is_equity": True, "weight": 0.0206},
    {"name": "MLP", "carson": "AMLP", "yf": "AMLP", "is_equity": True, "weight": 0.0206},
    # --- Fixed Income ---
    {"name": "Investment Grade", "carson": "AGG", "yf": "AGG", "is_equity": False, "weight": 0.0000},
    {"name": "Credit Sensitive", "carson": "HYG", "yf": "HYG", "is_equity": False, "weight": 0.0525},
    {"name": "Sovereign", "carson": "BWX", "yf": "BWX", "is_equity": False, "weight": 0.0000},
    {"name": "Intl Fixed Income", "carson": "IBND", "yf": "IBND", "is_equity": False, "weight": 0.0150},
    {"name": "Emerging Fixed Income", "carson": "ELD", "yf": "ELD", "is_equity": False, "weight": 0.0150},
    {"name": "Short Duration", "carson": "BSV", "yf": "BSV", "is_equity": False, "weight": 0.0150},
    {"name": "Floating Rate", "carson": "FLOT", "yf": "FLOT", "is_equity": False, "weight": 0.0450},
    {"name": "Inflation Protected", "carson": "TIP", "yf": "TIP", "is_equity": False, "weight": 0.0075},
    # --- Alternatives ---
    {"name": "Global Macro", "carson": "MCRO", "yf": "WTMF", "is_equity": True, "weight": 0.0000},
    {"name": "Currency / Commodity", "carson": "GLD", "yf": "GLD", "is_equity": True, "weight": 0.0000},
    {"name": "Private Investment", "carson": "PEX", "yf": "PSP", "is_equity": True, "weight": 0.0000},
    {"name": "Volatility Arbitrage", "carson": "HFRI I", "yf": "QAI", "is_equity": True, "weight": 0.0250},
    {"name": "Specialty Situation", "carson": "HFRI II","yf": "QAI", "is_equity": True, "weight": 0.0206},
]

# Convenience lists used throughout the app
names = [a["name"] for a in assets]
yf_tickers = [a["yf"] for a in assets]  

# -----------------------------------------------------------------
# FETCH PORTFOLIO DATA
# Pulls 5 years of monthly adjusted closing prices from yfinance,
# computes monthly returns, and derives expected returns, volatility
# (standard deviation), and the full correlation matrix.
# -----------------------------------------------------------------

@st.cache_data(ttl = 3600)
def fetch_portfolio_data():
    """
    Returns a dictionary with three keys:
        - 'returns': pd.DataFrame of monthly returns (rows = months, cols == assets)
        - 'exp_returns': pd.Series of annualized expected return per asset
        - 'volatility': pd.Series of annualized volatility (stv deviation) per asset
        - 'corr_matrix': pd.DataFrame of the full correlation matrix
        - 'failed': list of any ticket that could not be fethched
    """
    
    # 1: Download raw price data 
    # yfinance can fetch multiple tickers in one call.
    # auto_adjust=True gives us adjusted closing prices, which account
    # for dividends and stock splits — essential for accurate returns.
    raw = yf.download(
        tickers = yf_tickers, 
        period = "5y", 
        interval = "1mo",
        auto_adjust = True, 
        progress = False        
    )

    prices = raw["Close"]

    # 2: Identify any failed tickers 
    # If a ticker returned all NaN values, it failed to fetch.
    failed = [t for t in yf_tickers if prices[t].isna().all]

    # Drop columns that entirely empty
    prices = prices.dropna(axis = 1, how = "all")

    # 3: Compute monthly returns 
    monthly_returns = prices.pct_change(fill_method = None).dropna()

    # 4: Annualise expected returns and volatility 
    exp_returns = monthly_returns.mean()*12
    volatility = monthly_returns.std()*np.sqrt(12)

    # 5: Build the correlation matrix 
    corr_matrix = monthly_returns.corr()

    return {
        "returns": monthly_returns, 
        "exp_returns": exp_returns, 
        "volatility": volatility, 
        "corr_matrix": corr_matrix, 
        "failed": failed
    }

# -----------------------------------------------------------------
#Fetching live dividend yields
# -----------------------------------------------------------------
dividend_yields = {
    "SPY": 0.0125,
    "AGG": 0.0350,
    "HYG": 0.0550,
    "BWX": 0.0300,
    "IBND": 0.0280,
    "ELD": 0.0450,
    "BSV": 0.0320,
    "FLOT": 0.0480,
    "TIP": 0.0220,
    "WTMF": 0.0000,
    "GLD": 0.0000,
    "PSP": 0.0200,
    "QAI": 0.0150, 
    "MDY": 0.0130,
    "IWM": 0.0110,
    "EFA": 0.0310,
    "VWO": 0.0350,
    "DBC": 0.0000,
    "VNQ": 0.0380,
    "AMLP": 0.0780,
    }
@st.cache_data(ttl = 3600)
def fetch_dividend_yields() -> dict:
    """
    Fetches twelve-month dividend yields for each 
    yfinance ticker. Falls back to dividend_yields 
    defaults if yfinance returns None ot fails for a ticker.

    Returns: 
        dict mapping yf_ticker -> dividend yield (float)
    """
    yields = {}
    unique_tickers = list(set(yf_tickers))

    for ticker in unique_tickers:
        try:
            info = yf.Ticker(ticker).info
            q = info.get("dividendYield", None)
            if q is None or q != q:
                q = dividend_yields.get(ticker, 0.0)
            else:
                q = q/100
        except Exception:
            q = dividend_yields.get(ticker, 0.0)

        yields[ticker] = q
    
    return yields
# -----------------------------------------------------------------
# BUILD PORTFOLIO INPUTS
# Maps the fetched yfinance data back onto our 14-asset list.
# If a ticker failed, we fall back to the Carson Excel values so
# the simulation can still run, and we flag it for the user.
# -----------------------------------------------------------------
# Format: "yf_ticker": (annualised_return, annualised_volatility)
carson_fallbacks = {
    "SPY": (0.060,  0.1211),
    "AGG": (0.0323, 0.0283),
    "HYG": (0.0634, 0.0779),
    "BWX": (0.0012, 0.0680),
    "IBND": (0.0046, 0.0879),
    "ELD": (-0.0219,0.1189),
    "BSV": (0.0141, 0.0121),
    "FLOT": (0.0082, 0.0108),
    "TIP": (0.0140, 0.0484),
    "WTMF": (0.0109, 0.1651),
    "GLD": (0.0004, 0.1895),
    "PSP": (0.0543, 0.5127),
    "QAI": (0.0667, 0.3320),
    "MDY": (0.0670, 0.1432),
    "IWM": (0.0742, 0.1664),
    "EFA": (0.0658, 0.1528),
    "VWO": (0.0806, 0.1997),
    "DBC": (0.0386, 0.1719),
    "VNQ": (0.0591, 0.1641),
    "AMLP": (0.0318, 0.1525),
    }

def build_portfolio(data: dict) -> pd.DataFrame:
    """ 
    Returns a DataFrame with one row per asset and columns: 
    - name, carson_ticker, yf_ticker, exp_return, volatility, source.
    """ 
    rows = []
    for asset in assets:
        ticker = asset["yf"]
        name = asset["name"]
        carson = asset["carson"]

        if ticker in data["exp_returns"].index:
            rows.append({
                "name": name, 
                "carson_ticker": carson, 
                "yf_ticker": ticker, 
                "exp_return": data["exp_returns"][ticker], 
                "volatility": data["volatility"][ticker], 
                "is_equity": asset["is_equity"], 
                "weight" : asset["weight"], 
                "source": "yfinance", 
            }) 
        else:
            # Use Carson fallback values
            ret, vol = carson_fallbacks.get(ticker, (0.05, 0.15))
            rows.append({
                "name": name, 
                "carson_ticker": carson, 
                "yf_ticker": ticker, 
                "exp_return": ret, 
                "volatility": vol, 
                "is_equity": asset["is_equity"],
                "weight" : asset["weight"], 
                "source": "fallback", 
            })
    
    return pd.DataFrame(rows)

    
    

