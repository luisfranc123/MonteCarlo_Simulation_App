# =================================================================
# Monte Carlo engine for the four portfolio simulation cases.
# =================================================================

import numpy as np
import pandas as pd
from options import apply_portfolio_overlay

# -----------------------------------------------------------------
# EQUITY vs FIXED INCOME/ALTERNATIVE SPLIT
# -----------------------------------------------------------------

equity_assets = ["SPY", "MDY", "IWM", "EFA", "VWO", "DBC", "VNQ", "AMLP"]

def generate_spy_returns(exp_return: float, 
                         volatility: float, 
                         n_simulations: int) ->  np.ndarray:
    """
    Generates n_simuations random monthly SPY returns using a 
    normal distribution.

    Parameters:
        - exp_return: annualized expected return
        - volatility: annualized volatility
        - n_simulations: number of random returns to generate

    Returns:
        np.ndarray of shape (n_simulations, ) — montjly returns
    """
    # Convert annualized figures to monthly
    monthly_mean = exp_return/12
    monthly_std = volatility/np.sqrt(12)

    spy_returns = np.random.normal(
        loc = monthly_mean, 
        scale = monthly_std, 
        size = n_simulations
    )

    return spy_returns

# -----------------------------------------------------------------
# DERIVE ASSET RETURNS FROM SPY
# -----------------------------------------------------------------
def derive_asset_returns(spy_returns: np.ndarray,
                         asset_exp_return: float,
                         asset_volatility: float,
                         spy_volatility: float,
                         correlation: float) -> np.ndarray:
    """
    Derives monthly returns for a single asset from SPY returns
    using the beta relationship

    beta = (asset_vol/spy_vol)*corr
    asset_return = spy_return*beta

    Parameters:
        - spy_returns: the 1,000 SPY monthly returns
        - asset_exp_return: annualized expected return of the asset
        - asset_volatility: annualized volatility of the asset
        - spy_volatility: annualized volatility of SPY
        - correlation: Pearson correlation between asset and SPY
    
    Returns:
        - np.ndarray of shape (n_simulations,) — monthly asset returns
    """

    # Convert to monthly scale 
    asset_monthly_std = asset_volatility/np.sqrt(12)
    spy_monthly_std = spy_volatility/np.sqrt(12)

    # Calculate beta
    beta = (asset_monthly_std/spy_monthly_std)*correlation

    # Asset return = SPY return scaled by beta
    asset_returns = spy_returns*beta

    return asset_returns

def simulate_independence(exp_return: float,
                          volatility: float,
                          n_simulations: int) -> np.ndarray:
    """
    Generates independent monthly returns for fixed income and 
    alternative assets — not derived from SPY.  
    """
    monthly_mean = exp_return/12
    monthly_std = volatility/np.sqrt(12)

    return np.random.normal(
        loc = monthly_mean, 
        scale = monthly_std, 
        size = n_simulations
    )

# -----------------------------------------------------------------
# DYNAMIC BETA CASE
# -----------------------------------------------------------------
def derive_dynamic_returns(spy_returns: np.ndarray,
                           asset_volatility: float,
                           spy_volatility: float,
                           base_correlation: float,
                           stress_threshold: float = -0.02,
                           float_range: tuple   = (0.90, 1.10),
                           stress_multiplier: float = 1.50) -> np.ndarray:
    """
    Derives asset returns with floating/stressed correlation.

    Two regimes:
        - Normal (SPY return > -2%): correlation floats randomly
          between 90%-110% of base value.
        - Stressed (SPY return < -2%): correlation is multiplied by 
          150%, capped at 1.0
    This attempts to caputre the real-world phenomenom that assets become
    more correlated during market downturns. 

    Parameters:
        - spy_returns: the 1,000 SPY monthly returns
        - asset_volatility: annualized volatility of the asset
        - spy_volatility: annualized volatility of SPY
        - base_correlation: the historical Pearson correlation
        - stress_threshold: SPY return below which stress kicks in (-2%)
        - float_range: (min, max) multiplier for normal regime
        - stress_multiplier: correlation multiplier in stressed regime

    Returns:
        - np.ndarray of shape (n_simulations, ) — monthly asset returns
    """
    n = len(spy_returns)

    # Monthly volatilities for beta calculation
    asset_monthly_std = asset_volatility/np.sqrt(12)
    spy_monthly_std = spy_volatility/np.sqrt(12)

    # 1. Generate a floating correlation for every month
    float_multipliers = np.random.uniform(
        low = float_range[0], 
        high = float_range[1], 
        size = n
    )
    floating_corr = base_correlation*float_multipliers

    # 2. Compute stressed correlation for each month
    # np.minimum ensures that the stressed correlation 
    # does not surpass 1.0 (mathematically inconsistent)
    stressed_corr = np.minimum(
        base_correlation*stress_multiplier, 
        1.0
    )

    # 3. Choose which correaltion applies each month
    # If SPY return > -2%  → use floating correlation
    # If SPY return < -2%  → use stressed correlation
    # np.where(condition, value_if_true, value_if_false)
    effective_corr = np.where(
        spy_returns > stress_threshold, 
        floating_corr, 
        stressed_corr
    )

    # Compute dynamic beta and asset returns
    dynamic_beta = (asset_monthly_std/spy_monthly_std)*effective_corr
    asset_returns = spy_returns*dynamic_beta

    return asset_returns


# -----------------------------------------------------------------
# THE 10-YEAR PATH SAMPLER
# -----------------------------------------------------------------
def sample_ten_year_path(returns_matrix: pd.DataFrame, 
                        n_months: int = 120) -> pd.DataFrame:
    """
    Randomly samples n_months from 1,000 simulated monthly
    returns to construct a simulated -year return path. 

    The same random indices are used for every asset to 
    maintain consistency among all assets. 

    Parameters: 
        - returns_matrix: DataFrame of shape (1000, n_assets) 
        - n_months: number of months to sample (default 120)

    Returns:
        - pd.DataFrame of shape (n_months, n_assets)
    """ 
    n_simulations = len(returns_matrix)

    sampled_indices = np.random.choice(
        n_simulations, 
        size = n_months, 
        replace = False
    )
 
    # Use the above indices to select the same rows across all assets
    return returns_matrix.iloc[sampled_indices].reset_index(drop = True)

# -----------------------------------------------------------------
# THE MASTER SIMULATION FUNCTION
# -----------------------------------------------------------------
def run_simulation(portfolio: pd.DataFrame, 
                   corr_matrix: pd.DataFrame, 
                   dividend_yields: dict = None,
                   n_simulations: int = 1000, 
                   n_months: int = 120, 
                   risk_free_rate: float = 0.0225,
                   arin_fee: float = 0.0025, 
                   use_overlay: dict = None, 
                   random_seed: int = None) -> dict:
    """
    Runs all four Monte Carlo simulation cases and return their
    10-year sampled monthly return paths.

    Parameters: 
        - portfolio: DataFrame from build_portfolio() in data.py
        - corr_matrix: correlation matrix from fetch_portfolio_data()
        - n_simulations: number of random monthly returns to generate
        - n_months: months in the 10-year path (120 = 10 years)
        - use_overlay: dict controlling which cases apply the options
          overlay e.g. {"volatility": True, "equity": True}
        - random_seed: if set, fixes the random number generator so
          results are reproducible. 

    Returns: 
        dict with keys: "base", "dynamic", "volatility", "equity"
        Each value is a pd.DataFrame of shape (n_months, n_assets)
        containing the sampled monthly returns for that case.

    """
    # Default overlay
    if use_overlay is None:
        use_overlay = {"volatility": True, "equity": True}

    # If no dividend yields provided, default everything to 0.0
    if dividend_yields is None: 
        dividend_yields = {}
        
    # Fix the random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    # Locate SPY in the portfolio
    spy_row = portfolio[portfolio["yf_ticker"] == "SPY"].iloc[0]
    spy_vol = spy_row["volatility"]
    spy_ret = spy_row["exp_return"]

    # Generate the master SPY return series
    spy_returns = generate_spy_returns(spy_ret, spy_vol, n_simulations)

    # Build return matrices for base and dynamic beta cases
    base_returns = {}
    dynamic_returns = {}

    for _, asset in portfolio.iterrows():
        ticker = asset["yf_ticker"]
        name = asset["name"]
        ret = asset["exp_return"]
        vol = asset["volatility"]

        if ticker in corr_matrix.columns and "SPY" in corr_matrix.columns:
            corr = corr_matrix.loc[ticker, "SPY"]
        else:
            corr = 0.5 # defaul correlation

        # Equity assets: derive from SPY via beta
        is_equity = asset["is_equity"]

        if is_equity:
            base_returns[name] = derive_asset_returns(
                spy_returns, ret, vol, spy_vol, corr
            )
            dynamic_returns[name] = derive_dynamic_returns(
                spy_returns, vol, spy_vol, corr
            )
        else:
            independent = simulate_independence(
                ret, vol, n_simulations
            )
            base_returns[name] = independent
            dynamic_returns[name] = independent

    # Convert to DataFrames — rows = simulations, cols = assets
    base_df = pd.DataFrame(base_returns)
    dynamic_df = pd.DataFrame(dynamic_returns)

    # Sample a 120 months from each 
    base_path = sample_ten_year_path(base_df, n_months)
    dynamic_path = sample_ten_year_path(dynamic_df, n_months)

    # Volatility Managed and Equity Hedged cases
    volatility_path = apply_portfolio_overlay(
        returns_df = dynamic_path,
        portfolio = portfolio,
        risk_free_rate = risk_free_rate,
        dividend_yields = dividend_yields, 
        arin_fee = arin_fee, 
        equity_only = False,            
        )
    equity_path = apply_portfolio_overlay(
        returns_df = dynamic_path,
        portfolio = portfolio,
        risk_free_rate = risk_free_rate,
        dividend_yields = dividend_yields,
        arin_fee = arin_fee, 
        equity_only = True,              
        )

    return {
        "base": base_path, 
        "dynamic": dynamic_path, 
        "volatility": volatility_path, 
        "equity": equity_path
    }
            
    
