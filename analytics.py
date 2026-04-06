# =================================================================
# 1. Computes portfolio-level statistics for each simulation case.
#
# Takes the monthly returns DataFrames from simulation.py and
# produces a summary table with mean, std dev, Sharpe ratio,
# skewness and kurtosis — one row per simulation case.
# =================================================================

import numpy as np
import pandas as pd
from scipy import stats

# =================================================================
# 2. Computing weighted portfolio returns
# =================================================================

def weighted_portfolio_returns(returns_df: pd.DataFrame, 
                               portfolio: pd.DataFrame) -> pd.Series:
   """
   Computes the weighted portfolio returns for each month.
    
   Parameters:
    - returns_df: pd.DataFrame from simulation.py
    - portfolio: pd.DataFrame from build_portfolio() in data.py
   
   Returns:
       pd.Series of length n_months — one weighted return per month
   """
    
   # Build a weight series inexed by asset name
   weights = portfolio.set_index("name")["weight"]
    
   # Only keep assets that appear in both the returns DataFrame
   # and the weights Series — handles any missing column. 
   common_assets = [col for col in returns_df.columns if col in weights.index]
    
   # Subset both to the common assets
   returns_subset = returns_df[common_assets]
   weights_subset = weights[common_assets]
    
   # Verify weights sum to approx 1.0
   # If not, we normalize them to ensure valid portfolio math
   total_weight = weights_subset.sum()
   if not np.isclose(total_weight, 1.0, atol = 0.01):
       weights_subset = weights_subset/total_weight

   # Matrix multiply returns by weights
   portfolio_returns = returns_subset.dot(weights_subset)
    
   return portfolio_returns

# =================================================================
# 3. Computing statistics for one case
# =================================================================

def compute_case_statistics(portfolio_returns: pd.Series, 
                            risk_free_rate: float, 
                            n_months: int = 120) -> dict:
    """
    Computes summary statistics for one simulation case.
    
    Parameters:
     - portfolio_returns: pd.Series of monthly weighted returns
     - risk_free_rate: annualized risk-free rate
     - n_months: number of omnths in the simulation path
    
    Returns:
     dict with keys: mean, std_dev, sharpe, skewness, kurtosis, 
     annualized_return, annualized_std.
    """
    # Monthly statistics
    mean_monthly = portfolio_returns.mean()
    std_monthly = portfolio_returns.std()
    
    # Annualized figures
    annualized_return = mean_monthly*12
    annualized_std = std_monthly*np.sqrt(12)
    
    # Sharpe ratio
    if annualized_std > 0:
        sharpe = (annualized_return - risk_free_rate)/annualized_std
    else:
        sharpe = 0.0
    
    # Skewness
    # Positive: right tail is longer
    # Negative: left tail is longer
    skewness = stats.skew(portfolio_returns)
    
    # kurtosis
    # Positive: fatter tails than normal
    # Negative: thinner tails than normal
    kurt = stats.kurtosis(portfolio_returns)
    
    return {
     "mean_monthly": round(mean_monthly, 4),
     "std_monthly": round(std_monthly, 4),
     "annualized_return": round(annualized_return, 4),
     "annualized_std": round(annualized_std, 4),
     "sharpe": round(sharpe, 4),
     "skewness": round(skewness, 4),
     "kurtosis": round(kurt, 4),
        }
 
# =================================================================
# 4. Analytics Function
# =================================================================
def compute_all_statistics(simulation_results: dict, 
                           portfolio: pd.DataFrame, 
                           risk_free_rate: float) -> pd.DataFrame:
    """
    Computes statistics for all four simulation cases and returns
    a summary DataFrame
    
    Parameters:
     - simulation_results: dict from run_simulation() in simulation.py
     - portfolio: pd.DataFrame from build_portfolio() in data.py
     - risk_free_rate: annualized risk_free rate
     
    Returns
     pd.DataFrame with one row per case and columns for each statistic
    """
    # Readable labels for each case
    case_labels = {
     "base": "Base Case", 
     "dynamic": "Dynamic Beta Case", 
     "volatility": "Volatility Managed Case", 
     "equity": "Equity Hedged Case"
    }
    
    rows = []
    for case_key, case_label in case_labels.items():
        # get the returns DataFrame for this case
        returns_df = simulation_results[case_key]
        # Compute weighted portfolio returns
        portfolio_returns = weighted_portfolio_returns(
            returns_df, portfolio
        )

        # Compute statistics
        case_stats = compute_case_statistics(
            portfolio_returns, risk_free_rate
        )

        # Add the case label to the stats dict
        case_stats["case"] = case_label
        rows.append(case_stats)

    # Build DataFrame and set case as the index
    summary_df = pd.DataFrame(rows).set_index("case")

    return summary_df
 
# =================================================================
# 5. A helper for per-asset statistics
# =================================================================
def compute_asset_statistics(simulation_results: dict, 
                             portfolio: pd.DataFrame, 
                             risk_free_rate: float) -> pd.DataFrame:
    
    """
    Computes the annualized Sharpe ratio for each individual asset
    across all four simulation cases.

    Returns
    -------
        pd.DataFrame with assets as rows and cases as columns
    """
    case_labels = {
    "base": "Base Case", 
    "dynamic": "Dynamic Beta Case", 
    "volatility": "Volatility Managed Case", 
    "equity": "Equity Hedged Case"
    }
    # Collect Sharpe ratios: asset -> case -> sharpe
    sharpe_data = {}

    for case_key, case_label in case_labels.items():
        returns_df = simulation_results[case_key]
        sharpe_data[case_label] = {}

        for col in returns_df.columns:
            asset_returns = returns_df[col]
            ann_return = asset_returns.mean()*12
            ann_std = asset_returns.std()*np.sqrt(12)
            
            if ann_std > 0:
                sharpe = (ann_return - risk_free_rate)/ann_std
            else:
                sharpe = 0.0

            sharpe_data[case_label][col] = round(sharpe, 4)

    return pd.DataFrame(sharpe_data)
        
    

 
 