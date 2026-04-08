# =================================================================
# Black-Scholes collar pricing and return adjustment.
# Carson:
# A collar = long put (protection) + short call (financing).
# Applied monthly to each asset at ±5% OTM strikes.
# The net cost is subtracted from simulated returns before
# the return is truncated to the ±5% band.
# =================================================================

import numpy as np
from scipy.stats import norm

# -----------------------------------------------
# BLACK-SCHOLES PRICING
# ----------------------------------------------
def black_scholes_price(S: float, K: float, r: float, sigma: float, 
                        T: float, option_type: str, q: float = 0.0,) -> float:
    """
    Calculates the theoretical price of a European call or put option
    using the Black-Scholes model.

    Args:
        - S (float): current SPX level (spot price)
        - K (float): strike price
        - r (float): risk-free rate (decimal, e.g. 0.043)
        - q (float): dividend yield (decimal, e.g. 0.015)
        - sigma (float): implied volatility (decimal, e.g. 0.18)
        - T (float): time to expiration in years (days/365)
        - option_type (str): "call" or "put"
    Returns:
        - float: theoretical option price
    
    """
    if sigma <= 0:
        return 0.0
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)/S
        else:
            return max(K - S, 0)/S
    
    # d1 and d2 computation
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T) 
    

    if option_type == "call":
        price = (S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)* norm.cdf(d2))
    elif option_type == "put":
        price = (K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1))
    else:
        raise ValueError(
            f"option_value must be 'call' or 'put', got '{option_type}'"
        )

    return price/S

# -----------------------------------------------
# COLLAR COST
# ----------------------------------------------
def collar_cost(volatility: float,
                risk_free_rate: float,
                dividend_yield: float = 0.0,  
                strike_width: float = 0.05,
                skew_factor: float = 0.05,
                arin_fee: float = 0.0025,
                T: float = 1/12) -> float:
    """
    Computes the monthly net cost of a collar strategy for a
    single asset.

    The collar consists of:
      - Long put at strike S × (1 - strike_width) 
      - Short call at strike S × (1 + strike_width)  

    The put is made slightly more expensive by the skew_factor to
    reflect the volatility smile — out-of-the-money puts trade at
    higher implied volatility than equivalent calls in practice.

    Net cost = (put_price × (1 + skew_factor)) - call_price + arin_fee

    Parameters:
        - volatility: annualized asset volatility
        - risk_free_rate: annualized risk-free rate
        - strike_width: how far OTM the strikes are (0.05 = 5%)
        - skew_factor: premium added to put to reflect vol smile (0.05)
        - arin_fee: Arin's monthly advisory fee as decimal (0.0025)
        - T: option tenor in years (default 1/12 = 1 month)

    Returns:
        float — net monthly cost as a decimal (e.g. 0.008 = 0.8%)
    """
    S = 100.0
    K_put = S*(1 - strike_width)
    K_call = S*(1 + strike_width)

    put_price = black_scholes_price(
        S, K_put, T, risk_free_rate, 
        volatility, "put", q = dividend_yield
    )
    call_price = black_scholes_price(
        S, K_call, T, risk_free_rate, 
        volatility, "call", q = dividend_yield
    )

    adjusted_put = put_price*(1 + skew_factor)
    net_cost = adjusted_put - call_price + arin_fee

    return max(net_cost, 0.0)

def apply_collar(returns: np.ndarray,
                 volatility: float,
                 risk_free_rate: float,
                 dividend_yield: float = 0.0,  
                 strike_width: float = 0.05,
                 skew_factor: float = 0.05,
                 arin_fee: float = 0.0025) -> np.ndarray:
    """
    Applies the collar to a series of monthly returns.

    Steps:
      1. Compute the net monthly cost of the collar for this asset
      2. Truncate each return to the [-strike_width, +strike_width] band
      3. Subtract the net cost from every return

    Parameters:
        returns: np.ndarray of monthly returns for one asset
        volatility: annualised volatility of the asset
        risk_free_rate: annualised risk-free rate
        strike_width: collar width (default 0.05 = ±5%)
        skew_factor: vol smile adjustment on put (default 0.05)
        arin_fee: monthly advisory fee (default 0.0025 = 0.25%)

    Returns:
        np.ndarray — collar-adjusted monthly returns, same shape as input
    """
     # 1. compute net cost once for this asset
    cost = collar_cost(volatility, risk_free_rate, dividend_yield, 
                      strike_width, skew_factor, arin_fee)
    
    # 2. truncate returns to the +- strike_width band
    # # np.clip(array, min, max) replaces any value below min with min
    # and any value above max with max. Everything in between remains the same.
    clipped = np.clip(returns, -strike_width, strike_width)
    
    # 3. subtract the net cost from every return
    adjusted = clipped - cost

    return adjusted

# -----------------------------------------------
# PORTFOLIO LEVEL OVERLAY FUNCTION
# ----------------------------------------------
def apply_portfolio_overlay(returns_df,
                             portfolio,
                             risk_free_rate: float,
                             dividend_yields: dict,   
                             equity_only: bool = False,
                             strike_width: float = 0.05,
                             skew_factor: float = 0.05,
                             arin_fee: float = 0.0025):
    
    """
    Applies collar overlay to a portfolio returns DataFrame.

    Parameters:
        - returns_df: pd.DataFrame (rows = months, cols = asset names)
          from simulation.py
        - portfolio: pd.DataFrame from build_portfolio() in data.py
        - risk_free_rate annualised risk-free rate
        - equity_only: if True, only apply collar to equity assets
          (Equity Hedged Case) if False, apply to all assets
          (Volatility Managed Case)
        - strike_width: collar width (default 0.05 = ±5%)
        - skew_factor: vol smile adjustment (default 0.05)
        - arin_fee: monthly advisory fee (default 0.0025)

    Returns:
        pd.DataFrame — same shape as returns_df with collar applied
    """
    # Copy returns_df
    adjusted_df = returns_df.copy()

    for _, asset in portfolio.iterrows():
        name = asset["name"]
        vol = asset["volatility"]
        is_equity = asset["is_equity"]
        ticker = asset["yf_ticker"]
        
        # Skip if this asset's column isn't in the returns DataFrame
        if name not in adjusted_df.columns:
            continue
        should_hedge = is_equity if equity_only else True

        if should_hedge:
            # look up this asset's dividend yield
            q = dividend_yields.get(ticker, 0.0)

            # Decide whether to apply the collar to this asset
            adjusted_df[name] = apply_collar(
                returns = adjusted_df[name].values,
                volatility = vol,
                risk_free_rate = risk_free_rate,
                dividend_yield = q,              
                strike_width = strike_width,
                skew_factor = skew_factor,
                arin_fee = arin_fee,
            )

    return adjusted_df
            
            
    
    