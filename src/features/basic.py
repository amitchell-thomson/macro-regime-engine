"""
Basic feature computations: returns, changes, momentum, volatility, levels.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def should_compute_returns(asset_class):
    """
    Determine if ticker should get returns vs changes.
    
    Returns True for price-based assets, False for rate-based.
    """
    return asset_class in ['EQUITY', 'FX', 'COMMODITIES', 'VOL']


def compute_returns(df, ticker, asset_class, periods=[1, 5, 20, 60]):
    """
    Compute returns for price-based tickers.
    
    Args:
        df: DataFrame with dt index and value column
        ticker: Ticker symbol
        asset_class: Asset class
        periods: List of periods (in days) to compute returns
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    if not should_compute_returns(asset_class):
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore
    
    results = []
    
    # Ensure dt is index
    if 'dt' in df.columns:
        df = df.set_index('dt').sort_index()
    else:
        df = df.sort_index()
    
    for period in periods:
        # Compute returns: (price[t] / price[t-period]) - 1
        returns = df['value'].pct_change(periods=period)
        
        # Create feature name
        feature_name = f"{ticker}_RET_{period}D"
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'dt': returns.index,
            'ticker': ticker,
            'asset_class': asset_class,
            'feature': feature_name,
            'value': returns.values
        })
        
        # Remove NaN values
        result_df = result_df[result_df['value'].notna()]
        
        results.append(result_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore


def compute_changes(df, ticker, asset_class, periods=[1, 20, 60]):
    """
    Compute changes for rate-based tickers (yields, spreads, rates).
    
    Args:
        df: DataFrame with dt index and value column
        ticker: Ticker symbol
        asset_class: Asset class
        periods: List of periods (in days) to compute changes
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    if should_compute_returns(asset_class):
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore
    
    results = []
    
    # Ensure dt is index
    if 'dt' in df.columns:
        df = df.set_index('dt').sort_index()
    else:
        df = df.sort_index()
    
    for period in periods:
        # Compute changes: value[t] - value[t-period]
        changes = df['value'].diff(periods=period)
        
        # Create feature name
        feature_name = f"{ticker}_CHG_{period}D"
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'dt': changes.index,
            'ticker': ticker,
            'asset_class': asset_class,
            'feature': feature_name,
            'value': changes.values
        })
        
        # Remove NaN values
        result_df = result_df[result_df['value'].notna()]
        
        results.append(result_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore


def compute_momentum(df, ticker, asset_class, periods=[20, 60]):
    """
    Compute momentum (price change) for price-based tickers.
    
    Args:
        df: DataFrame with dt index and value column
        ticker: Ticker symbol
        asset_class: Asset class
        periods: List of periods (in days) to compute momentum
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    if not should_compute_returns(asset_class):
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore
    
    results = []
    
    # Ensure dt is index
    if 'dt' in df.columns:
        df = df.set_index('dt').sort_index()
    else:
        df = df.sort_index()
    
    for period in periods:
        # Compute momentum: price[t] - price[t-period]
        momentum = df['value'] - df['value'].shift(periods=period)
        
        # Create feature name
        feature_name = f"{ticker}_MOM_{period}D"
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'dt': momentum.index,
            'ticker': ticker,
            'asset_class': asset_class,
            'feature': feature_name,
            'value': momentum.values
        })
        
        # Remove NaN values
        result_df = result_df[result_df['value'].notna()]
        
        results.append(result_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore


def compute_volatility(df, ticker, asset_class, periods=[20, 60]):
    """
    Compute rolling volatility for price-based tickers.
    
    Args:
        df: DataFrame with dt index and value column
        ticker: Ticker symbol
        asset_class: Asset class
        periods: List of periods (in days) for rolling window
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    if not should_compute_returns(asset_class):
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore
    
    results = []
    
    # Ensure dt is index
    if 'dt' in df.columns:
        df = df.set_index('dt').sort_index()
    else:
        df = df.sort_index()
    
    # Compute daily returns first
    daily_returns = df['value'].pct_change()
    
    for period in periods:
        # Compute rolling volatility (annualized)
        volatility = daily_returns.rolling(window=period, min_periods=int(period * 0.5)).std() * np.sqrt(252)
        
        # Create feature name
        feature_name = f"{ticker}_VOL_{period}D"
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'dt': volatility.index,
            'ticker': ticker,
            'asset_class': asset_class,
            'feature': feature_name,
            'value': volatility.values
        })
        
        # Remove NaN values
        result_df = result_df[result_df['value'].notna()]
        
        results.append(result_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore


def compute_level(df, ticker, asset_class):
    """
    Store raw level as a feature (useful for spreads/ratios).
    
    Args:
        df: DataFrame with dt index and value column
        ticker: Ticker symbol
        asset_class: Asset class
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    # Ensure dt is index
    if 'dt' in df.columns:
        df = df.set_index('dt').sort_index()
    else:
        df = df.sort_index()
    
    feature_name = f"{ticker}_LEVEL"
    
    result_df = pd.DataFrame({
        'dt': df.index,
        'ticker': ticker,
        'asset_class': asset_class,
        'feature': feature_name,
        'value': df['value'].values
    })
    
    # Remove NaN values
    result_df = result_df[result_df['value'].notna()]
    
    return result_df


def compute_all_basic_features(df, ticker, asset_class):
    """
    Compute all basic features for a ticker.
    
    Args:
        df: DataFrame with dt and value columns
        ticker: Ticker symbol
        asset_class: Asset class
    
    Returns:
        DataFrame with all basic features
    """
    results = []
    
    # Ensure dt is index for computations
    df_indexed = df.copy()
    if 'dt' in df_indexed.columns:
        df_indexed = df_indexed.set_index('dt')
    
    # Compute level (always)
    level_df = compute_level(df_indexed, ticker, asset_class)
    results.append(level_df)
    
    # Compute returns/changes based on asset class
    if should_compute_returns(asset_class):
        returns_df = compute_returns(df_indexed, ticker, asset_class)
        results.append(returns_df)
        
        momentum_df = compute_momentum(df_indexed, ticker, asset_class)
        results.append(momentum_df)
        
        volatility_df = compute_volatility(df_indexed, ticker, asset_class)
        results.append(volatility_df)
    else:
        changes_df = compute_changes(df_indexed, ticker, asset_class)
        results.append(changes_df)
    
    # Combine all results
    if results:
        combined = pd.concat([r for r in results if not r.empty], ignore_index=True)
        return combined
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore
