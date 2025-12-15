"""
Basic feature computations: returns, changes, momentum, volatility, levels.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


def should_compute_returns(asset_class):
    """
    Determine if ticker should get returns vs changes.
    
    Returns True for price-based assets, False for rate-based.
    """
    return asset_class in ['EQUITY', 'FX', 'COMMODITIES', 'VOL']


def detect_data_frequency(df, ticker):
    """
    Detect the actual frequency of data updates.
    
    Args:
        df: DataFrame with dt index and value column
        ticker: Ticker symbol (for known frequency mapping)
    
    Returns:
        Tuple of (frequency_type, typical_period_days)
        frequency_type: 'daily', 'weekly', 'monthly', 'quarterly'
        typical_period_days: typical days between updates
    """
    # Known frequency mappings for specific tickers
    quarterly_tickers = ['GDP']
    monthly_tickers = ['UNRATE', 'CPIAUCSL', 'PCEPI', 'UMCSENT', 'M2SL', 'USSLIND']
    weekly_tickers = []  # Add if needed
    daily_tickers = ['DCOILWTICO', 'NFCI', 'WALCL']  # Some macro indicators are daily
    
    if ticker in quarterly_tickers:
        return ('quarterly', 90)
    elif ticker in monthly_tickers:
        return ('monthly', 30)
    elif ticker in weekly_tickers:
        return ('weekly', 7)
    elif ticker in daily_tickers:
        return ('daily', 1)
    
    # Auto-detect from data gaps if not in known list
    if len(df) < 2:
        return ('daily', 1)  # Default assumption
    
    # Calculate gaps between non-null values
    non_null_df = df[df['value'].notna()]
    if len(non_null_df) < 2:
        return ('daily', 1)
    
    gaps = non_null_df.index.to_series().diff().dt.days.dropna()
    
    if len(gaps) == 0:
        return ('daily', 1)
    
    median_gap = gaps.median()
    
    # Classify based on median gap
    if median_gap <= 2:
        return ('daily', 1)
    elif median_gap <= 10:
        return ('weekly', 7)
    elif median_gap <= 45:
        return ('monthly', 30)
    else:
        return ('quarterly', 90)


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
    Automatically detects data frequency and computes appropriate changes.
    
    Args:
        df: DataFrame with dt index and value column
        ticker: Ticker symbol
        asset_class: Asset class
        periods: List of periods (in days) - used only for daily data
    
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
    
    # Detect data frequency
    freq_type, typical_period = detect_data_frequency(df, ticker)
    
    # Get only non-null values for frequency-based computations
    non_null_df = df[df['value'].notna()].copy()
    
    if len(non_null_df) < 2:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore
    
    # Compute changes based on frequency
    if freq_type == 'daily':
        # For daily data, use the original periods
        for period in periods:
            changes = df['value'].diff(periods=period)
            
            feature_name = f"{ticker}_CHG_{period}D"
            
            result_df = pd.DataFrame({
                'dt': changes.index,
                'ticker': ticker,
                'asset_class': asset_class,
                'feature': feature_name,
                'value': changes.values
            })
            
            result_df = result_df[result_df['value'].notna()]
            results.append(result_df)
    
    elif freq_type == 'monthly':
        # For monthly data, compute month-over-month and year-over-year
        # MoM: compare to previous month
        mom_changes = non_null_df['value'].diff(periods=1)
        
        mom_df = pd.DataFrame({
            'dt': mom_changes.index,
            'ticker': ticker,
            'asset_class': asset_class,
            'feature': f"{ticker}_CHG_MOM",
            'value': mom_changes.values
        })
        mom_df = mom_df[mom_df['value'].notna()]
        results.append(mom_df)
        
        # YoY: compare to same month last year (approximately 12 periods back)
        if len(non_null_df) >= 13:
            yoy_changes = non_null_df['value'].diff(periods=12)
            
            yoy_df = pd.DataFrame({
                'dt': yoy_changes.index,
                'ticker': ticker,
                'asset_class': asset_class,
                'feature': f"{ticker}_CHG_YOY",
                'value': yoy_changes.values
            })
            yoy_df = yoy_df[yoy_df['value'].notna()]
            results.append(yoy_df)
    
    elif freq_type == 'quarterly':
        # For quarterly data, compute quarter-over-quarter and year-over-year
        # QoQ: compare to previous quarter
        qoq_changes = non_null_df['value'].diff(periods=1)
        
        qoq_df = pd.DataFrame({
            'dt': qoq_changes.index,
            'ticker': ticker,
            'asset_class': asset_class,
            'feature': f"{ticker}_CHG_QOQ",
            'value': qoq_changes.values
        })
        qoq_df = qoq_df[qoq_df['value'].notna()]
        results.append(qoq_df)
        
        # YoY: compare to same quarter last year (4 periods back)
        if len(non_null_df) >= 5:
            yoy_changes = non_null_df['value'].diff(periods=4)
            
            yoy_df = pd.DataFrame({
                'dt': yoy_changes.index,
                'ticker': ticker,
                'asset_class': asset_class,
                'feature': f"{ticker}_CHG_YOY",
                'value': yoy_changes.values
            })
            yoy_df = yoy_df[yoy_df['value'].notna()]
            results.append(yoy_df)
    
    elif freq_type == 'weekly':
        # For weekly data, compute week-over-week
        wow_changes = non_null_df['value'].diff(periods=1)
        
        wow_df = pd.DataFrame({
            'dt': wow_changes.index,
            'ticker': ticker,
            'asset_class': asset_class,
            'feature': f"{ticker}_CHG_WOW",
            'value': wow_changes.values
        })
        wow_df = wow_df[wow_df['value'].notna()]
        results.append(wow_df)
    
    # Reindex results back to original date index for consistency
    # This ensures we have values on all dates (forward-filled from actual updates)
    if results:
        combined = pd.concat(results, ignore_index=True)
        
        # For non-daily frequencies, forward-fill the computed changes to daily frequency
        # This allows the features to be used alongside daily features
        if freq_type != 'daily':
            # Process each feature separately to forward-fill correctly
            daily_results = []
            
            for feature_name in combined['feature'].unique():
                feature_data = combined[combined['feature'] == feature_name].copy()
                
                # Create daily index from min to max date
                daily_index = pd.date_range(
                    start=feature_data['dt'].min(),
                    end=feature_data['dt'].max(),
                    freq='D'
                )
                
                # Set dt as index, reindex to daily, forward fill
                feature_daily = feature_data.set_index('dt')
                feature_daily = feature_daily.reindex(daily_index)
                feature_daily = feature_daily.ffill()
                
                # Reset index
                feature_daily = feature_daily.reset_index()
                feature_daily = feature_daily.rename(columns={'index': 'dt'})
                
                # Remove rows where value is still NaN (before first actual value)
                feature_daily = feature_daily[feature_daily['value'].notna()]
                
                daily_results.append(feature_daily)
            
            if daily_results:
                return pd.concat(daily_results, ignore_index=True)
            else:
                return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore
        else:
            return combined
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
