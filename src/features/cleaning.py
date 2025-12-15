"""
Data cleaning utilities for feature engineering.
Handles missing data, frequency alignment, outlier detection, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def clean_raw_data(df, ticker, asset_class):
    """
    Apply ticker-specific cleaning rules.
    
    Args:
        df: DataFrame with columns: dt, value
        ticker: Ticker symbol
        asset_class: Asset class
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    df = df.sort_values('dt')
    
    # Remove duplicates (keep most recent)
    df = df.drop_duplicates(subset=['dt'], keep='last')
    
    # Validate data quality
    # Check for negative prices (for price-based assets)
    if asset_class in ['EQUITY', 'FX', 'COMMODITIES', 'VOL']:
        if (df['value'] < 0).any():
            # Log warning but don't remove (might be valid for some edge cases)
            pass
    
    # Check for extreme values
    if asset_class in ['EQUITY', 'FX', 'COMMODITIES']:
        # Flag returns > 50% daily (likely data error)
        if len(df) > 1:
            returns = df['value'].pct_change()
            extreme_returns = returns.abs() > 0.5
            if extreme_returns.any():
                # Set to NaN for manual review
                df.loc[extreme_returns, 'value'] = np.nan
    
    return df


def forward_fill_data(df, max_days, freq='D'):
    """
    Forward fill missing data with limits.
    
    Args:
        df: DataFrame with dt index and value column
        max_days: Maximum days to forward fill
        freq: Frequency string ('D' for daily, etc.)
    
    Returns:
        DataFrame with forward-filled values
    """
    df = df.copy()
    
    # Ensure dt is index
    if 'dt' in df.columns:
        df = df.set_index('dt')
    
    # Create full date range
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(date_range)
    
    # Forward fill with limit
    if max_days is not None:
        df['value'] = df['value'].ffill(limit=max_days)
    else:
        df['value'] = df['value'].ffill()
    
    return df.reset_index().rename(columns={'index': 'dt'})


def detect_outliers(df, method='iqr', feature_col='value'):
    """
    Detect outliers using specified method.
    
    Args:
        df: DataFrame
        method: 'iqr' (interquartile range) or 'zscore'
        feature_col: Column name to check for outliers
    
    Returns:
        Boolean Series indicating outliers
    """
    if method == 'iqr':
        Q1 = df[feature_col].quantile(0.25)
        Q3 = df[feature_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[feature_col] < lower_bound) | (df[feature_col] > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((df[feature_col] - df[feature_col].mean()) / df[feature_col].std())
        outliers = z_scores > 3
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers


def align_frequencies(df_dict, target_freq='D'):
    """
    Align multiple dataframes to common date index.
    
    Args:
        df_dict: Dictionary of {ticker: DataFrame} where each DataFrame has dt index
        target_freq: Target frequency ('D' for daily)
    
    Returns:
        Dictionary of aligned DataFrames
    """
    if not df_dict:
        return {}
    
    # Find common date range
    all_dates = []
    for df in df_dict.values():
        if 'dt' in df.columns:
            dates = pd.to_datetime(df['dt'])
        else:
            dates = pd.to_datetime(df.index)
        all_dates.extend(dates.tolist())
    
    if not all_dates:
        return df_dict
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    common_index = pd.date_range(start=min_date, end=max_date, freq=target_freq)
    
    aligned_dict = {}
    for ticker, df in df_dict.items():
        df_aligned = df.copy()
        
        # Ensure dt is index
        if 'dt' in df_aligned.columns:
            df_aligned = df_aligned.set_index('dt')
        
        # Reindex to common dates
        df_aligned = df_aligned.reindex(common_index)
        
        # Forward fill based on frequency
        # Daily data: fill up to 5 days
        # Monthly/weekly: fill until next observation
        if target_freq == 'D':
            df_aligned['value'] = df_aligned['value'].ffill(limit=5)
        else:
            df_aligned['value'] = df_aligned['value'].ffill()
        
        aligned_dict[ticker] = df_aligned.reset_index().rename(columns={'index': 'dt'})
    
    return aligned_dict


def handle_missing_data(df, asset_class, max_fill_days=5):
    """
    Apply asset-class-specific missing data handling.
    
    Args:
        df: DataFrame with dt and value columns
        asset_class: Asset class
        max_fill_days: Maximum days to forward fill for daily data
    
    Returns:
        DataFrame with handled missing data
    """
    df = df.copy()
    df = df.sort_values('dt')
    
    # Determine frequency based on asset class
    if asset_class in ['EQUITY', 'FX', 'COMMODITIES', 'VOL', 'RATES']:
        # Daily data - forward fill with limit
        df = forward_fill_data(df, max_days=max_fill_days, freq='D')
    elif asset_class in ['MACRO', 'CREDIT']:
        # Monthly/weekly data - forward fill until next observation
        df = forward_fill_data(df, max_days=None, freq='D')
    else:
        # Default: forward fill with limit
        df = forward_fill_data(df, max_days=max_fill_days, freq='D')
    
    return df


def validate_data_quality(df, ticker, asset_class):
    """
    Run quality checks and return validation report.
    
    Args:
        df: DataFrame with dt and value columns
        ticker: Ticker symbol
        asset_class: Asset class
    
    Returns:
        Dictionary with validation results
    """
    report = {
        'ticker': ticker,
        'asset_class': asset_class,
        'total_rows': len(df),
        'missing_values': df['value'].isna().sum(),
        'missing_pct': df['value'].isna().sum() / len(df) * 100,
        'duplicate_dates': df['dt'].duplicated().sum(),
        'date_range': (df['dt'].min(), df['dt'].max()) if len(df) > 0 else None,
        'outliers_detected': 0,
        'warnings': []
    }
    
    # Check for duplicates
    if report['duplicate_dates'] > 0:
        report['warnings'].append(f"Found {report['duplicate_dates']} duplicate dates")
    
    # Check missing data rate
    if report['missing_pct'] > 50:
        report['warnings'].append(f"High missing data rate: {report['missing_pct']:.1f}%")
    
    # Detect outliers (if enough data)
    if len(df) > 10 and df['value'].notna().sum() > 10:
        outliers = detect_outliers(df[df['value'].notna()])
        report['outliers_detected'] = outliers.sum()
        if outliers.sum() > 0:
            report['warnings'].append(f"Detected {outliers.sum()} outliers")
    
    # Asset-class-specific checks
    if asset_class in ['EQUITY', 'FX', 'COMMODITIES']:
        if (df['value'] < 0).any():
            report['warnings'].append("Found negative prices")
    
    if asset_class == 'RATES':
        if (df['value'] < 0).any():
            report['warnings'].append("Found negative yields (may be valid in rare cases)")
    
    if asset_class == 'CREDIT':
        if (df['value'] < 0).any():
            report['warnings'].append("Found negative credit spreads (data error)")
    
    return report
