"""
Spread feature computations: yield curve spreads, credit spreads, real rates.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_yield_curve_spreads(df_rates: Dict[str, pd.DataFrame]):
    """
    Compute yield curve spreads from rate data.
    
    Args:
        df_rates: Dictionary of {ticker: DataFrame} with dt index and value column
                 Expected tickers: DGS1, DGS2, DGS5, DGS10, DGS30
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    results = []
    
    # Required tickers for yield curve spreads
    required_tickers = {
        'YCURVE_2S10S': ['DGS2', 'DGS10'],
        'YCURVE_5S30S': ['DGS5', 'DGS30'],
        'YCURVE_2S5S': ['DGS2', 'DGS5'],
    }
    
    for spread_name, tickers in required_tickers.items():
        if tickers[0] not in df_rates or tickers[1] not in df_rates:
            continue
        
        df1 = df_rates[tickers[0]].copy()
        df2 = df_rates[tickers[1]].copy()
        
        # Ensure dt is index
        if 'dt' in df1.columns:
            df1 = df1.set_index('dt')
        if 'dt' in df2.columns:
            df2 = df2.set_index('dt')
        
        # Align dates
        common_dates = df1.index.intersection(df2.index)
        df1_aligned = df1.loc[common_dates]
        df2_aligned = df2.loc[common_dates]
        
        # Compute spread: long_rate - short_rate
        spread = df2_aligned['value'] - df1_aligned['value']
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'dt': spread.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': spread_name,
            'value': spread.values
        })
        
        # Remove NaN values
        result_df = result_df[result_df['value'].notna()]
        
        results.append(result_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore


def compute_credit_spreads(df_credit: Dict[str, pd.DataFrame]):
    """
    Compute credit spread features.
    
    Args:
        df_credit: Dictionary of {ticker: DataFrame} with dt index and value column
                  Expected tickers: BAMLH0A0HYM2, BAMLC0A0CM
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    results = []
    
    # Credit spread levels
    credit_tickers = {
        'CREDIT_HY_SPREAD': 'BAMLH0A0HYM2',
        'CREDIT_IG_SPREAD': 'BAMLC0A0CM',
    }
    
    for feature_name, ticker in credit_tickers.items():
        if ticker not in df_credit:
            continue
        
        df = df_credit[ticker].copy()
        
        # Ensure dt is index
        if 'dt' in df.columns:
            df = df.set_index('dt')
        
        # Store level
        level_df = pd.DataFrame({
            'dt': df.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': feature_name,
            'value': df['value'].values
        })
        level_df = level_df[level_df['value'].notna()]
        results.append(level_df)
        
        # Compute 20-day change
        change_20d = df['value'].diff(periods=20)
        change_df = pd.DataFrame({
            'dt': change_20d.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': feature_name.replace('_SPREAD', '_CHG_20D'),
            'value': change_20d.values
        })
        change_df = change_df[change_df['value'].notna()]
        results.append(change_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore


def compute_real_rates(df_rates: Dict[str, pd.DataFrame]):
    """
    Compute real rate features from inflation expectations and TIPS.
    
    Args:
        df_rates: Dictionary of {ticker: DataFrame} with dt index and value column
                 Expected tickers: DFII10 (10Y real rate), T5YIE, T10YIE
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    results = []
    
    # Real rate features
    if 'DFII10' in df_rates:
        df_real = df_rates['DFII10'].copy()
        
        # Ensure dt is index
        if 'dt' in df_real.columns:
            df_real = df_real.set_index('dt')
        
        # Store 10Y real rate level
        real_rate_df = pd.DataFrame({
            'dt': df_real.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': 'REAL_RATE_10Y',
            'value': df_real['value'].values
        })
        real_rate_df = real_rate_df[real_rate_df['value'].notna()]
        results.append(real_rate_df)
    
    # Average breakeven inflation
    breakeven_tickers = []
    if 'T5YIE' in df_rates:
        breakeven_tickers.append('T5YIE')
    if 'T10YIE' in df_rates:
        breakeven_tickers.append('T10YIE')
    
    if len(breakeven_tickers) >= 1:
        # Use available breakeven rates
        dfs = []
        for ticker in breakeven_tickers:
            df = df_rates[ticker].copy()
            if 'dt' in df.columns:
                df = df.set_index('dt')
            dfs.append(df)
        
        # Align dates and compute average
        if len(dfs) == 1:
            avg_breakeven = dfs[0]['value']
        else:
            # Align and average
            common_dates = dfs[0].index
            for df in dfs[1:]:
                common_dates = common_dates.intersection(df.index)
            
            values = []
            for df in dfs:
                values.append(df.loc[common_dates, 'value'])
            avg_breakeven = pd.Series(np.mean(values, axis=0), index=common_dates)
        
        breakeven_df = pd.DataFrame({
            'dt': avg_breakeven.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': 'GLOBAL_INFLATION_EXPECTATIONS',
            'value': avg_breakeven.values
        })
        breakeven_df = breakeven_df[breakeven_df['value'].notna()]
        results.append(breakeven_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore
