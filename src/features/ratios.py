"""
Ratio feature computations: style factors, cross-asset ratios, volatility ratios.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_style_factors(df_equity: Dict[str, pd.DataFrame]):
    """
    Compute style factor ratios from equity ETFs.
    
    Args:
        df_equity: Dictionary of {ticker: DataFrame} with dt index and value column
                  Expected tickers: IVW, IVE, XLU, XLF, RUT, GSPC
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    results = []
    
    # Style factor definitions
    style_factors = {
        'STYLE_GROWTH_VS_VALUE': ('IVW', 'IVE'),
        'STYLE_CYCLICAL_VS_DEFENSIVE': ('XLF', 'XLU'),
        'STYLE_SMALL_VS_LARGE': ('^RUT', '^GSPC'),
    }
    
    for feature_name, (ticker1, ticker2) in style_factors.items():
        # Handle ticker name variations
        ticker1_actual = ticker1.replace('^', '')
        ticker2_actual = ticker2.replace('^', '')
        
        if ticker1 not in df_equity and ticker1_actual not in df_equity:
            continue
        if ticker2 not in df_equity and ticker2_actual not in df_equity:
            continue
        
        # Get actual ticker keys
        key1 = ticker1 if ticker1 in df_equity else ticker1_actual
        key2 = ticker2 if ticker2 in df_equity else ticker2_actual
        
        df1 = df_equity[key1].copy()
        df2 = df_equity[key2].copy()
        
        # Ensure dt is index
        if 'dt' in df1.columns:
            df1 = df1.set_index('dt')
        if 'dt' in df2.columns:
            df2 = df2.set_index('dt')
        
        # Align dates
        common_dates = df1.index.intersection(df2.index)
        df1_aligned = df1.loc[common_dates]
        df2_aligned = df2.loc[common_dates]
        
        # Compute ratio: numerator / denominator
        ratio = df1_aligned['value'] / df2_aligned['value']
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'dt': ratio.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': feature_name,
            'value': ratio.values
        })
        
        # Remove NaN and infinite values
        result_df = result_df[result_df['value'].notna() & np.isfinite(result_df['value'])]
        
        results.append(result_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore


def compute_cross_asset_ratios(df_all: Dict[str, pd.DataFrame]):
    """
    Compute cross-asset ratio features.
    
    Args:
        df_all: Dictionary of {ticker: DataFrame} with dt index and value column
                Expected tickers: AUDUSD=X, USDJPY=X, GC=F, CL=F, EEM, GSPC
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    results = []
    
    # Cross-asset ratio definitions
    cross_asset_ratios = {
        'FX_AUD_JPY': ('AUDUSD=X', 'USDJPY=X'),  # AUD/USD / USD/JPY = AUD/JPY
        'COMMODITY_GOLD_OIL': ('GC=F', 'CL=F'),
        'EQUITY_EM_VS_US': ('EEM', '^GSPC'),
    }
    
    for feature_name, (ticker1, ticker2) in cross_asset_ratios.items():
        # Handle ticker name variations
        ticker1_actual = ticker1.replace('^', '')
        ticker2_actual = ticker2.replace('^', '')
        
        # Try different variations
        key1 = None
        key2 = None
        
        for key in df_all.keys():
            if key == ticker1 or key == ticker1_actual:
                key1 = key
            if key == ticker2 or key == ticker2_actual:
                key2 = key
        
        if key1 is None or key2 is None:
            continue
        
        df1 = df_all[key1].copy()
        df2 = df_all[key2].copy()
        
        # Ensure dt is index
        if 'dt' in df1.columns:
            df1 = df1.set_index('dt')
        if 'dt' in df2.columns:
            df2 = df2.set_index('dt')
        
        # Align dates
        common_dates = df1.index.intersection(df2.index)
        df1_aligned = df1.loc[common_dates]
        df2_aligned = df2.loc[common_dates]
        
        # Special handling for FX_AUD_JPY (division of two FX pairs)
        if feature_name == 'FX_AUD_JPY':
            # AUD/USD / USD/JPY = AUD/JPY
            ratio = df1_aligned['value'] / df2_aligned['value']
        else:
            # Standard ratio
            ratio = df1_aligned['value'] / df2_aligned['value']
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'dt': ratio.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': feature_name,
            'value': ratio.values
        })
        
        # Remove NaN and infinite values
        result_df = result_df[result_df['value'].notna() & np.isfinite(result_df['value'])]
        
        results.append(result_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore


def compute_volatility_features(df_vol: Dict[str, pd.DataFrame]):
    """
    Compute volatility-related features.
    
    Args:
        df_vol: Dictionary of {ticker: DataFrame} with dt index and value column
                Expected tickers: ^VIX, ^VVIX
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    results = []
    
    # VIX level
    vix_tickers = ['^VIX', 'VIX']
    vix_key = None
    for ticker in vix_tickers:
        if ticker in df_vol:
            vix_key = ticker
            break
    
    if vix_key:
        df_vix = df_vol[vix_key].copy()
        
        # Ensure dt is index
        if 'dt' in df_vix.columns:
            df_vix = df_vix.set_index('dt')
        
        # Store VIX level
        vix_level_df = pd.DataFrame({
            'dt': df_vix.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': 'VOL_VIX_LEVEL',
            'value': df_vix['value'].values
        })
        vix_level_df = vix_level_df[vix_level_df['value'].notna()]
        results.append(vix_level_df)
    
    # VVIX / VIX ratio (vol-of-vol)
    vvix_tickers = ['^VVIX', 'VVIX']
    vvix_key = None
    for ticker in vvix_tickers:
        if ticker in df_vol:
            vvix_key = ticker
            break
    
    if vix_key and vvix_key:
        df_vvix = df_vol[vvix_key].copy()
        
        # Ensure dt is index
        if 'dt' in df_vvix.columns:
            df_vvix = df_vvix.set_index('dt')
        
        # Align dates
        common_dates = df_vix.index.intersection(df_vvix.index)  # type: ignore
        df_vix_aligned = df_vix.loc[common_dates]  # type: ignore
        df_vvix_aligned = df_vvix.loc[common_dates]
        
        # Compute ratio
        ratio = df_vvix_aligned['value'] / df_vix_aligned['value']
        
        ratio_df = pd.DataFrame({
            'dt': ratio.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': 'VOL_VVIX_VIX',
            'value': ratio.values
        })
        ratio_df = ratio_df[ratio_df['value'].notna() & np.isfinite(ratio_df['value'])]
        results.append(ratio_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore
