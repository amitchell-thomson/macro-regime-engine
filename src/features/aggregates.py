"""
Global aggregate feature computations: risk appetite, growth, inflation, monetary policy composites.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_risk_appetite_aggregates(df_all: Dict[str, pd.DataFrame]):
    """
    Compute risk appetite composite from equities, credit spreads, VIX.
    
    Args:
        df_all: Dictionary of {ticker: DataFrame} with dt index and value column
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    results = []
    
    # Equity momentum components
    equity_tickers = ['^GSPC', 'GSPC', '^DJI', 'DJI', '^IXIC', 'IXIC']
    equity_dfs = []
    
    for ticker in equity_tickers:
        if ticker in df_all:
            df = df_all[ticker].copy()
            if 'dt' in df.columns:
                df = df.set_index('dt')
            # Compute 20-day returns
            returns = df['value'].pct_change(periods=20)
            equity_dfs.append(returns)
            break  # Use first available
    
    # Credit spread component (inverse - tighter spreads = higher risk appetite)
    credit_tickers = ['BAMLH0A0HYM2']
    credit_df = None
    for ticker in credit_tickers:
        if ticker in df_all:
            df = df_all[ticker].copy()
            if 'dt' in df.columns:
                df = df.set_index('dt')
            # Use negative spread change (tighter = positive signal)
            credit_df = -df['value'].diff(periods=20)
            break
    
    # VIX component (inverse - lower VIX = higher risk appetite)
    vix_tickers = ['^VIX', 'VIX']
    vix_df = None
    for ticker in vix_tickers:
        if ticker in df_all:
            df = df_all[ticker].copy()
            if 'dt' in df.columns:
                df = df.set_index('dt')
            # Use negative VIX change
            vix_df = -df['value'].diff(periods=20)
            break
    
    # Combine components (simple average after normalization)
    components = []
    if equity_dfs:
        components.append(equity_dfs[0])
    if credit_df is not None:
        components.append(credit_df)
    if vix_df is not None:
        components.append(vix_df)
    
    if components:
        # Align dates
        common_dates = components[0].index
        for comp in components[1:]:
            common_dates = common_dates.intersection(comp.index)
        
        # Normalize each component (z-score) and average
        normalized = []
        for comp in components:
            comp_aligned = comp.loc[common_dates]
            mean = comp_aligned.mean()
            std = comp_aligned.std()
            if std > 0:
                normalized.append((comp_aligned - mean) / std)
            else:
                normalized.append(comp_aligned - mean)
        
        if normalized:
            risk_appetite = pd.Series(
                np.mean([n.values for n in normalized], axis=0),
                index=common_dates
            )
            
            result_df = pd.DataFrame({
                'dt': risk_appetite.index,
                'ticker': 'GLOBAL',
                'asset_class': 'MACRO',
                'feature': 'GLOBAL_RISK_APPETITE',
                'value': risk_appetite.values
            })
            result_df = result_df[result_df['value'].notna()]
            results.append(result_df)
    
    # Global equity momentum (average across major indices)
    equity_momentum_dfs = []
    for ticker in equity_tickers:
        if ticker in df_all:
            df = df_all[ticker].copy()
            if 'dt' in df.columns:
                df = df.set_index('dt')
            returns = df['value'].pct_change(periods=20)
            equity_momentum_dfs.append(returns)
    
    if equity_momentum_dfs:
        # Align dates
        common_dates = equity_momentum_dfs[0].index
        for df in equity_momentum_dfs[1:]:
            common_dates = common_dates.intersection(df.index)
        
        # Average momentum
        values = [df.loc[common_dates].values for df in equity_momentum_dfs]
        avg_momentum = pd.Series(np.mean(values, axis=0), index=common_dates)
        
        momentum_df = pd.DataFrame({
            'dt': avg_momentum.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': 'GLOBAL_EQUITY_MOMENTUM',
            'value': avg_momentum.values
        })
        momentum_df = momentum_df[momentum_df['value'].notna()]
        results.append(momentum_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore


def compute_growth_aggregates(df_macro: Dict[str, pd.DataFrame]):
    """
    Compute growth composite from leading indicators, sentiment.
    
    Args:
        df_macro: Dictionary of {ticker: DataFrame} with dt index and value column
                 Expected tickers: USSLIND, UMCSENT
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    results = []
    
    # Leading indicators
    leading_tickers = ['USSLIND']
    sentiment_tickers = ['UMCSENT']
    
    components = []
    
    # Leading index
    for ticker in leading_tickers:
        if ticker in df_macro:
            df = df_macro[ticker].copy()
            if 'dt' in df.columns:
                df = df.set_index('dt')
            # Use change in leading index
            change = df['value'].diff(periods=20)
            components.append(change)
            break
    
    # Consumer sentiment
    for ticker in sentiment_tickers:
        if ticker in df_macro:
            df = df_macro[ticker].copy()
            if 'dt' in df.columns:
                df = df.set_index('dt')
            # Use change in sentiment
            change = df['value'].diff(periods=20)
            components.append(change)
            break
    
    if components:
        # Align dates
        common_dates = components[0].index
        for comp in components[1:]:
            common_dates = common_dates.intersection(comp.index)
        
        # Normalize and average
        normalized = []
        for comp in components:
            comp_aligned = comp.loc[common_dates]
            mean = comp_aligned.mean()
            std = comp_aligned.std()
            if std > 0:
                normalized.append((comp_aligned - mean) / std)
            else:
                normalized.append(comp_aligned - mean)
        
        if normalized:
            growth_signal = pd.Series(
                np.mean([n.values for n in normalized], axis=0),
                index=common_dates
            )
            
            result_df = pd.DataFrame({
                'dt': growth_signal.index,
                'ticker': 'GLOBAL',
                'asset_class': 'MACRO',
                'feature': 'GLOBAL_GROWTH_SIGNAL',
                'value': growth_signal.values
            })
            result_df = result_df[result_df['value'].notna()]
            results.append(result_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore


def compute_inflation_aggregates(df_macro: Dict[str, pd.DataFrame], df_rates: Dict[str, pd.DataFrame]):
    """
    Compute inflation aggregates from CPI, PCE, breakeven inflation.
    
    Args:
        df_macro: Dictionary of {ticker: DataFrame} with dt index and value column
                 Expected tickers: CPIAUCSL, PCEPI
        df_rates: Dictionary of {ticker: DataFrame} with dt index and value column
                 Expected tickers: T5YIE, T10YIE
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    results = []
    
    # Realized inflation components
    inflation_tickers = ['CPIAUCSL', 'PCEPI']
    inflation_dfs = []
    
    for ticker in inflation_tickers:
        if ticker in df_macro:
            df = df_macro[ticker].copy()
            if 'dt' in df.columns:
                df = df.set_index('dt')
            # Compute YoY change (monthly data, so use 12-period change)
            yoy_change = df['value'].pct_change(periods=12) * 100
            inflation_dfs.append(yoy_change)
    
    if inflation_dfs:
        # Align dates
        common_dates = inflation_dfs[0].index
        for df in inflation_dfs[1:]:
            common_dates = common_dates.intersection(df.index)
        
        # Average realized inflation
        values = [df.loc[common_dates].values for df in inflation_dfs]
        avg_realized = pd.Series(np.mean(values, axis=0), index=common_dates)
        
        realized_df = pd.DataFrame({
            'dt': avg_realized.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': 'GLOBAL_INFLATION_REALIZED',
            'value': avg_realized.values
        })
        realized_df = realized_df[realized_df['value'].notna()]
        results.append(realized_df)
    
    # Inflation expectations (from breakeven rates)
    breakeven_tickers = ['T5YIE', 'T10YIE']
    breakeven_dfs = []
    
    for ticker in breakeven_tickers:
        if ticker in df_rates:
            df = df_rates[ticker].copy()
            if 'dt' in df.columns:
                df = df.set_index('dt')
            breakeven_dfs.append(df['value'])
    
    if breakeven_dfs:
        # Align dates
        common_dates = breakeven_dfs[0].index
        for df in breakeven_dfs[1:]:
            common_dates = common_dates.intersection(df.index)
        
        # Average breakeven inflation
        values = [df.loc[common_dates].values for df in breakeven_dfs]
        avg_expectations = pd.Series(np.mean(values, axis=0), index=common_dates)
        
        expectations_df = pd.DataFrame({
            'dt': avg_expectations.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': 'GLOBAL_INFLATION_EXPECTATIONS',
            'value': avg_expectations.values
        })
        expectations_df = expectations_df[expectations_df['value'].notna()]
        results.append(expectations_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore


def compute_monetary_aggregates(df_rates: Dict[str, pd.DataFrame], df_macro: Dict[str, pd.DataFrame]):
    """
    Compute monetary policy aggregates from yield curve, financial conditions.
    
    Args:
        df_rates: Dictionary of {ticker: DataFrame} with dt index and value column
                 Expected tickers: DGS2, DGS10
        df_macro: Dictionary of {ticker: DataFrame} with dt index and value column
                 Expected tickers: NFCI
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value
    """
    results = []
    
    # Yield curve slope (2s10s)
    if 'DGS2' in df_rates and 'DGS10' in df_rates:
        df_2y = df_rates['DGS2'].copy()
        df_10y = df_rates['DGS10'].copy()
        
        if 'dt' in df_2y.columns:
            df_2y = df_2y.set_index('dt')
        if 'dt' in df_10y.columns:
            df_10y = df_10y.set_index('dt')
        
        common_dates = df_2y.index.intersection(df_10y.index)
        slope = df_10y.loc[common_dates, 'value'] - df_2y.loc[common_dates, 'value']
        
        slope_df = pd.DataFrame({
            'dt': slope.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': 'GLOBAL_YIELD_CURVE_SLOPE',
            'value': slope.values
        })
        slope_df = slope_df[slope_df['value'].notna()]
        results.append(slope_df)
    
    # Financial conditions index
    if 'NFCI' in df_macro:
        df_nfci = df_macro['NFCI'].copy()
        
        if 'dt' in df_nfci.columns:
            df_nfci = df_nfci.set_index('dt')
        
        nfci_df = pd.DataFrame({
            'dt': df_nfci.index,
            'ticker': 'GLOBAL',
            'asset_class': 'MACRO',
            'feature': 'GLOBAL_FINANCIAL_CONDITIONS',
            'value': df_nfci['value'].values
        })
        nfci_df = nfci_df[nfci_df['value'].notna()]
        results.append(nfci_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value'])  # type: ignore
