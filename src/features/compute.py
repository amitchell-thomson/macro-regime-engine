"""
Main feature computation pipeline.
Orchestrates data loading, cleaning, feature computation, and storage.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime

from .database import get_raw_series, upsert_features
from .cleaning import clean_raw_data, handle_missing_data, align_frequencies, validate_data_quality
from .basic import compute_all_basic_features
from .spreads import compute_yield_curve_spreads, compute_credit_spreads, compute_real_rates
from .ratios import compute_style_factors, compute_cross_asset_ratios, compute_volatility_features
from .aggregates import (
    compute_risk_appetite_aggregates,
    compute_growth_aggregates,
    compute_inflation_aggregates,
    compute_monetary_aggregates
)


def load_raw_data(conn, ticker_filter=None, start_date=None, end_date=None):
    """
    Load raw data from database and organize by ticker and asset class.
    
    Args:
        conn: Database connection
        ticker_filter: Optional list of tickers to filter
        start_date: Optional start date
        end_date: Optional end date
    
    Returns:
        Dictionary of {ticker: DataFrame} where each DataFrame has dt and value columns
    """
    # Load all raw series
    df_raw = get_raw_series(
        conn,
        ticker=ticker_filter,
        start_date=start_date,
        end_date=end_date
    )
    
    if df_raw.empty:
        return {}
    
    # Organize by ticker
    data_dict = {}
    for ticker in df_raw['ticker'].unique():
        ticker_data = df_raw[df_raw['ticker'] == ticker].copy()
        ticker_data = ticker_data[['dt', 'value']].copy()
        ticker_data = ticker_data.sort_values('dt')  # type: ignore
        data_dict[ticker] = ticker_data
    
    return data_dict


def prepare_data_for_features(data_dict, asset_class_map):
    """
    Clean and prepare data for feature computation.
    
    Args:
        data_dict: Dictionary of {ticker: DataFrame}
        asset_class_map: Dictionary of {ticker: asset_class}
    
    Returns:
        Dictionary of cleaned data organized by asset class
    """
    cleaned_by_class = {
        'EQUITY': {},
        'RATES': {},
        'FX': {},
        'CREDIT': {},
        'COMMODITIES': {},
        'VOL': {},
        'MACRO': {},
        'ALL': {}
    }
    
    # Clean each ticker's data
    for ticker, df in data_dict.items():
        asset_class = asset_class_map.get(ticker, 'UNKNOWN')
        
        if asset_class == 'UNKNOWN':
            continue
        
        # Clean raw data
        df_cleaned = clean_raw_data(df, ticker, asset_class)
        
        # Handle missing data
        df_cleaned = handle_missing_data(df_cleaned, asset_class)
        
        # Store in appropriate category
        if asset_class in cleaned_by_class:
            cleaned_by_class[asset_class][ticker] = df_cleaned
        
        # Also store in ALL for cross-asset features
        cleaned_by_class['ALL'][ticker] = df_cleaned
    
    return cleaned_by_class


def compute_all_features(conn, version='V1_BASELINE', start_date=None, 
                         end_date=None, ticker_filter=None):
    """
    Main function to compute all features.
    
    Args:
        conn: Database connection
        version: Version string (e.g., 'V1_BASELINE')
        start_date: Optional start date
        end_date: Optional end date
        ticker_filter: Optional list of tickers to filter
    
    Returns:
        Dictionary with summary statistics
    """
    print(f"Starting feature computation (version: {version})...")
    
    # Load raw data
    print("Loading raw data...")
    data_dict = load_raw_data(conn, ticker_filter=ticker_filter, 
                              start_date=start_date, end_date=end_date)
    
    if not data_dict:
        print("No data found!")
        return {'status': 'error', 'message': 'No data found'}
    
    print(f"Loaded {len(data_dict)} tickers")
    
    # Create asset class mapping
    # Load from database to get accurate asset classes (batch query)
    df_all_tickers = get_raw_series(conn, ticker=list(data_dict.keys()))
    asset_class_map = {}
    if not df_all_tickers.empty:
        for ticker in df_all_tickers['ticker'].unique():
            ticker_data = df_all_tickers[df_all_tickers['ticker'] == ticker]
            if not ticker_data.empty:
                asset_class_map[ticker] = ticker_data['asset_class'].iloc[0]  # type: ignore
    
    # Prepare data
    print("Cleaning and preparing data...")
    cleaned_data = prepare_data_for_features(data_dict, asset_class_map)
    
    all_features = []
    
    # 1. Compute basic features for all tickers
    print("Computing basic features...")
    basic_count = 0
    for ticker, df in data_dict.items():
        asset_class = asset_class_map.get(ticker)
        if asset_class:
            # Ensure df has dt and value columns
            df_for_features = df.copy()
            if 'dt' not in df_for_features.columns:
                df_for_features = df_for_features.reset_index()
            df_features = compute_all_basic_features(df_for_features, ticker, asset_class)
            if not df_features.empty:
                all_features.append(df_features)
                basic_count += len(df_features)
    
    print(f"  Computed {basic_count} basic features")
    
    # 2. Compute spread features
    print("Computing spread features...")
    spread_features = []
    
    # Yield curve spreads
    if cleaned_data['RATES']:
        yield_spreads = compute_yield_curve_spreads(cleaned_data['RATES'])
        if not yield_spreads.empty:
            spread_features.append(yield_spreads)
    
    # Credit spreads
    if cleaned_data['CREDIT']:
        credit_spreads = compute_credit_spreads(cleaned_data['CREDIT'])
        if not credit_spreads.empty:
            spread_features.append(credit_spreads)
    
    # Real rates
    if cleaned_data['RATES']:
        real_rates = compute_real_rates(cleaned_data['RATES'])
        if not real_rates.empty:
            spread_features.append(real_rates)
    
    if spread_features:
        combined_spreads = pd.concat(spread_features, ignore_index=True)
        all_features.append(combined_spreads)
        print(f"  Computed {len(combined_spreads)} spread features")
    
    # 3. Compute ratio features
    print("Computing ratio features...")
    ratio_features = []
    
    # Style factors
    if cleaned_data['EQUITY']:
        style_factors = compute_style_factors(cleaned_data['EQUITY'])
        if not style_factors.empty:
            ratio_features.append(style_factors)
    
    # Cross-asset ratios
    if cleaned_data['ALL']:
        cross_ratios = compute_cross_asset_ratios(cleaned_data['ALL'])
        if not cross_ratios.empty:
            ratio_features.append(cross_ratios)
    
    # Volatility features
    if cleaned_data['VOL']:
        vol_features = compute_volatility_features(cleaned_data['VOL'])
        if not vol_features.empty:
            ratio_features.append(vol_features)
    
    if ratio_features:
        combined_ratios = pd.concat(ratio_features, ignore_index=True)
        all_features.append(combined_ratios)
        print(f"  Computed {len(combined_ratios)} ratio features")
    
    # 4. Compute global aggregates
    print("Computing global aggregates...")
    aggregate_features = []
    
    # Risk appetite
    if cleaned_data['ALL']:
        risk_appetite = compute_risk_appetite_aggregates(cleaned_data['ALL'])
        if not risk_appetite.empty:
            aggregate_features.append(risk_appetite)
    
    # Growth
    if cleaned_data['MACRO']:
        growth = compute_growth_aggregates(cleaned_data['MACRO'])
        if not growth.empty:
            aggregate_features.append(growth)
    
    # Inflation
    if cleaned_data['MACRO'] and cleaned_data['RATES']:
        inflation = compute_inflation_aggregates(cleaned_data['MACRO'], cleaned_data['RATES'])
        if not inflation.empty:
            aggregate_features.append(inflation)
    
    # Monetary
    if cleaned_data['RATES'] and cleaned_data['MACRO']:
        monetary = compute_monetary_aggregates(cleaned_data['RATES'], cleaned_data['MACRO'])
        if not monetary.empty:
            aggregate_features.append(monetary)
    
    if aggregate_features:
        combined_aggregates = pd.concat(aggregate_features, ignore_index=True)
        all_features.append(combined_aggregates)
        print(f"  Computed {len(combined_aggregates)} aggregate features")
    
    # Combine all features
    if not all_features:
        print("No features computed!")
        return {'status': 'error', 'message': 'No features computed'}
    
    print("Combining all features...")
    all_features_df = pd.concat(all_features, ignore_index=True)
    
    # Ensure dt is date type
    if 'dt' in all_features_df.columns:
        all_features_df['dt'] = pd.to_datetime(all_features_df['dt']).dt.date
    
    print(f"Total features computed: {len(all_features_df)}")
    print(f"Unique features: {all_features_df['feature'].nunique()}")
    print(f"Date range: {all_features_df['dt'].min()} to {all_features_df['dt'].max()}")
    
    # Store in database
    print(f"Storing features in database (version: {version})...")
    rows_stored = upsert_features(conn, all_features_df, version)
    print(f"Stored {rows_stored} feature rows")
    
    return {
        'status': 'success',
        'version': version,
        'total_features': len(all_features_df),
        'unique_features': all_features_df['feature'].nunique(),
        'rows_stored': rows_stored,
        'date_range': (str(all_features_df['dt'].min()), str(all_features_df['dt'].max()))
    }
