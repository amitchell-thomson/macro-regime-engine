"""
Data quality diagnostics for feature engineering.
Provides tools to analyze feature completeness, frequency, and data quality.
"""

import pandas as pd
import numpy as np
from typing import Optional


def categorize_feature_frequency(conn, version: str = 'V2_FWD_FILL') -> pd.DataFrame:
    """
    Automatically detect the frequency of each feature based on observation gaps.
    
    Args:
        conn: Database connection
        version: Feature version to analyze
    
    Returns:
        DataFrame with frequency analysis for each feature
    """
    query = """
    SELECT 
        feature,
        ticker,
        asset_class,
        dt
    FROM macro.features
    WHERE version = %s
    ORDER BY feature, dt
    """
    
    df = pd.read_sql(query, conn, params=[version])
    df['dt'] = pd.to_datetime(df['dt'])
    
    results = []
    
    for feature in df['feature'].unique():
        feature_df = df[df['feature'] == feature].sort_values('dt')  # type: ignore
        ticker = feature_df['ticker'].iloc[0]  # type: ignore
        asset_class = feature_df['asset_class'].iloc[0]  # type: ignore
        
        # Calculate median gap between consecutive observations
        date_diffs = feature_df['dt'].diff().dropna()
        median_gap = date_diffs.median().days if len(date_diffs) > 0 else 0  # type: ignore
        
        # Categorize frequency
        if median_gap <= 5:
            frequency = 'DAILY'
            expected_behavior = 'Should be nearly complete'
        elif 5 < median_gap <= 10:
            frequency = 'WEEKLY'
            expected_behavior = 'Weekly updates'
        elif 20 <= median_gap <= 35:
            frequency = 'MONTHLY'
            expected_behavior = 'Sparse OK - monthly releases'
        elif 80 <= median_gap <= 100:
            frequency = 'QUARTERLY'
            expected_behavior = 'Sparse OK - quarterly releases'
        elif median_gap > 100:
            frequency = 'ANNUAL'
            expected_behavior = 'Very sparse - annual data'
        else:
            frequency = 'IRREGULAR'
            expected_behavior = 'Check data quality'
        
        # Count observations
        first_date = feature_df['dt'].min()
        last_date = feature_df['dt'].max()
        years_span = (last_date - first_date).days / 365.25  # type: ignore
        obs_per_year = len(feature_df) / years_span if years_span > 0 else 0
        
        results.append({
            'feature': feature,
            'ticker': ticker,
            'asset_class': asset_class,
            'frequency': frequency,
            'median_gap_days': median_gap,
            'observations': len(feature_df),
            'obs_per_year': round(obs_per_year, 1),
            'expected_behavior': expected_behavior,
            'first_date': first_date.date(),
            'last_date': last_date.date()
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary by frequency
    print("=" * 80)
    print("FEATURE FREQUENCY ANALYSIS")
    print("=" * 80)
    print(f"\nTotal features: {len(results_df)}")
    print(f"\nFrequency distribution:")
    print(results_df['frequency'].value_counts())
    print(f"\nObservations per year by frequency:")
    print(results_df.groupby('frequency')['obs_per_year'].describe())
    
    return results_df


def check_forward_fill_behavior(conn, version: str = 'V2_FWD_FILL') -> pd.DataFrame:
    """
    Check if forward-fill is creating data on weekends/holidays inappropriately.
    
    Args:
        conn: Database connection
        version: Feature version to analyze
    
    Returns:
        DataFrame with date analysis including day-of-week information
    """
    query = """
    SELECT 
        dt,
        COUNT(DISTINCT feature) as num_features
    FROM macro.features
    WHERE version = %s
    GROUP BY dt
    ORDER BY dt
    """
    
    df = pd.read_sql(query, conn, params=[version])
    df['dt'] = pd.to_datetime(df['dt'])
    df['day_of_week'] = df['dt'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    print("=" * 80)
    print("FORWARD-FILL BEHAVIOR CHECK")
    print("=" * 80)
    print(f"\nTotal dates in database: {len(df):,}")
    print(f"Weekend dates: {df['is_weekend'].sum()}")
    print(f"\nDay of week distribution:")
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day_num in range(7):
        count = (df['day_of_week'] == day_num).sum()
        print(f"  {day_names[day_num]}: {count:,}")
    
    if df['is_weekend'].sum() > 0:
        print("\n⚠️  WARNING: You have data on weekends!")
        print("This explains >100% completeness in earlier analysis.")
        print("\nSample weekend dates:")
        weekend_dates = df[df['is_weekend']].head(10)
        print(weekend_dates[['dt', 'num_features']])
        print("\n💡 RECOMMENDATION: Rerun compute_all_features with business_days_only=True")
    else:
        print("\n✅ No weekend data found - forward-fill is working correctly!")
    
    return df


def find_missing_dates(conn, feature_name: str, version: str = 'V2_FWD_FILL') -> Optional[pd.DataFrame]:
    """
    Find missing business days for a specific feature.
    
    Args:
        conn: Database connection
        feature_name: Name of feature to check
        version: Feature version to analyze
    
    Returns:
        DataFrame with missing dates, or None if feature not found
    """
    query = """
    SELECT dt
    FROM macro.features
    WHERE feature = %s 
      AND version = %s
    ORDER BY dt
    """
    
    df = pd.read_sql(query, conn, params=[feature_name, version])
    df['dt'] = pd.to_datetime(df['dt'])
    
    if df.empty:
        print(f"Feature '{feature_name}' not found!")
        return None
    
    # Generate expected business days
    first_date = df['dt'].min()
    last_date = df['dt'].max()
    expected_dates = pd.bdate_range(start=first_date, end=last_date)
    
    # Find missing
    actual_dates = set(df['dt'])
    missing_dates = [d for d in expected_dates if d not in actual_dates]
    
    print(f"Feature: {feature_name}")
    print(f"Date range: {first_date.date()} to {last_date.date()}")
    print(f"Expected business days: {len(expected_dates):,}")
    print(f"Actual days: {len(actual_dates):,}")
    print(f"Missing days: {len(missing_dates):,}")
    print(f"Completeness: {100 * len(actual_dates) / len(expected_dates):.1f}%")
    
    if missing_dates:
        print(f"\nFirst 20 missing dates:")
        for d in missing_dates[:20]:
            print(f"  {d.date()}")
        
        if len(missing_dates) > 20:
            print(f"  ... and {len(missing_dates) - 20:,} more")
    else:
        print("\n✅ No missing dates!")
    
    return pd.DataFrame({'missing_date': missing_dates})


def analyze_feature_completeness(conn, version: str = 'V2_FWD_FILL') -> pd.DataFrame:
    """
    Comprehensive analysis of feature start dates, coverage, and missing data.
    
    Args:
        conn: Database connection
        version: Feature version to analyze
    
    Returns:
        DataFrame with completeness metrics for each feature
    """
    query = """
    SELECT 
        feature,
        ticker,
        asset_class,
        dt,
        value
    FROM macro.features
    WHERE version = %s
    ORDER BY feature, ticker, dt
    """
    
    df = pd.read_sql(query, conn, params=[version])
    df['dt'] = pd.to_datetime(df['dt'])
    
    print(f"Loaded {len(df):,} feature rows")
    print(f"Date range: {df['dt'].min().date()} to {df['dt'].max().date()}")
    print(f"Unique features: {df['feature'].nunique()}\n")  # type: ignore
    
    # Summary by feature
    summary = []
    
    for feature in df['feature'].unique():
        feature_df = df[df['feature'] == feature]
        ticker = feature_df['ticker'].iloc[0]  # type: ignore
        asset_class = feature_df['asset_class'].iloc[0]  # type: ignore
        
        first_date = feature_df['dt'].min()
        last_date = feature_df['dt'].max()
        actual_rows = len(feature_df)
        
        # Calculate expected business days
        all_bdays = pd.bdate_range(start=first_date, end=last_date)
        expected_rows = len(all_bdays)
        
        # Check for gaps
        feature_df_sorted = feature_df.sort_values('dt')  # type: ignore
        date_diffs = feature_df_sorted['dt'].diff()
        max_gap = date_diffs.max().days if len(date_diffs) > 1 else 0  # type: ignore
        gaps = (date_diffs > pd.Timedelta(days=7)).sum()  # Gaps > 1 week
        
        summary.append({
            'feature': feature,
            'ticker': ticker,
            'asset_class': asset_class,
            'first_date': first_date.date(),
            'last_date': last_date.date(),
            'actual_rows': actual_rows,
            'expected_bdays': expected_rows,
            'pct_complete': round(100 * actual_rows / expected_rows, 1),
            'missing_days': expected_rows - actual_rows,
            'max_gap_days': max_gap,
            'large_gaps': gaps
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Display summary statistics
    print("=" * 80)
    print("FEATURE COMPLETENESS ANALYSIS")
    print("=" * 80)
    print(f"\nTotal features: {len(summary_df)}")
    print(f"\nStart date distribution:")
    print(summary_df['first_date'].value_counts().head(10))
    
    print(f"\nCompleteness summary:")
    print(summary_df['pct_complete'].describe())
    
    # Show features with potential issues
    print("\n" + "=" * 80)
    print("FEATURES WITH MISSING DATA (< 95% complete)")
    print("=" * 80)
    incomplete = summary_df[summary_df['pct_complete'] < 95].sort_values('pct_complete')  # type: ignore
    if len(incomplete) > 0:
        print(f"Found {len(incomplete)} features with <95% completeness\n")
        print(incomplete[['feature', 'ticker', 'first_date', 'actual_rows', 
                          'expected_bdays', 'pct_complete', 'missing_days', 'large_gaps']].to_string())
    else:
        print("✅ No features with significant missing data!")
    
    # Show features with large gaps
    print("\n" + "=" * 80)
    print("FEATURES WITH LARGE GAPS (> 7 days)")
    print("=" * 80)
    with_gaps = summary_df[summary_df['large_gaps'] > 0].sort_values('large_gaps', ascending=False)  # type: ignore
    if len(with_gaps) > 0:
        print(f"Found {len(with_gaps)} features with gaps > 7 days\n")
        print(with_gaps[['feature', 'ticker', 'max_gap_days', 'large_gaps']].head(20).to_string())
    else:
        print("✅ No features with large gaps!")
    
    return summary_df


def get_feature_start_dates(conn, version: str = 'V2_FWD_FILL') -> pd.DataFrame:
    """
    Get the start date of each feature for determining common date ranges.
    Useful for preparing ML datasets.
    
    Args:
        conn: Database connection
        version: Feature version to analyze
    
    Returns:
        DataFrame with feature, ticker, and first observation date
    """
    query = """
    SELECT 
        feature,
        ticker,
        MIN(dt) as first_date,
        COUNT(*) as total_observations
    FROM macro.features
    WHERE version = %s
    GROUP BY feature, ticker
    ORDER BY first_date, feature
    """
    
    df = pd.read_sql(query, conn, params=[version])
    df['first_date'] = pd.to_datetime(df['first_date'])
    
    print("=" * 80)
    print("FEATURE START DATES")
    print("=" * 80)
    print(f"\nTotal features: {len(df)}")
    print(f"\nEarliest start: {df['first_date'].min().date()}")
    print(f"Latest start: {df['first_date'].max().date()}")
    print(f"\nFeatures starting by common dates:")
    
    common_dates = ['2010-01-04', '2011-01-03', '2014-01-02', '2018-01-02']
    for date in common_dates:
        count = (df['first_date'] <= date).sum()
        print(f"  By {date}: {count} features")
    
    print(f"\nStart date distribution:")
    print(df['first_date'].value_counts().head(15))
    
    return df


def get_core_features(conn, version: str = 'V2_FWD_FILL', 
                      cutoff_date: str = '2010-01-05') -> list:
    """
    Get list of "core" features that started by a specific cutoff date.
    Useful for creating ML datasets with maximum historical coverage.
    
    Args:
        conn: Database connection
        version: Feature version to analyze
        cutoff_date: Only include features starting on or before this date
    
    Returns:
        List of feature names
    """
    df = get_feature_start_dates(conn, version)
    core_features = df[df['first_date'] <= cutoff_date]['feature'].tolist()
    
    print(f"\n" + "=" * 80)
    print(f"CORE FEATURES (started by {cutoff_date})")
    print("=" * 80)
    print(f"Core features: {len(core_features)}")
    print(f"Excluded features: {len(df) - len(core_features)}")
    
    excluded = df[df['first_date'] > cutoff_date]
    if len(excluded) > 0:
        print(f"\nExcluded features (started after {cutoff_date}):")
        for _, row in excluded.head(20).iterrows():
            print(f"  {row['feature']}: started {row['first_date'].date()}")  # type: ignore
        if len(excluded) > 20:
            print(f"  ... and {len(excluded) - 20} more")
    
    return core_features

