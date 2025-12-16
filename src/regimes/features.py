"""
Feature selection and preparation for regime detection.

This module selects regime-relevant features from the FEATURES table and
prepares them in wide format (dates x features) for modeling.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import sys
import os

# Add parent directory to path to import from features module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.database import get_features


# Regime-relevant features covering four key dimensions:
# 1. Risk appetite (RISK_ON vs RISK_OFF)
# 2. Growth expectations (expansion vs contraction)
# 3. Inflation dynamics (INFLATION_SHOCK vs DISINFLATION)
# 4. Cross-asset flows and market structure

REGIME_FEATURES = [
    # Risk Appetite Indicators
    'VOL_VIX_LEVEL',                    # Fear gauge (high = RISK_OFF)
    'CREDIT_HY_CHG_20D',                # Credit spread changes (widening = RISK_OFF)
    '^GSPC_RET_20D',                    # Equity returns (positive = RISK_ON)
    'STYLE_CYCLICAL_VS_DEFENSIVE',      # XLF/XLU (high = RISK_ON)
    'STYLE_GROWTH_VS_VALUE',            # IVW/IVE (high = RISK_ON)
    
    # Growth Indicators
    'GLOBAL_YIELD_CURVE_SLOPE',         # 2s10s spread (steep = growth expectations)
    'USSLIND_CHG_MOM',                  # Leading indicators momentum
    
    # Inflation Indicators
    'GLOBAL_INFLATION_EXPECTATIONS',    # Breakeven inflation rates
    'CL=F_RET_20D',                     # Oil returns (proxy for inflation pressure)
    'REAL_RATE_10Y',                    # Real rates (negative = inflation concern)
    
    # Cross-Asset Flows
    'FX_AUD_JPY',                       # Risk-on/risk-off currency pair
    'COMMODITY_GOLD_OIL',               # Safe haven vs growth ratio
    'EQUITY_EM_VS_US',                  # EM risk appetite
    
    # Financial Conditions
    'GLOBAL_FINANCIAL_CONDITIONS',      # NFCI (positive = tight conditions)
]


def get_regime_features(
    conn,
    version: str = 'V2_FWD_FILL',
    start_date: Optional[str] = '2010-01-01',
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load regime-relevant features from FEATURES table and pivot to wide format.
    
    Args:
        conn: Database connection
        version: Feature version to load (default: 'V2_FWD_FILL')
        start_date: Start date for feature data (default: '2010-01-01')
        end_date: Optional end date for feature data
    
    Returns:
        df_wide: DataFrame with dates as index, features as columns
                 Shape: (n_dates, n_features)
    
    Raises:
        ValueError: If critical features are missing or data quality issues detected
    """
    print(f"Loading {len(REGIME_FEATURES)} regime features...")
    print(f"Version: {version}, Start date: {start_date}")
    
    # Load features from database (long format)
    df_long = get_features(
        conn,
        version=version,
        start_date=start_date,
        end_date=end_date
    )
    
    if df_long.empty:
        raise ValueError(f"No features found for version '{version}'")
    
    print(f"Loaded {len(df_long):,} rows from FEATURES table")
    
    # Filter to regime-relevant features only
    df_filtered = df_long[df_long['feature'].isin(REGIME_FEATURES)].copy()
    
    if df_filtered.empty:
        raise ValueError("No regime features found in database")
    
    # Check which features are present
    features_present = set(df_filtered['feature'].unique())  # type: ignore
    features_missing = set(REGIME_FEATURES) - features_present
    
    if features_missing:
        print(f"\n⚠️  Warning: {len(features_missing)} features missing:")
        for feat in sorted(features_missing):
            print(f"  - {feat}")
        print(f"\nContinuing with {len(features_present)} available features...")
    else:
        print(f"✅ All {len(REGIME_FEATURES)} features present")
    
    # Pivot to wide format: dates x features
    df_wide = df_filtered.pivot(
        index='dt',
        columns='feature',
        values='value'
    )
    
    # Sort by date
    df_wide = df_wide.sort_index()
    
    print(f"\nPivoted to wide format:")
    print(f"  Shape: {df_wide.shape} (dates x features)")
    print(f"  Date range: {df_wide.index.min()} to {df_wide.index.max()}")
    print(f"  Total days: {len(df_wide)}")
    
    # Check for missing values
    missing_counts = df_wide.isnull().sum()
    if missing_counts.any():
        print(f"\n⚠️  Warning: Missing values detected:")
        for feat, count in missing_counts[missing_counts > 0].items():
            pct = 100 * count / len(df_wide)
            print(f"  - {feat}: {count} ({pct:.1f}%)")
    else:
        print(f"\n✅ No missing values")
    
    return df_wide


def validate_feature_data(df_wide: pd.DataFrame) -> dict:
    """
    Check data quality: no NaNs, date continuity, reasonable ranges.
    
    Args:
        df_wide: Wide-format feature DataFrame (dates x features)
    
    Returns:
        validation_report: Dict with diagnostics and pass/fail status
    """
    report = {
        'passed': True,
        'checks': {},
        'warnings': [],
        'errors': []
    }
    
    # Check 1: No missing values
    missing_counts = df_wide.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        report['checks']['missing_values'] = False
        report['errors'].append(f"{total_missing} missing values found across features")
        report['passed'] = False
    else:
        report['checks']['missing_values'] = True
    
    # Check 2: Date continuity (business days)
    date_diffs = pd.Series(df_wide.index).diff().dropna()
    max_gap = date_diffs.max().days if len(date_diffs) > 0 else 0
    
    # Allow up to 4 days gap (long weekends)
    if max_gap > 4:
        report['checks']['date_continuity'] = False
        report['warnings'].append(f"Large date gap detected: {max_gap} days")
    else:
        report['checks']['date_continuity'] = True
    
    # Check 3: Sufficient data
    if len(df_wide) < 252:  # Less than 1 year of data
        report['checks']['sufficient_data'] = False
        report['errors'].append(f"Insufficient data: only {len(df_wide)} days")
        report['passed'] = False
    else:
        report['checks']['sufficient_data'] = True
    
    # Check 4: Feature variance (not all constant)
    zero_variance_features = []
    for col in df_wide.columns:
        if df_wide[col].std() == 0:
            zero_variance_features.append(col)
    
    if zero_variance_features:
        report['checks']['feature_variance'] = False
        report['warnings'].append(f"{len(zero_variance_features)} features have zero variance")
        report['passed'] = False
    else:
        report['checks']['feature_variance'] = True
    
    # Check 5: Extreme outliers (>10 std from mean)
    extreme_outliers = {}
    for col in df_wide.columns:
        z_scores = np.abs((df_wide[col] - df_wide[col].mean()) / df_wide[col].std())
        n_extreme = (z_scores > 10).sum()
        if n_extreme > 0:
            extreme_outliers[col] = n_extreme
    
    if extreme_outliers:
        report['checks']['extreme_outliers'] = False
        report['warnings'].append(f"{len(extreme_outliers)} features have extreme outliers (>10σ)")
    else:
        report['checks']['extreme_outliers'] = True
    
    # Summary statistics
    report['summary'] = {
        'n_dates': len(df_wide),
        'n_features': len(df_wide.columns),
        'date_range': (str(df_wide.index.min()), str(df_wide.index.max())),
        'total_missing': int(total_missing),
        'max_gap_days': int(max_gap),
        'zero_variance_features': zero_variance_features,
        'extreme_outliers': extreme_outliers
    }
    
    return report


def print_validation_report(report: dict) -> None:
    """
    Pretty print validation report.
    
    Args:
        report: Validation report from validate_feature_data()
    """
    print("\n" + "="*60)
    print("FEATURE DATA VALIDATION REPORT")
    print("="*60)
    
    # Overall status
    if report['passed']:
        print("\n✅ PASSED - Data is ready for modeling")
    else:
        print("\n❌ FAILED - Data quality issues detected")
    
    # Individual checks
    print("\nChecks:")
    for check, passed in report['checks'].items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    # Warnings
    if report['warnings']:
        print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
        for warning in report['warnings']:
            print(f"  - {warning}")
    
    # Errors
    if report['errors']:
        print(f"\n❌ Errors ({len(report['errors'])}):")
        for error in report['errors']:
            print(f"  - {error}")
    
    # Summary
    print("\nSummary:")
    summary = report['summary']
    print(f"  Dates: {summary['n_dates']} ({summary['date_range'][0]} to {summary['date_range'][1]})")
    print(f"  Features: {summary['n_features']}")
    print(f"  Missing values: {summary['total_missing']}")
    print(f"  Max gap: {summary['max_gap_days']} days")
    
    print("="*60 + "\n")

