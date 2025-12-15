"""
Feature engineering module for macro regime detection.
"""

from .database import get_raw_series, upsert_features, get_features
from .compute import compute_all_features, forward_fill_features, detect_feature_frequency
from .basic import compute_all_basic_features
from .cleaning import clean_raw_data, handle_missing_data, align_frequencies
from .diagnostics import (
    categorize_feature_frequency,
    check_forward_fill_behavior,
    find_missing_dates,
    analyze_feature_completeness,
    get_feature_start_dates,
    get_core_features
)

__all__ = [
    'get_raw_series',
    'upsert_features',
    'get_features',
    'compute_all_features',
    'forward_fill_features',
    'detect_feature_frequency',
    'compute_all_basic_features',
    'clean_raw_data',
    'handle_missing_data',
    'align_frequencies',
    # Diagnostics
    'categorize_feature_frequency',
    'check_forward_fill_behavior',
    'find_missing_dates',
    'analyze_feature_completeness',
    'get_feature_start_dates',
    'get_core_features',
]
