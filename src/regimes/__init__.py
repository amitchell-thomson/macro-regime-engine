"""
Regime detection module.

This module provides tools for detecting macro market regimes from cross-asset features.
Currently supports HMM-based regime detection with plans for additional methods in the future.
"""

from .features import get_regime_features, validate_feature_data, REGIME_FEATURES

__all__ = [
    'get_regime_features',
    'validate_feature_data',
    'REGIME_FEATURES',
]

