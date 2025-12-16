"""
Regime detection module.

This module provides tools for detecting macro market regimes from cross-asset features.
Currently supports HMM-based regime detection with plans for additional methods in the future.
"""

from .feature_selection import (
    select_regime_features,
    ECONOMIC_BLOCKS_DEFAULT,
    economic_prefilter,
    prune_redundant_features,
    apply_block_pca
)

__all__ = [
    'select_regime_features',
    'ECONOMIC_BLOCKS_DEFAULT',
    'economic_prefilter',
    'prune_redundant_features',
    'apply_block_pca',
]

