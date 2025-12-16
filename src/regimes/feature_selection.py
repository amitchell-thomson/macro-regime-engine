"""
3-stage feature selection pipeline for regime detection.

Reduces 376 engineered features to 20-40 optimal features using:
1. Economic prefiltering (domain knowledge)
2. Statistical redundancy pruning (correlation-based)
3. PCA within blocks (dimensionality reduction)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import re
import sys
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import from features module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.database import get_features


# Default critical features - can be overridden in notebook
# This list is provided as a reference but should be defined in the notebook
DEFAULT_CRITICAL_FEATURES = None  # Set to None to require explicit whitelist


def _categorize_features(all_feature_names: List[str]) -> Dict:
    """
    Automatically categorize all features into economic blocks based on naming patterns.
    
    Args:
        all_feature_names: List of all feature names from database
    
    Returns:
        Dict mapping block names to feature lists with selection rules
    """
    blocks = {
        'EQUITY_RISK': {'features': [], 'keep': 6, 'priority': 'variance'},
        'VOLATILITY': {'features': [], 'keep': 6, 'priority': 'varaince'},
        'CREDIT': {'features': [], 'keep': 5, 'priority': 'variance'},
        'RATES_CURVE': {'features': [], 'keep': 9, 'priority': 'variance'},
        'INFLATION_EXPECTATIONS': {'features': [], 'keep': 5, 'priority': 'variance'},
        'INFLATION_REALIZED': {'features': [], 'keep': 5, 'priority': 'variance'},
        'COMMODITIES': {'features': [], 'keep': 6, 'priority': 'variance'},
        'FX': {'features': [], 'keep': 5, 'priority': 'variance'},
        'STYLE_FACTORS': {'features': [], 'keep': 3, 'priority': 'all'},
        'CROSS_ASSET': {'features': [], 'keep': 2, 'priority': 'all'},
        'GROWTH_INDICATORS': {'features': [], 'keep': 5, 'priority': 'variance'},
        'FINANCIAL_CONDITIONS': {'features': [], 'keep': 5, 'priority': 'variance'},
    }
    
    # Define patterns for each block
    for feature in all_feature_names:
        categorized = False
        
        # VOLATILITY - VIX, VVIX, MOVE, vol-related
        if any(x in feature for x in ['VOL_VIX', 'VOL_VVIX', '^VIX', '^VVIX', '^MOVE', 'VOL_MOVE']):
            blocks['VOLATILITY']['features'].append(feature)
            categorized = True
        
        # CREDIT - credit spreads, HY, IG
        elif any(x in feature for x in ['CREDIT_HY', 'CREDIT_IG', 'BAMLH0A0HYM2', 'BAMLC0A0CM']):
            blocks['CREDIT']['features'].append(feature)
            categorized = True
        
        # STYLE FACTORS - explicit style ratios
        elif any(x in feature for x in ['STYLE_GROWTH_VS_VALUE', 'STYLE_CYCLICAL_VS_DEFENSIVE', 'STYLE_SMALL_VS_LARGE']):
            blocks['STYLE_FACTORS']['features'].append(feature)
            categorized = True
        
        # INFLATION EXPECTATIONS - breakeven rates, TIPS
        elif any(x in feature for x in ['T5YIE', 'T10YIE', 'GLOBAL_INFLATION_EXPECTATIONS']):
            blocks['INFLATION_EXPECTATIONS']['features'].append(feature)
            categorized = True
        
        # INFLATION REALIZED - CPI, PCE (both changes and levels)
        elif any(x in feature for x in ['CPIAUCSL', 'PCEPI']):
            blocks['INFLATION_REALIZED']['features'].append(feature)
            categorized = True
        
        # RATES & YIELD CURVE - Treasury yields, SOFR, Fed Funds, Real Rates, alternative curve measures
        elif any(x in feature for x in ['YCURVE', 'DGS', 'FEDFUNDS', 'SOFR', 'REAL_RATE', 'DFII', 'T10Y2Y', 'T10Y3M', 'TEDRATE']):
            blocks['RATES_CURVE']['features'].append(feature)
            categorized = True
        
        # COMMODITIES - oil, gold, copper, silver, etc.
        elif any(x in feature for x in ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'BZ=F', 'NG=F', 'COMMODITY_GOLD_OIL', 'DCOILWTICO']):
            blocks['COMMODITIES']['features'].append(feature)
            categorized = True
        
        # FX - currency pairs, dollar index
        elif any(x in feature for x in ['FX_AUD_JPY', 'DX-Y.NYB', 'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCNY=X']):
            blocks['FX']['features'].append(feature)
            categorized = True
        
        # CROSS ASSET - EM vs US, gold/oil ratios not already categorized
        elif 'EQUITY_EM_VS_US' in feature:
            blocks['CROSS_ASSET']['features'].append(feature)
            categorized = True
        
        # GROWTH INDICATORS - GDP, unemployment, sentiment, leading index
        elif any(x in feature for x in ['USSLIND', 'GDP', 'UMCSENT', 'UNRATE']):
            blocks['GROWTH_INDICATORS']['features'].append(feature)
            categorized = True
        
        # FINANCIAL CONDITIONS - NFCI, money supply, Fed balance sheet
        elif any(x in feature for x in ['GLOBAL_FINANCIAL_CONDITIONS', 'NFCI', 'M2SL', 'WALCL']):
            blocks['FINANCIAL_CONDITIONS']['features'].append(feature)
            categorized = True
        
        # EQUITY RISK - equity indices, sector ETFs (everything else with returns/vol)
        elif not categorized:
            # Equity tickers and their features
            equity_tickers = ['^GSPC', '^DJI', '^IXIC', '^RUT', 'EEM', '^FTSE', '^N225', 
                            '000001.SS', '^STOXX50E', 'IVW', 'IVE', 'XLU', 'XLF', 'BTC-USD']
            if any(ticker in feature for ticker in equity_tickers):
                blocks['EQUITY_RISK']['features'].append(feature)
                categorized = True
        
        # If still not categorized, try to guess based on feature type
        if not categorized:
            if 'RET' in feature or 'VOL' in feature or 'MOM' in feature:
                blocks['EQUITY_RISK']['features'].append(feature)
            elif 'GLOBAL' in feature:
                blocks['FINANCIAL_CONDITIONS']['features'].append(feature)
            # Otherwise skip (uncategorized features)
    
    return blocks


def _build_economic_blocks(all_feature_names: List[str]) -> Dict:
    """
    Build ECONOMIC_BLOCKS dynamically from all available features.
    
    Args:
        all_feature_names: List of all feature names
    
    Returns:
        Economic blocks dictionary
    """
    return _categorize_features(all_feature_names)


def _load_all_features(conn, version: str, start_date: str, 
                       max_nan_pct: float = 0.10) -> pd.DataFrame:
    """
    Load all features from database and pivot to wide format.
    Cleans NaN values caused by different feature start dates.
    
    Args:
        conn: Database connection
        version: Feature version
        start_date: Start date
        max_nan_pct: Maximum percentage of NaN values allowed per feature (default 10%)
    
    Returns:
        DataFrame with dates as index, features as columns (cleaned of NaNs)
    """
    print(f"Loading all features from database...")
    df_long = get_features(conn, version=version, start_date=start_date)
    
    if df_long.empty:
        raise ValueError(f"No features found for version '{version}'")
    
    print(f"  Loaded {len(df_long):,} rows")
    
    # Pivot to wide format
    df_wide = df_long.pivot(index='dt', columns='feature', values='value')
    df_wide = df_wide.sort_index()
    
    print(f"  Pivoted to {df_wide.shape[0]:,} dates × {df_wide.shape[1]} features")
    
    # Clean NaN values
    print(f"\nCleaning NaN values...")
    initial_shape = df_wide.shape
    
    # Step 1: Drop features with too many NaN values
    nan_counts = df_wide.isnull().sum()
    nan_pct = nan_counts / len(df_wide)
    features_to_drop = nan_pct[nan_pct > max_nan_pct].index.tolist()
    
    if features_to_drop:
        print(f"  Dropping {len(features_to_drop)} features with >{max_nan_pct:.0%} NaN values")
        df_wide = df_wide.drop(columns=features_to_drop)
    
    # Step 2: Drop rows with any remaining NaN values
    # This typically drops early rows where some features haven't started yet
    rows_before = len(df_wide)
    df_wide = df_wide.dropna()
    rows_dropped = rows_before - len(df_wide)
    
    if rows_dropped > 0:
        print(f"  Dropped {rows_dropped} rows with NaN values (early dates)")
        print(f"  New date range: {df_wide.index.min()} to {df_wide.index.max()}")
    
    print(f"  Final shape: {df_wide.shape[0]:,} dates × {df_wide.shape[1]} features")
    
    # Final validation - ensure no NaN values
    final_nan_count = df_wide.isnull().sum().sum()
    if final_nan_count > 0:
        print(f"  ⚠️  Warning: {final_nan_count} NaN values still present!")
        # Force drop all NaN
        df_wide = df_wide.dropna(how='any')
        print(f"  Forced dropna: {len(df_wide)} dates remaining")
    else:
        print(f"  ✅ No NaN values remaining")
    
    return df_wide


def _select_top_features(df_block: pd.DataFrame, n_keep: int, priority: str) -> List[str]:
    """
    Select top N features from a block based on priority rule.
    
    Args:
        df_block: DataFrame with features from a single block
        n_keep: Number of features to keep
        priority: Selection rule ('variance', 'all', etc.)
    
    Returns:
        List of selected feature names
    """
    if priority == 'all':
        return list(df_block.columns)
    
    if priority == 'variance':
        # Select features with highest variance (most information)
        variances = df_block.var().sort_values(ascending=False) # type: ignore
        return list[str](variances.head(n_keep).index)
    
    # Default: return first N
    return list[str](df_block.columns[:n_keep])


def economic_prefilter(df_all_features: pd.DataFrame, 
                       blocks: Optional[Dict] = None,
                       whitelist: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Stage 1: Select 5-6 most regime-relevant features per economic block.
    
    Prioritizes whitelisted features, then fills remaining slots by variance.
    
    Args:
        df_all_features: DataFrame with all features (dates x features)
        blocks: Dict defining economic blocks and selection criteria
        whitelist: Optional list of critical features to prioritize
    
    Returns:
        df_filtered: DataFrame with ~40-50 features
        block_assignments: Dict mapping selected features to their blocks
        stage_report: Dict with selection statistics
    """
    print("\n" + "="*60)
    print("STAGE 1: ECONOMIC PREFILTER")
    print("="*60)
    
    # Initialize whitelist
    if whitelist is None:
        whitelist = []
    
    # Filter whitelist to available features
    active_whitelist = [f for f in whitelist if f in df_all_features.columns]
    
    if active_whitelist:
        print(f"  🔒 Prioritizing {len(active_whitelist)} whitelisted features")
    
    # Auto-build blocks from available features if not provided
    if blocks is None:
        print("  Auto-categorizing features into economic blocks...")
        blocks = _build_economic_blocks(list(df_all_features.columns))
        
        # Report categorization
        categorized_features = set()
        for block_config in blocks.values():
            categorized_features.update(block_config['features'])
        
        all_features = set(df_all_features.columns)
        uncategorized_features = all_features - categorized_features
        
        print(f"  Categorized {len(categorized_features)} / {len(df_all_features.columns)} features")
        if uncategorized_features:
            print(f"  ⚠️  {len(uncategorized_features)} features uncategorized (will be skipped):")
            for feat in sorted(uncategorized_features):
                print(f"     - {feat}")
    
    selected_features = []
    block_assignments = {}
    whitelist_selected_count = 0
    
    for block_name, block_config in blocks.items():
        # Find available features for this block
        available = [f for f in block_config['features'] if f in df_all_features.columns]
        
        if not available:
            print(f"  {block_name}: No features available (skipping)")
            continue
        
        n_keep = block_config['keep']
        
        # STEP 1: Prioritize whitelisted features from this block
        whitelisted_in_block = [f for f in available if f in active_whitelist]
        non_whitelisted = [f for f in available if f not in active_whitelist]
        
        # Keep all whitelisted features (up to n_keep)
        keep_features = whitelisted_in_block[:n_keep]
        whitelist_selected_count += len(keep_features)
        
        # STEP 2: Fill remaining slots with high-variance features
        remaining_slots = n_keep - len(keep_features)
        if remaining_slots > 0 and non_whitelisted:
            # Select top variance features for remaining slots
            if len(non_whitelisted) <= remaining_slots:
                # Keep all non-whitelisted
                keep_features.extend(non_whitelisted)
            else:
                # Select top N by priority rule
                top_features = _select_top_features(
                    df_all_features[non_whitelisted], # type: ignore
                    n_keep=remaining_slots,
                    priority=block_config['priority']
                )
                keep_features.extend(top_features)
        
        selected_features.extend(keep_features)
        for feat in keep_features:
            block_assignments[feat] = block_name
        
        # Report with whitelist indicator
        whitelist_str = f" (🔒 {len(whitelisted_in_block)} whitelisted)" if whitelisted_in_block else ""
        print(f"  {block_name}: {len(available)} available → {len(keep_features)} selected{whitelist_str}")
    
    df_filtered = df_all_features[selected_features].copy()
    
    # Safety check: ensure no NaN values
    nan_count = df_filtered.isnull().sum().sum()
    if nan_count > 0:
        print(f"\n  ⚠️  Warning: {nan_count} NaN values in filtered data, cleaning...")
        df_filtered = df_filtered.dropna()
        print(f"  After cleaning: {len(df_filtered)} dates")
    
    stage_report = {
        'input_features': len(df_all_features.columns),
        'output_features': len(selected_features),
        'blocks_processed': len(blocks),
        'features_per_block': {k: len([f for f in selected_features if block_assignments.get(f) == k]) 
                               for k in blocks.keys()},
        'whitelisted_count': len(active_whitelist),
        'whitelisted_selected': whitelist_selected_count
    }
    
    print(f"\nStage 1 Complete:")
    print(f"  Input: {stage_report['input_features']} features")
    print(f"  Output: {stage_report['output_features']} features")
    print(f"  Reduction: {100 * (1 - stage_report['output_features'] / stage_report['input_features']):.1f}%")
    if active_whitelist:
        print(f"  🔒 Whitelisted: {whitelist_selected_count}/{len(active_whitelist)} selected")
    
    return df_filtered, block_assignments, stage_report # type: ignore


def prune_redundant_features(df_features: pd.DataFrame, 
                             block_assignments: Dict,
                             threshold: float = 0.85,
                             whitelist: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict, List, Dict]:
    """
    Stage 2: Remove highly correlated features to reduce redundancy.
    
    Args:
        df_features: DataFrame from Stage 1
        block_assignments: Dict from Stage 1 (used for tie-breaking)
        threshold: Correlation threshold (default 0.85)
        whitelist: Optional list of features to protect from removal
    
    Returns:
        df_pruned: DataFrame with redundant features removed
        pruned_block_assignments: Dict with updated block assignments
        removed_features: List of (feature, reason, correlation) tuples
        stage_report: Dict with pruning statistics
    """
    print("\n" + "="*60)
    print("STAGE 2: STATISTICAL REDUNDANCY PRUNING")
    print("="*60)
    
    # Initialize whitelist
    if whitelist is None:
        whitelist = []
    
    # Filter whitelist to only include features that exist in df_features
    active_whitelist = [f for f in whitelist if f in df_features.columns]
    
    if active_whitelist:
        print(f"  🔒 Protected features (whitelist): {len(active_whitelist)}")
        for feat in active_whitelist:
            block = block_assignments.get(feat, 'Unknown')
            print(f"     - {feat} [{block}]")
    
    removed = []
    features_to_keep = list(df_features.columns)
    
    # Compute correlation matrix
    print(f"  Computing correlation matrix...")
    corr_matrix = df_features.corr().abs()
    
    # Find high correlations
    print(f"  Identifying redundant pairs (threshold: {threshold})...")
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feat_i = corr_matrix.columns[i]
            feat_j = corr_matrix.columns[j]
            
            if feat_i not in features_to_keep or feat_j not in features_to_keep:
                continue
            
            corr_val = corr_matrix.iloc[i, j]
            
            if corr_val > threshold:
                # Check if either feature is whitelisted
                i_protected = feat_i in active_whitelist
                j_protected = feat_j in active_whitelist
                
                if i_protected and j_protected:
                    # Both whitelisted - keep both (critical features)
                    print(f"    ⚠️  High correlation {feat_i} <-> {feat_j} (r={corr_val:.3f}), but BOTH whitelisted - keeping both")
                    continue
                elif i_protected:
                    # feat_i is protected, remove feat_j
                    features_to_keep.remove(feat_j)
                    removed.append((feat_j, f"Corr({feat_i}, whitelisted)", corr_val))
                    print(f"    Removed {feat_j} (r={corr_val:.3f} with whitelisted {feat_i})")
                    continue
                elif j_protected:
                    # feat_j is protected, remove feat_i
                    features_to_keep.remove(feat_i)
                    removed.append((feat_i, f"Corr({feat_j}, whitelisted)", corr_val))
                    print(f"    Removed {feat_i} (r={corr_val:.3f} with whitelisted {feat_j})")
                    continue
                
                # Neither is whitelisted - use existing logic
                # Prefer keeping features from different blocks
                block_i = block_assignments.get(feat_i)
                block_j = block_assignments.get(feat_j)
                
                if block_i != block_j:
                    # Different blocks - keep both (cross-block correlation is informative)
                    continue
                
                # Same block - remove lower variance
                var_i = df_features[feat_i].var()
                var_j = df_features[feat_j].var()
                
                if var_i < var_j:
                    remove_feat = feat_i
                    keep_feat = feat_j
                else:
                    remove_feat = feat_j
                    keep_feat = feat_i
                
                features_to_keep.remove(remove_feat)
                removed.append((remove_feat, f"Corr({keep_feat})", corr_val))
                print(f"    Removed {remove_feat} (r={corr_val:.3f} with {keep_feat})")
    
    df_pruned = df_features[features_to_keep].copy()
    
    # Safety check: ensure no NaN values
    nan_count = df_pruned.isnull().sum().sum()
    if nan_count > 0:
        print(f"\n  ⚠️  Warning: {nan_count} NaN values in pruned data, cleaning...")
        df_pruned = df_pruned.dropna()
        print(f"  After cleaning: {len(df_pruned)} dates")
    
    # Update block assignments
    pruned_block_assignments = {k: v for k, v in block_assignments.items() if k in features_to_keep}
    
    # Count whitelisted features that survived
    survived_whitelist = [f for f in active_whitelist if f in features_to_keep]
    
    stage_report = {
        'input_features': len(df_features.columns),
        'output_features': len(features_to_keep),
        'removed_count': len(removed),
        'threshold': threshold,
        'whitelisted_count': len(active_whitelist),
        'whitelisted_survived': len(survived_whitelist)
    }
    
    print(f"\nStage 2 Complete:")
    print(f"  Input: {stage_report['input_features']} features")
    print(f"  Output: {stage_report['output_features']} features")
    print(f"  Removed: {stage_report['removed_count']} redundant features")
    if active_whitelist:
        print(f"  🔒 Whitelisted: {len(survived_whitelist)}/{len(active_whitelist)} survived")
    
    return df_pruned, pruned_block_assignments, removed, stage_report # type: ignore


def apply_block_pca(df_features: pd.DataFrame, 
                   block_assignments: Dict,
                   variance_threshold: float = 0.85,
                   min_features_for_pca: int = 3) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """
    Stage 3: Apply PCA within each economic block.
    
    Args:
        df_features: DataFrame from Stage 2
        block_assignments: Dict mapping features to blocks
        variance_threshold: Cumulative variance to retain (default 0.85)
        min_features_for_pca: Minimum features in block to apply PCA (default 3)
    
    Returns:
        df_pca: DataFrame with PCA components replacing multi-feature blocks
        pca_models: Dict of fitted PCA objects per block
        feature_loadings: Dict of PC loadings for interpretation
        stage_report: Dict with PCA statistics
    """
    print("\n" + "="*60)
    print("STAGE 3: BLOCK PCA")
    print("="*60)
    
    # Check for NaN values and clean if necessary
    nan_count = df_features.isnull().sum().sum()
    if nan_count > 0:
        print(f"  ⚠️  Warning: {nan_count} NaN values detected, cleaning...")
        # Drop any rows with NaN values
        before_rows = len(df_features)
        df_features = df_features.dropna()
        dropped = before_rows - len(df_features)
        print(f"  Dropped {dropped} rows with NaN values")
        
        if len(df_features) == 0:
            raise ValueError("All rows contain NaN values - cannot proceed with PCA")
    else:
        print(f"  ✅ No NaN values detected")
    
    # Group features by block
    blocks_dict = {}
    for feat, block in block_assignments.items():
        if feat in df_features.columns:
            blocks_dict.setdefault(block, []).append(feat)
    
    df_result = pd.DataFrame(index=df_features.index)
    pca_models = {}
    feature_loadings = {}
    
    for block_name, features in blocks_dict.items():
        if len(features) < min_features_for_pca:
            # Keep original features (too few for PCA)
            block_data = df_features[features].copy()
            
            # Check for NaN in this block
            block_nan = block_data.isnull().sum().sum()
            if block_nan > 0:
                print(f"  {block_name}: {len(features)} features - ⚠️  {block_nan} NaN values, dropping rows...")
                # Get clean indices for this block
                clean_idx = block_data.dropna().index
                df_result = df_result.loc[clean_idx]
                block_data = block_data.loc[clean_idx]
            
            df_result[features] = block_data
            print(f"  {block_name}: {len(features)} features (kept original - too few for PCA)")
        else:
            # Apply PCA
            block_data = df_features[features].copy()
            
            # Check for NaN in this block BEFORE PCA
            block_nan = block_data.isnull().sum().sum()
            if block_nan > 0:
                print(f"  {block_name}: {len(features)} features - ⚠️  {block_nan} NaN values detected!")
                print(f"    NaN counts per feature:")
                for feat in features:
                    nan_cnt = block_data[feat].isnull().sum() # type: ignore
                    if nan_cnt > 0:
                        print(f"      {feat}: {nan_cnt} NaN")
                
                # Drop rows with NaN for this block
                before_len = len(block_data)
                block_data = block_data.dropna()
                dropped = before_len - len(block_data)
                print(f"    Dropped {dropped} rows with NaN")
                
                # Update df_result to match these clean indices
                if len(df_result) > 0:
                    df_result = df_result.loc[block_data.index]
            
            # Now apply PCA on clean data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(block_data)
            
            pca = PCA(n_components=variance_threshold)
            X_pca = pca.fit_transform(X_scaled)
            
            # Add PCA components to result (using clean data index)
            n_components = pca.n_components_
            pca_df = pd.DataFrame(X_pca, index=block_data.index)
            for i in range(n_components):
                component_name = f"{block_name}_PC{i+1}"
                df_result[component_name] = pca_df[i]
            
            # Store model and loadings
            pca_models[block_name] = {'pca': pca, 'scaler': scaler, 'features': features}
            feature_loadings[block_name] = {
                'original_features': features,
                'loadings': pd.DataFrame(
                    pca.components_,
                    columns=features,
                    index=[f"PC{i+1}" for i in range(n_components)] # type: ignore
                ),
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
            }
            
            total_var = np.sum(pca.explained_variance_ratio_)
            print(f"  {block_name}: {len(features)} features → {n_components} PCs ({total_var:.1%} variance)")
    
    stage_report = {
        'input_features': len(df_features.columns),
        'output_features': len(df_result.columns),
        'blocks_with_pca': len(pca_models),
        'total_variance_retained': variance_threshold
    }
    
    print(f"\nStage 3 Complete:")
    print(f"  Input: {stage_report['input_features']} features")
    print(f"  Output: {stage_report['output_features']} features")
    print(f"  Blocks with PCA: {stage_report['blocks_with_pca']}")
    
    return df_result, pca_models, feature_loadings, stage_report


def select_regime_features(conn, 
                          version: str = 'V2_FWD_FILL', 
                          start_date: str = '2010-01-01',
                          redundancy_threshold: float = 0.85,
                          pca_variance: float = 0.85,
                          max_nan_pct: float = 0.10,
                          features_per_block: Optional[Dict[str, int]] = None,
                          min_pca_features: int = 3,
                          whitelist_features: Optional[List[str]] = None,
                          verbose: bool = True) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Complete 3-stage feature selection pipeline.
    
    Reduces 376 features to 20-40 optimal features using:
    1. Economic prefiltering (domain knowledge)
    2. Statistical redundancy pruning (correlation-based)
    3. PCA within blocks (dimensionality reduction)
    
    Args:
        conn: Database connection
        version: Feature version to load (default: 'V2_FWD_FILL')
        start_date: Start date for feature data (default: '2010-01-01')
        redundancy_threshold: Correlation threshold for pruning (default: 0.85)
        pca_variance: Cumulative variance to retain in PCA (default: 0.85)
        max_nan_pct: Maximum % of NaN values allowed per feature (default: 0.10)
        features_per_block: Optional dict to override number of features per block
        min_pca_features: Minimum features in a block to apply PCA (default: 3)
        whitelist_features: Optional list of critical features to protect from removal
                          (default: None, no whitelist protection)
        verbose: Print detailed progress information (default: True)
    
    Returns:
        df_selected: Final feature DataFrame (dates × 20-40 features)
        selection_report: Dict with detailed statistics from all stages
        metadata: Dict with block assignments, PCA models, loadings
    """
    if verbose:
        print("\n" + "="*60)
        print("FEATURE SELECTION PIPELINE")
        print("="*60)
        print(f"Version: {version}")
        print(f"Start date: {start_date}")
        print(f"Redundancy threshold: {redundancy_threshold}")
        print(f"PCA variance threshold: {pca_variance}")
        print(f"Max NaN %: {max_nan_pct:.1%}")
        print(f"Min features for PCA: {min_pca_features}")
    
    # Load ALL features from database (with NaN cleaning)
    df_all = _load_all_features(conn, version, start_date, max_nan_pct=max_nan_pct)
    
    # Use provided whitelist (or empty list if None)
    active_whitelist = whitelist_features if whitelist_features is not None else []
    
    # Stage 1: Economic prefilter (WITH WHITELIST PRIORITY)
    # Override feature counts per block if provided
    blocks = None
    if features_per_block is not None:
        blocks = _build_economic_blocks(list(df_all.columns))
        for block_name, n_features in features_per_block.items():
            if block_name in blocks:
                blocks[block_name]['keep'] = n_features
    
    df_stage1, block_assignments, report1 = economic_prefilter(
        df_all, blocks=blocks, whitelist=active_whitelist
    )
    
    # Stage 2: Statistical redundancy pruning (with whitelist protection)
    
    df_stage2, block_assignments_pruned, removed, report2 = prune_redundant_features(
        df_stage1, block_assignments, threshold=redundancy_threshold, whitelist=active_whitelist
    )
    
    # Stage 3: Block PCA
    df_final, pca_models, loadings, report3 = apply_block_pca(
        df_stage2, block_assignments_pruned, variance_threshold=pca_variance,
        min_features_for_pca=min_pca_features
    )
    
    # Combine reports
    selection_report = {
        'stage1_economic_prefilter': report1,
        'stage2_redundancy_pruning': report2,
        'stage3_block_pca': report3,
        'pipeline_summary': {
            'input_features': len(df_all.columns),
            'final_features': len(df_final.columns),
            'reduction_pct': 100 * (1 - len(df_final.columns) / len(df_all.columns)),
            'date_range': (str(df_final.index.min()), str(df_final.index.max())),
            'n_dates': len(df_final)
        }
    }
    
    metadata = {
        'block_assignments': block_assignments_pruned,
        'pca_models': pca_models,
        'feature_loadings': loadings,
        'removed_features': removed
    }
    
    if verbose:
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"✅ Input: {selection_report['pipeline_summary']['input_features']} features")
        print(f"✅ Output: {selection_report['pipeline_summary']['final_features']} features")
        print(f"✅ Reduction: {selection_report['pipeline_summary']['reduction_pct']:.1f}%")
        print(f"✅ Date range: {selection_report['pipeline_summary']['date_range'][0]} to {selection_report['pipeline_summary']['date_range'][1]}")
        print(f"✅ Observations: {selection_report['pipeline_summary']['n_dates']:,}")
        print("="*60)
        
        # Print final selected features
        print("\nFINAL SELECTED FEATURES:")
        print("="*60)
        
        # Group by type (original vs PCA)
        original_features = [f for f in df_final.columns if '_PC' not in f]
        pca_features = [f for f in df_final.columns if '_PC' in f]
        
        if original_features:
            print(f"\nOriginal Features ({len(original_features)}):")
            for i, feat in enumerate(sorted(original_features), 1):
                block = metadata['block_assignments'].get(feat, 'Unknown')
                print(f"  {i:2d}. {feat:40s} [{block}]")
        
        if pca_features:
            print(f"\nPCA Components ({len(pca_features)}):")
            for i, feat in enumerate(sorted(pca_features), 1):
                block_name = feat.split('_PC')[0]
                if block_name in metadata['feature_loadings']:
                    loading_data = metadata['feature_loadings'][block_name]
                    pc_num = int(feat.split('_PC')[1]) - 1
                    var_explained = loading_data['explained_variance'][pc_num]
                    print(f"  {i:2d}. {feat:40s} ({var_explained:.1%} var)")
                else:
                    print(f"  {i:2d}. {feat}")
        
        print("="*60 + "\n")
    
    return df_final, selection_report, metadata

