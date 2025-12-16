# Regimes Module

## Overview

This module detects macro market regimes using cross-asset features and stores daily regime probabilities and metadata in the database for portfolio optimization.

**V1 Implementation**: Hidden Markov Model (HMM) with 4 regimes
- **RISK_ON**: Low volatility, positive equity returns, tight credit spreads
- **RISK_OFF**: High volatility, negative equity returns, wide credit spreads  
- **INFLATION_SHOCK**: High inflation expectations, commodity strength, negative real rates
- **DISINFLATION**: Falling inflation, bond rally, flattening commodities

---

## Target Outputs

The regime detection model produces **two key outputs** that are stored in the database and designed specifically for portfolio optimization:

### 1. Daily Regime Probabilities (PRIMARY OUTPUT)

**Table**: `MACRO.REGIMES_GLOBAL`

**Format**: 4 rows per date (one per regime)

| dt | regime | score | version |
|----|--------|-------|---------|
| 2024-01-15 | RISK_ON | 0.65 | V1_HMM |
| 2024-01-15 | RISK_OFF | 0.20 | V1_HMM |
| 2024-01-15 | INFLATION_SHOCK | 0.10 | V1_HMM |
| 2024-01-15 | DISINFLATION | 0.05 | V1_HMM |

**Requirements**:
- ✅ Probabilities sum to 1.0 for each date
- ✅ All 4 regimes present every date
- ✅ No missing dates in date range
- ✅ Smoothly varying (no erratic jumps)

**Portfolio Use Case**: 
Blend regime-conditional portfolios weighted by probabilities. Example:
```python
portfolio = (
    P(RISK_ON) * portfolio_risk_on +
    P(RISK_OFF) * portfolio_risk_off +
    P(INFLATION_SHOCK) * portfolio_inflation +
    P(DISINFLATION) * portfolio_disinflation
)
```

### 2. Regime Metadata (SECONDARY OUTPUTS)

**Table**: `MACRO.REGIME_METADATA`

**Format**: 1 row per date with summary metrics

| Field | Description | Portfolio Use |
|-------|-------------|---------------|
| `dominant_regime` | Regime with max probability | Simple decision rules |
| `max_probability` | Confidence in dominant regime (0.25-1.0) | Position sizing |
| `entropy` | Uncertainty measure (0-1.39) | Risk scaling |
| `regime_changed` | Boolean: regime switch today? | Rebalancing trigger |
| `prev_regime` | Yesterday's dominant regime | Transition detection |

**Portfolio Use Cases**:
- **Entropy > 0.8**: Scale down position sizes (high uncertainty)
- **regime_changed = TRUE**: Trigger portfolio rebalancing
- **max_probability < 0.5**: Wait-and-see, avoid regime bets
- **max_probability > 0.7**: High confidence, take directional positions

---

## Module Architecture

```
Feature Loading → HMM Training → State Decoding → Regime Labeling → Database Storage → Validation
     (features.py)   (hmm.py)       (hmm.py)      (hmm.py)         (database.py)    (diagnostics.py)
```

### Data Flow

1. **Input**: N regime-relevant features from `MACRO.FEATURES` table
2. **Processing**: Train HMM to identify 4 hidden states, decode probabilities
3. **Interpretation**: Map anonymous states to interpretable regime names
4. **Output**: Store regime probabilities and metadata in database
5. **Validation**: Verify outputs match expected behavior

---

## File Structure

```
src/regimes/
├── __init__.py           # Module initialization
├── features.py           # Feature selection and preparation ✅ DONE
├── hmm.py                # HMM training, decoding, regime labeling (TODO)
├── database.py           # Database storage functions (TODO)
├── diagnostics.py        # Validation and visualization (TODO)
└── README.md             # This file
```

---

## Implementation Plan

### ✅ Step 1: Feature Selection and Preparation (COMPLETED)

**File**: `features.py`

**What it does**:
- Selects N regime-relevant features covering four dimensions:
  - **Risk appetite**: VIX, credit spreads, equity returns, style factors
  - **Growth**: Yield curve slope, leading indicators
  - **Inflation**: Breakeven inflation, commodities, real rates
  - **Cross-asset**: AUD/JPY, gold/oil, EM vs US
- Loads features from database in long format
- Pivots to wide format: dates (index) × features (columns)
- Validates data quality (no NaNs, date continuity, sufficient data)

**Key Functions**:
- `get_regime_features(conn, version, start_date)` → DataFrame (dates × features)
- `validate_feature_data(df_wide)` → validation report dict
- `print_validation_report(report)` → formatted output

**Feature List** (14 features):
```python
REGIME_FEATURES = [
    'VOL_VIX_LEVEL',                    # Risk appetite
    'CREDIT_HY_CHG_20D',
    '^GSPC_RET_20D',
    'STYLE_CYCLICAL_VS_DEFENSIVE',
    'STYLE_GROWTH_VS_VALUE',
    'GLOBAL_YIELD_CURVE_SLOPE',         # Growth
    'USSLIND_CHG_MOM',
    'GLOBAL_INFLATION_EXPECTATIONS',    # Inflation
    'CL=F_RET_20D',
    'REAL_RATE_10Y',
    'FX_AUD_JPY',                       # Cross-asset
    'COMMODITY_GOLD_OIL',
    'EQUITY_EM_VS_US',
    'GLOBAL_FINANCIAL_CONDITIONS',      # Financial conditions
]
```

---

### 🔄 Step 2: HMM Training and State Decoding (IN PROGRESS)

**File**: `hmm.py`

**What it will do**:
- Train GaussianHMM on normalized features (4 hidden states)
- Decode state probabilities using forward-backward algorithm
- Analyze state characteristics (mean feature values per state)
- Map anonymous states to interpretable regime names
- Compute regime probabilities and metadata

**Functions to implement**:

```python
def train_hmm(df_features, n_regimes=4, random_state=42):
    """
    Train GaussianHMM, decode state probabilities.
    
    Steps:
    1. Normalize features with StandardScaler
    2. Initialize GaussianHMM (n_components=4, covariance_type='full')
    3. Fit model to normalized features
    4. Check convergence
    5. Use forward-backward algorithm for smooth probabilities
    
    Returns:
        model: Trained HMM
        state_probs: Array (T x 4) of state probabilities
        scaler: StandardScaler object
    """

def label_regimes(model, feature_names):
    """
    Analyze mean feature values per state, assign regime names.
    
    Labeling heuristics (analyze model.means_):
    - RISK_ON: Low VIX + positive equity + tight credit spreads
    - RISK_OFF: High VIX + negative equity + wide credit spreads
    - INFLATION_SHOCK: High inflation + positive commodities
    - DISINFLATION: Low inflation + falling yields
    
    Returns:
        state_to_regime: Dict {0: 'RISK_ON', 1: 'RISK_OFF', ...}
    """

def compute_regime_probabilities(state_probs, state_to_regime, dates):
    """
    Map state probability matrix to regime names.
    
    Returns:
        regime_probs_df: DataFrame with regime names as columns, dates as index
                        Columns: ['RISK_ON', 'RISK_OFF', 'INFLATION_SHOCK', 'DISINFLATION']
    """

def compute_regime_metadata(regime_probs_df):
    """
    Calculate metadata for each date.
    
    Calculations:
    - dominant_regime = argmax(probabilities)
    - max_probability = max(probabilities)
    - entropy = -sum(p * log(p))
    - regime_changed = (dominant_regime != previous dominant_regime)
    - prev_regime = lag(dominant_regime, 1)
    
    Returns:
        metadata_df: DataFrame with metadata columns
    """
```

**Implementation Notes**:
- Use `hmmlearn.hmm.GaussianHMM` library
- Use **forward-backward** algorithm (not just Viterbi) for smooth probabilities
- Full covariance matrix to capture feature interactions
- Verify each probability row sums to 1.0
- Entropy formula: H = -Σ(p_i * log(p_i)), max value ≈ 1.39 for 4 states

---

### 📊 Step 3: Database Storage (TODO)

**File**: `database.py`

**What it will do**:
- Store regime probabilities in `REGIMES_GLOBAL` table (4 rows per date)
- Store regime metadata in `REGIME_METADATA` table (1 row per date)
- Use bulk insert with `execute_values` for performance
- Handle transactions properly (commit on success, rollback on error)

**Functions to implement**:

```python
def store_regime_outputs(conn, regime_probs_df, metadata_df, version='V1_HMM'):
    """
    Store both outputs in single transaction.
    
    Stores:
        - regime_probs_df → REGIMES_GLOBAL (4 rows per date)
        - metadata_df → REGIME_METADATA (1 row per date)
    
    Returns:
        dict with row counts and date range
    """
```

**Database Pattern**:
Follow bulk insert pattern from `src/features/database.py` lines 80-120:
- Prepare data as list of tuples
- Use `psycopg2.extras.execute_values` for bulk insert
- Wrap both inserts in single transaction
- Delete existing version data before inserting (allow reruns)

---

### 🔍 Step 4: Validation and Diagnostics (TODO)

**File**: `diagnostics.py`

**What it will do**:
- Load regime data from database
- Validate output quality
- Create visualizations for manual inspection
- Compute summary statistics

**Functions to implement**:

```python
def validate_regime_data(conn, version):
    """
    Run validation checks on stored regime data.
    
    Checks:
    - Probabilities sum to 1.0 for each date
    - No missing dates in range
    - Entropy values in valid range (0-1.39)
    - Regime changes are reasonable (not too frequent)
    - No obvious data quality issues
    
    Returns:
        validation_report: Dict with pass/fail for each check
    """

def plot_regime_outputs(conn, version, start_date=None):
    """
    Visualize outputs for manual validation.
    
    Creates 3 plots:
    1. Regime probability evolution (stacked area chart)
    2. Entropy over time (line plot)
    3. Regime transitions timeline (color-coded bars)
    """

def summarize_regime_statistics(conn, version):
    """
    Compute summary statistics for outputs.
    
    Returns DataFrame with:
    - % time in each regime
    - Average entropy
    - Number of regime transitions
    - Average regime duration
    """
```

**Validation Criteria**:
- ✅ 2020 COVID crash shows clear RISK_OFF period
- ✅ 2022 shows INFLATION_SHOCK regime
- ✅ Each regime has distinct feature characteristics
- ✅ Regime distribution is reasonable (no single regime >60% of time)
- ✅ Smooth probability evolution (no jumps >0.5 in one day)

---

## Usage Example

```python
from regimes.features import get_regime_features
from regimes.hmm import train_hmm, label_regimes, compute_regime_probabilities, compute_regime_metadata
from regimes.database import store_regime_outputs
from regimes.diagnostics import validate_regime_data, plot_regime_outputs
from ingestion.database import get_connection

# 1. Load features
conn = get_connection(password=db_password)
df_features = get_regime_features(conn, version='V2_FWD_FILL', start_date='2010-01-01')

# 2. Train HMM and decode
model, state_probs, scaler = train_hmm(df_features, n_regimes=4)

# 3. Label regimes
state_to_regime = label_regimes(model, df_features.columns)

# 4. Compute outputs
regime_probs_df = compute_regime_probabilities(state_probs, state_to_regime, df_features.index)
metadata_df = compute_regime_metadata(regime_probs_df)

# 5. Store in database
store_regime_outputs(conn, regime_probs_df, metadata_df, version='V1_HMM')

# 6. Validate
validation_report = validate_regime_data(conn, version='V1_HMM')
plot_regime_outputs(conn, version='V1_HMM')
```

---

## Portfolio Optimization Queries

Once regime outputs are stored in the database, portfolio optimization systems can query them:

### Get Today's Regime Probabilities

```sql
SELECT regime, score
FROM macro.regimes_global
WHERE dt = CURRENT_DATE AND version = 'V1_HMM'
ORDER BY score DESC;
```

**Output**:
```
regime              | score
--------------------|------
RISK_ON             | 0.65
RISK_OFF            | 0.20
INFLATION_SHOCK     | 0.10
DISINFLATION        | 0.05
```

### Get Current Regime Metadata

```sql
SELECT dominant_regime, max_probability, entropy, regime_changed
FROM macro.regime_metadata
WHERE dt = CURRENT_DATE AND version = 'V1_HMM';
```

**Output**:
```
dominant_regime | max_probability | entropy | regime_changed
----------------|-----------------|---------|---------------
RISK_ON         | 0.65            | 0.82    | false
```

### Get Historical Regime Time Series (for backtesting)

```sql
SELECT dt, regime, score
FROM macro.regimes_global
WHERE version = 'V1_HMM' AND dt >= '2020-01-01'
ORDER BY dt, regime;
```

### Find Recent Regime Transitions (for alerts)

```sql
SELECT dt, dominant_regime, prev_regime, max_probability
FROM macro.regime_metadata
WHERE version = 'V1_HMM' 
  AND regime_changed = TRUE
  AND dt >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY dt DESC;
```

---

## Success Criteria

V1 is successful when:

### Database Tables Populated
- ✅ `REGIMES_GLOBAL`: ~4 rows × 3,900 dates = ~15,600 rows
- ✅ `REGIME_METADATA`: ~3,900 rows (one per business day 2010-2024)

### Output Quality Checks Pass
- ✅ Probabilities sum to 1.0 (all dates)
- ✅ Entropy values in valid range (0-1.39)
- ✅ No missing dates
- ✅ Smooth probability evolution (no erratic jumps)
- ✅ Regime changes align with major market events

### Regime Interpretability
- ✅ RISK_OFF clearly visible in March 2020 (COVID crash)
- ✅ INFLATION_SHOCK visible in 2021-2022
- ✅ Each regime has distinct feature characteristics
- ✅ Regime distribution is reasonable (no single regime >60% of time)

### Queryable for Portfolio Use
- ✅ All example queries execute successfully
- ✅ Results are in correct format for downstream consumption
- ✅ Can easily join with feature data or returns data by date

---

## Dependencies

Add to `requirements.txt`:
```
hmmlearn>=0.3.0        # Hidden Markov Model implementation
scikit-learn>=1.3.0    # StandardScaler and preprocessing
matplotlib>=3.7.0      # Plotting
seaborn>=0.12.0        # Advanced visualization
```

---

## Future Enhancements (Post-V1)

**V2**: Alternative modeling approaches
- Rules-based regime detection for comparison
- K-means clustering approach
- Gaussian Mixture Models (GMM)

**V3**: Advanced features
- Online updating (incremental regime detection)
- Regime-conditional return forecasting
- Transition probability matrices

**V4**: Portfolio integration
- Regime-conditional covariance matrices
- Dynamic risk budgeting based on entropy
- Automated rebalancing triggers

---

## References

- **Database Schema**: `src/db/schema.sql` (lines 53-94)
- **Feature Engineering**: `src/features/README.md`
- **Example Notebook**: `notebooks/regime_detection.ipynb`
- **HMM Library**: [hmmlearn documentation](https://hmmlearn.readthedocs.io/)

---

## Current Status

**Completed** ✅:
- Module structure (`__init__.py`)
- Feature selection and preparation (`features.py`)
- Test notebook with visualizations (`notebooks/regime_detection.ipynb`)

**In Progress** 🔄:
- HMM training and decoding (`hmm.py`)

**TODO** 📋:
- State-to-regime labeling (`hmm.py`)
- Regime probability computation (`hmm.py`)
- Metadata calculation (`hmm.py`)
- Database storage (`database.py`)
- Validation and diagnostics (`diagnostics.py`)
- End-to-end testing

