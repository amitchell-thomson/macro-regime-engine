# Macro Regime Engine

A Python-based system for detecting and analyzing global macro regimes using cross-asset time series data.

## Overview

This project identifies **market-wide macro states** (e.g., RISK_ON, RISK_OFF, INFLATION_SHOCK, DISINFLATION) by analyzing coordinated movements across multiple asset classes. It ingests data, computes interpretable features, and infers regimes—all stored in PostgreSQL for inspection, comparison, and versioning.


---

## Key Features

### 1. **Cross-Asset Data Ingestion**
- 59 tickers across 7 asset classes (Equity, Rates, FX, Credit, Commodities, Volatility, Macro)
- Free data sources: yfinance + FRED API
- Covers risk appetite, growth, inflation, monetary policy, liquidity, and global flows
- Daily or lower frequency data suitable for macro regime detection

See [`configs/README.md`](configs/README.md) for detailed universe rationale.

### 2. **Feature Engineering**
- Compute cross-asset features in Python (returns, spreads, momentum, ratios, etc.)
- Features are fully regeneratable and versioned
- Support for both ticker-specific and global aggregate features
- Examples: yield curve slopes, credit spread changes, style factor returns

### 3. **Regime Detection**
- Infer market-wide regimes from cross-asset feature patterns
- Output regime labels with confidence scores
- Support for multiple regime models (rules-based, HMM, clustering, etc.)
- Version control for different regime methodologies

### 4. **Database-Backed Persistence**
- PostgreSQL schema with three core tables:
  - `raw_series`: Immutable source-of-truth time series data
  - `features`: Derived metrics (regeneratable, versioned)
  - `regimes_global`: Market-wide regime labels (versioned, never overwritten)
- All data timestamped and versioned for reproducible research
- Inspect outputs via SQLTools in VS Code

### 5. **Version Control & Comparison**
- Multiple versions of features and regimes can coexist
- Compare regime model performance across versions
- Never overwrite old results—always append with new version tags
- Track model evolution: `V0_RULES`, `V1_BASELINE`, `V2_HMM`

### 6. **Research-Oriented Design**
- Simple, explicit Python (no heavy frameworks)
- Transparent database interactions
- Easy to iterate on features and regime logic
- Built for exploration, not production latency

---

## Architecture

### Design Philosophy

**Python does all logic:**
- Data ingestion and cleaning
- Feature engineering
- Regime inference

**Postgres is passive storage:**
- Stores raw observations
- Stores derived features
- Stores regime outputs

**SQL contains no business logic:**
- Only schema, indexing, and inspection queries

### Data Flow

```
1. Ingest → raw_series (truth table, immutable)
2. Compute → features (derived, regeneratable, versioned)
3. Infer → regimes_global (market-wide labels, versioned)
4. Inspect → SQLTools / Notebooks
5. Iterate → New versions, compare results
```

---

## Use Cases

Once you have regime probabilities, you can:

1. **Dynamic Asset Allocation**: Tilt portfolio weights based on regime (overweight equities in RISK_ON, shift to bonds in RISK_OFF)
2. **Risk Management**: Scale exposure and adjust hedges based on regime uncertainty
3. **Factor Timing**: Apply momentum in RISK_ON, quality in RISK_OFF
4. **Signal Filtering**: Only trade signals aligned with current regime
5. **Scenario Analysis**: Compute regime-conditional returns and volatility
6. **Research**: Backtest strategies under different regime conditions

---

## Project Structure

```
macro-regime-engine/
├── configs/
│   ├── universe.yml          # 59 tickers across 7 asset classes
│   └── README.md             # Detailed universe documentation
├── src/
│   └── db/
│       └── schema.sql        # PostgreSQL schema
├── notebooks/
│   └── sanity_check.ipynb    # Exploration and validation
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Database Schema

All tables live in the `macro` schema:

### `raw_series`
- One row per (ticker, date)
- Append/upsert only
- Columns: ticker, asset_class, dt, value, source, ingested_at

### `features`
- Derived features in long form
- Fully regeneratable
- Versioned
- Global features use ticker='GLOBAL', asset_class='MACRO'
- Columns: dt, ticker, asset_class, feature, value, version, computed_at

### `regimes_global`
- Market-wide regime labels
- One row per (date, version)
- Never overwritten
- Columns: dt, regime, score, version, created_at

---

## Data Sources

All data comes from **free sources**:

- **yfinance**: Equities, FX, commodities, volatility, crypto (33 tickers)
- **FRED API**: Rates, credit spreads, macro indicators, financial conditions (26 tickers)

Both provide daily or lower frequency data with excellent Python libraries.

---

## Example Regimes

Macro regimes are patterns of cross-asset moves:

**RISK_ON:**
- Equities ↑, Credit spreads ↓, VIX ↓
- Growth > Value, Cyclicals outperform
- Dollar ↓, EM outperforms

**RISK_OFF:**
- Equities ↓, Credit spreads ↑, VIX ↑
- Defensives (Utilities) outperform
- Gold ↑, Dollar ↑

**INFLATION_SHOCK:**
- Commodities ↑↑, Breakeven inflation ↑
- Real rates ↓ (nominal lags inflation)
- Value > Growth

**DISINFLATION:**
- Bonds rally (yields ↓)
- Inflation expectations ↓
- Equities + Bonds both rally ("Goldilocks")

---

## Non-Goals

This project intentionally **does not**:
- Build execution systems
- Optimize for latency
- Encode strategy logic in SQL
- Prematurely productionize models
- Tightly couple models to schema

Those concerns are deferred.

---

## Documentation

- **Universe details**: See [`configs/README.md`](configs/README.md)
- **Project context**: See [`.cursor/context.md`](.cursor/context.md)
- **Database schema**: See [`src/db/schema.sql`](src/db/schema.sql)

---

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Set up PostgreSQL database (see `src/db/schema.sql`)
3. Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
4. Run data ingestion scripts (to be implemented)
5. Compute features in Python
6. Infer regimes
7. Inspect results using SQLTools or notebooks

---

## Philosophy

> "A macro research notebook, except the notebook is persistent, versioned, inspectable, and lives across Python + Postgres."

Prefer:
- Clarity over cleverness
- Reproducibility over performance
- Explicit versioning over overwriting