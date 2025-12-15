# Feature Engineering Documentation

## Overview

This module computes interpretable cross-asset features from raw time series data. Features are designed to capture regime-defining patterns across asset classes and are stored in the `FEATURES` table with versioning support.

## Feature Categories

Features are organized into four main categories:

1. **Basic Features**: Returns, changes, momentum, volatility, levels
2. **Spread Features**: Yield curve spreads, credit spreads, real rates
3. **Ratio Features**: Style factors, cross-asset ratios, volatility ratios
4. **Global Aggregates**: Market-wide composite indicators

## Feature Applicability

Not all features are computed for all tickers. Feature computation is asset-class specific:

- **Returns/Momentum/Volatility**: Only for price-based assets (Equity, FX, Commodities, Volatility)
- **Changes**: Only for rate-based assets (Rates, some Macro indicators)
- **Spreads**: Only for specific ticker combinations (e.g., yield curve requires rate tickers)
- **Ratios**: Only when both components exist
- **Global Aggregates**: Computed from multiple tickers across asset classes

---

## Basic Features

### Returns (Price-Based Assets Only)

Returns are computed for Equity, FX, Commodities, and Volatility indices.

#### `{TICKER}_RET_1D`
- **Formula**: `(Price[t] / Price[t-1]) - 1`
- **Calculation**: Daily percentage return
- **Interpretation**: One-day price change
- **Regime Signal**: Positive in RISK_ON, negative in RISK_OFF
- **Typical Range**: -10% to +10% for daily returns
- **Applicable To**: EQUITY, FX, COMMODITIES, VOL

#### `{TICKER}_RET_5D`
- **Formula**: `(Price[t] / Price[t-5]) - 1`
- **Calculation**: 5-day (weekly) percentage return
- **Interpretation**: Short-term momentum
- **Regime Signal**: Captures weekly regime shifts
- **Typical Range**: -15% to +15%

#### `{TICKER}_RET_20D`
- **Formula**: `(Price[t] / Price[t-20]) - 1`
- **Calculation**: 20-day (monthly) percentage return
- **Interpretation**: Medium-term momentum, captures monthly trends
- **Regime Signal**: Strong positive in RISK_ON, negative in RISK_OFF
- **Typical Range**: -20% to +20%

#### `{TICKER}_RET_60D`
- **Formula**: `(Price[t] / Price[t-60]) - 1`
- **Calculation**: 60-day (quarterly) percentage return
- **Interpretation**: Long-term trend
- **Regime Signal**: Confirms sustained regime shifts
- **Typical Range**: -30% to +30%

### Changes (Rate-Based Assets Only)

Changes are computed for Rates, Credit spreads, and some Macro indicators.

#### `{TICKER}_CHG_1D`
- **Formula**: `Value[t] - Value[t-1]`
- **Calculation**: Daily change in rate/spread
- **Interpretation**: One-day change in yield or spread
- **Regime Signal**: Rising yields in INFLATION_SHOCK, falling in DISINFLATION
- **Typical Range**: -50 bps to +50 bps for yields
- **Applicable To**: RATES, CREDIT, MACRO (rate-based)

#### `{TICKER}_CHG_20D`
- **Formula**: `Value[t] - Value[t-20]`
- **Calculation**: 20-day change in rate/spread
- **Interpretation**: Medium-term rate/spread movement
- **Regime Signal**: Captures monetary policy regime shifts
- **Typical Range**: -200 bps to +200 bps

#### `{TICKER}_CHG_60D`
- **Formula**: `Value[t] - Value[t-60]`
- **Calculation**: 60-day change in rate/spread
- **Interpretation**: Long-term rate/spread trend
- **Regime Signal**: Confirms sustained monetary regime changes

### Momentum (Price-Based Assets Only)

#### `{TICKER}_MOM_20D`
- **Formula**: `Price[t] - Price[t-20]`
- **Calculation**: 20-day price change (absolute, not percentage)
- **Interpretation**: Absolute momentum over 1 month
- **Regime Signal**: Positive in RISK_ON, negative in RISK_OFF
- **Typical Range**: Varies by asset price level
- **Applicable To**: EQUITY, FX, COMMODITIES, VOL

#### `{TICKER}_MOM_60D`
- **Formula**: `Price[t] - Price[t-60]`
- **Calculation**: 60-day price change
- **Interpretation**: Long-term absolute momentum
- **Regime Signal**: Confirms sustained trends

### Volatility (Price-Based Assets Only)

#### `{TICKER}_VOL_20D`
- **Formula**: `std(daily_returns, window=20) * sqrt(252)`
- **Calculation**: 20-day rolling annualized volatility
- **Interpretation**: Short-term volatility measure
- **Regime Signal**: High in RISK_OFF, low in RISK_ON
- **Typical Range**: 5% to 50% annualized
- **Applicable To**: EQUITY, FX, COMMODITIES, VOL

#### `{TICKER}_VOL_60D`
- **Formula**: `std(daily_returns, window=60) * sqrt(252)`
- **Calculation**: 60-day rolling annualized volatility
- **Interpretation**: Medium-term volatility
- **Regime Signal**: Captures volatility regime shifts

### Levels

#### `{TICKER}_LEVEL`
- **Formula**: Raw value from database
- **Calculation**: Unprocessed ticker value
- **Interpretation**: Baseline reference for spreads/ratios
- **Use Case**: Used as input for spread and ratio computations
- **Applicable To**: All tickers

---

## Spread Features

### Yield Curve Spreads

#### `YCURVE_2S10S`
- **Formula**: `DGS10 - DGS2`
- **Calculation**: 10-year yield minus 2-year yield
- **Interpretation**: Classic yield curve slope, recession predictor
- **Regime Signal**: 
  - Positive (steep): RISK_ON, growth expectations
  - Negative (inverted): RISK_OFF, recession warning
- **Typical Range**: -100 bps to +300 bps
- **Components**: DGS2, DGS10

#### `YCURVE_5S30S`
- **Formula**: `DGS30 - DGS5`
- **Calculation**: 30-year yield minus 5-year yield
- **Interpretation**: Long-end of yield curve slope
- **Regime Signal**: Captures long-term growth/inflation expectations
- **Typical Range**: -50 bps to +200 bps
- **Components**: DGS5, DGS30

#### `YCURVE_2S5S`
- **Formula**: `DGS5 - DGS2`
- **Calculation**: 5-year yield minus 2-year yield
- **Interpretation**: Short-end yield curve slope
- **Regime Signal**: Captures near-term monetary policy expectations
- **Typical Range**: -50 bps to +150 bps
- **Components**: DGS2, DGS5

### Credit Spreads

#### `CREDIT_HY_SPREAD`
- **Formula**: Raw value from BAMLH0A0HYM2
- **Calculation**: High yield credit spread (OAS)
- **Interpretation**: Risk premium for high-yield bonds
- **Regime Signal**: 
  - Low (< 400 bps): RISK_ON
  - High (> 800 bps): RISK_OFF, credit stress
- **Typical Range**: 200 bps to 2000 bps
- **Component**: BAMLH0A0HYM2

#### `CREDIT_HY_CHG_20D`
- **Formula**: `CREDIT_HY_SPREAD[t] - CREDIT_HY_SPREAD[t-20]`
- **Calculation**: 20-day change in HY spread
- **Interpretation**: Widening = risk-off, tightening = risk-on
- **Regime Signal**: Rapid widening signals regime shift to RISK_OFF
- **Typical Range**: -200 bps to +200 bps

#### `CREDIT_IG_SPREAD`
- **Formula**: Raw value from BAMLC0A0CM
- **Calculation**: Investment grade credit spread (OAS)
- **Interpretation**: Risk premium for investment-grade bonds
- **Regime Signal**: More stable than HY, but still widens in RISK_OFF
- **Typical Range**: 50 bps to 500 bps
- **Component**: BAMLC0A0CM

#### `CREDIT_IG_CHG_20D`
- **Formula**: `CREDIT_IG_SPREAD[t] - CREDIT_IG_SPREAD[t-20]`
- **Calculation**: 20-day change in IG spread
- **Interpretation**: Change in investment-grade credit conditions
- **Regime Signal**: Widening signals credit stress

### Real Rates

#### `REAL_RATE_10Y`
- **Formula**: Raw value from DFII10
- **Calculation**: 10-year TIPS yield (real rate)
- **Interpretation**: Real interest rate after inflation
- **Regime Signal**: 
  - Negative: "TINA" (There Is No Alternative to stocks)
  - Positive: Bonds competitive with stocks
- **Typical Range**: -2% to +3%
- **Component**: DFII10

---

## Ratio Features

### Style Factors

#### `STYLE_GROWTH_VS_VALUE`
- **Formula**: `IVW / IVE`
- **Calculation**: Growth ETF divided by Value ETF
- **Interpretation**: Growth vs Value performance ratio
- **Regime Signal**: 
  - Rising: RISK_ON (growth outperforms)
  - Falling: RISK_OFF (value outperforms, defensive)
- **Typical Range**: 0.8 to 1.2
- **Components**: IVW, IVE

#### `STYLE_CYCLICAL_VS_DEFENSIVE`
- **Formula**: `XLF / XLU`
- **Calculation**: Financials ETF divided by Utilities ETF
- **Interpretation**: Cyclical vs Defensive sector performance
- **Regime Signal**: 
  - Rising: RISK_ON (cyclicals outperform)
  - Falling: RISK_OFF (defensives outperform)
- **Typical Range**: 0.5 to 2.0
- **Components**: XLF, XLU

#### `STYLE_SMALL_VS_LARGE`
- **Formula**: `RUT / GSPC`
- **Calculation**: Russell 2000 divided by S&P 500
- **Interpretation**: Small cap vs Large cap performance
- **Regime Signal**: 
  - Rising: RISK_ON (small caps outperform)
  - Falling: RISK_OFF (large caps outperform, flight to quality)
- **Typical Range**: 0.3 to 0.7
- **Components**: ^RUT, ^GSPC

### Cross-Asset Ratios

#### `FX_AUD_JPY`
- **Formula**: `(AUD/USD) / (USD/JPY)`
- **Calculation**: Implied AUD/JPY cross rate
- **Interpretation**: Risk-sensitive currency pair (AUD = risk-on, JPY = safe haven)
- **Regime Signal**: 
  - Rising: RISK_ON (carry trade works)
  - Falling: RISK_OFF (carry trade unwinds)
- **Typical Range**: 50 to 120
- **Components**: AUDUSD=X, USDJPY=X

#### `COMMODITY_GOLD_OIL`
- **Formula**: `GC=F / CL=F`
- **Calculation**: Gold price divided by WTI crude oil price
- **Interpretation**: Precious metal vs energy ratio
- **Regime Signal**: 
  - Rising: RISK_OFF or INFLATION_SHOCK (gold outperforms)
  - Falling: RISK_ON (oil outperforms with growth)
- **Typical Range**: 10 to 30
- **Components**: GC=F, CL=F

#### `EQUITY_EM_VS_US`
- **Formula**: `EEM / GSPC`
- **Calculation**: Emerging Markets ETF divided by S&P 500
- **Interpretation**: EM vs US equity performance
- **Regime Signal**: 
  - Rising: RISK_ON (EM outperforms)
  - Falling: RISK_OFF (US outperforms, flight to quality)
- **Typical Range**: 0.2 to 0.5
- **Components**: EEM, ^GSPC

### Volatility Ratios

#### `VOL_VIX_LEVEL`
- **Formula**: Raw value from ^VIX
- **Calculation**: CBOE Volatility Index level
- **Interpretation**: Equity market fear gauge
- **Regime Signal**: 
  - Low (< 15): RISK_ON
  - High (> 25): RISK_OFF
- **Typical Range**: 10 to 80
- **Component**: ^VIX

#### `VOL_VVIX_VIX`
- **Formula**: `VVIX / VIX`
- **Calculation**: Vol-of-vol divided by VIX
- **Interpretation**: Volatility of volatility (tail risk indicator)
- **Regime Signal**: 
  - Rising: Extreme fear, tail risk events
  - Falling: Calm markets
- **Typical Range**: 0.5 to 2.0
- **Components**: ^VVIX, ^VIX

---

## Global Aggregate Features

Global aggregates are stored with `TICKER='GLOBAL'` and `ASSET_CLASS='MACRO'`.

### Risk Appetite Aggregates

#### `GLOBAL_RISK_APPETITE`
- **Formula**: Normalized composite of:
  - Equity momentum (20-day returns)
  - Credit spread changes (inverse)
  - VIX changes (inverse)
- **Calculation**: Z-score normalized average of components
- **Interpretation**: Composite risk-on/risk-off indicator
- **Regime Signal**: 
  - Positive: RISK_ON
  - Negative: RISK_OFF
- **Typical Range**: -3 to +3 (z-scores)
- **Components**: Major equity indices, credit spreads, VIX

#### `GLOBAL_EQUITY_MOMENTUM`
- **Formula**: Average of 20-day returns across major equity indices
- **Calculation**: Mean of GSPC_RET_20D, DJI_RET_20D, IXIC_RET_20D
- **Interpretation**: Broad equity market momentum
- **Regime Signal**: Positive in RISK_ON, negative in RISK_OFF
- **Typical Range**: -15% to +15%
- **Components**: ^GSPC, ^DJI, ^IXIC

### Growth Aggregates

#### `GLOBAL_GROWTH_SIGNAL`
- **Formula**: Normalized composite of:
  - Leading Index changes (USSLIND)
  - Consumer Sentiment changes (UMCSENT)
- **Calculation**: Z-score normalized average
- **Interpretation**: Forward-looking growth indicator
- **Regime Signal**: 
  - Positive: Growth acceleration
  - Negative: Growth slowdown/recession
- **Typical Range**: -3 to +3 (z-scores)
- **Components**: USSLIND, UMCSENT

### Inflation Aggregates

#### `GLOBAL_INFLATION_EXPECTATIONS`
- **Formula**: Average of breakeven inflation rates
- **Calculation**: Mean of T5YIE, T10YIE
- **Interpretation**: Market-implied inflation expectations
- **Regime Signal**: 
  - Rising: INFLATION_SHOCK
  - Falling: DISINFLATION
- **Typical Range**: 1% to 4%
- **Components**: T5YIE, T10YIE

#### `GLOBAL_INFLATION_REALIZED`
- **Formula**: Average of realized inflation measures
- **Calculation**: Mean of YoY CPI and PCE changes
- **Interpretation**: Actual inflation rate
- **Regime Signal**: 
  - High (> 3%): INFLATION_SHOCK
  - Low (< 2%): DISINFLATION
- **Typical Range**: -1% to 10%
- **Components**: CPIAUCSL, PCEPI

### Monetary Policy Aggregates

#### `GLOBAL_YIELD_CURVE_SLOPE`
- **Formula**: `DGS10 - DGS2`
- **Calculation**: 2s10s yield curve spread
- **Interpretation**: Monetary policy and growth expectations
- **Regime Signal**: 
  - Steep (positive): RISK_ON, growth expectations
  - Flat/Inverted (negative): RISK_OFF, recession warning
- **Typical Range**: -100 bps to +300 bps
- **Components**: DGS2, DGS10

#### `GLOBAL_FINANCIAL_CONDITIONS`
- **Formula**: Raw value from NFCI
- **Calculation**: Chicago Fed National Financial Conditions Index
- **Interpretation**: Aggregate financial conditions (credit, leverage, risk)
- **Regime Signal**: 
  - Negative: Easy financial conditions (RISK_ON)
  - Positive: Tight financial conditions (RISK_OFF)
- **Typical Range**: -2 to +2
- **Component**: NFCI

---

## Feature Naming Conventions

- **Ticker-specific**: `{TICKER}_{FEATURE_TYPE}_{PERIOD}` (e.g., `GSPC_RET_20D`)
- **Global features**: `GLOBAL_{FEATURE_NAME}` (e.g., `GLOBAL_RISK_APPETITE`)
- **Spreads**: `{SPREAD_NAME}` (e.g., `YCURVE_2S10S`)
- **Changes**: `{TICKER}_CHG_{PERIOD}` (e.g., `DGS10_CHG_20D`)

## Versioning

All features are stored with a version tag (e.g., `V1_BASELINE`). Multiple versions can coexist, allowing comparison of different feature definitions or computation methods.

## Data Cleaning

Features are computed on cleaned data:
- **Missing data**: Forward-filled with limits (5 days for daily data, unlimited for monthly)
- **Outliers**: Detected but not automatically removed (may be regime-defining)
- **Frequency alignment**: Multi-ticker features align to common date index
- **Validation**: Pre and post-computation quality checks

## Usage

See `feature_engineering.ipynb` for interactive feature computation and exploration.

To compute all features:
```python
from features.compute import compute_all_features
from features.database import get_connection

conn = get_connection(password=db_password)
result = compute_all_features(conn, version='V1_BASELINE', start_date='2010-01-01')
```
