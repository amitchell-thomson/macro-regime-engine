# Universe Configuration

## Overview

The `universe.yml` file defines the 59 tickers that form the data foundation for regime detection. This universe provides comprehensive cross-asset coverage across 7 asset classes.

## Why These Assets Matter for Regime Detection

Macro regimes are **market-wide states** characterized by coordinated movements across multiple asset classes. No single ticker can tell you the regime - you need cross-asset signals. This universe is designed to capture:

- **Risk appetite** (equities, credit spreads, volatility)
- **Growth expectations** (leading indicators, cyclical vs defensive)
- **Inflation dynamics** (realized inflation, expectations, commodities)
- **Monetary policy** (rates, Fed balance sheet, financial conditions)
- **Liquidity conditions** (credit spreads, Fed assets, crypto)
- **Global flows** (FX, emerging markets, international equities)

---

## Asset Classes & Rationale

### 1. **Equity (13 tickers)**

**Global Indices (9 tickers)**
- `^GSPC` (S&P 500), `^DJI` (Dow), `^IXIC` (Nasdaq)
- `^RUT` (Russell 2000 - small cap)
- `EEM` (Emerging Markets)
- `^FTSE` (UK), `^N225` (Japan), `000001.SS` (China), `^STOXX50E` (Europe)

**Why**: Global equity indices capture risk sentiment and growth expectations. Small caps (^RUT) and emerging markets (EEM) are particularly sensitive to risk-on/risk-off dynamics.

**Style Factors (4 tickers)**
- `IVW` (Growth) vs `IVE` (Value)
- `XLU` (Utilities - defensive) vs `XLF` (Financials - cyclical)

**Why**: Style rotation is a **direct regime signal**. In risk-on regimes, growth and cyclicals outperform. In risk-off or recession fears, value and defensives dominate. Rate-sensitive sectors (XLF) signal monetary policy regime shifts.

---

### 2. **Rates (11 tickers)**

**Treasury Curve (5 tickers)**
- `DGS1`, `DGS2`, `DGS5`, `DGS10`, `DGS30`

**Why**: The yield curve shape signals growth and recession expectations. Inversions (short rates > long rates) predict recessions.

**Spreads (2 tickers)**
- `T10Y2Y` (2s10s spread - recession indicator)
- `T10Y3M` (10Y minus 3M - credit/term premium)

**Why**: These spreads are among the best leading indicators of recessions and growth slowdowns.

**Policy Rates (2 tickers)**
- `FEDFUNDS` (Fed Funds Rate)
- `SOFR` (Secured Overnight Financing Rate)

**Why**: Central bank policy rates define the monetary regime. SOFR captures money market stress.

**Inflation Expectations & Real Rates (3 tickers)**
- `T5YIE` (5-year breakeven inflation)
- `T10YIE` (10-year breakeven inflation)
- `DFII10` (10-year TIPS / real rate)

**Why Critical**: Real rates are the **key driver of asset allocation**. Negative real rates = "TINA" (there is no alternative to stocks). Inflation expectations distinguish "inflation shock" from "disinflation" regimes. You need *expected* inflation, not just realized.

---

### 3. **FX (6 tickers)**

- `DX-Y.NYB` (US Dollar Index)
- `EURUSD=X`, `GBPUSD=X`, `USDJPY=X`
- `AUDUSD=X` (risk-sensitive commodity currency)
- `USDCNY=X` (China exposure)

**Why**: The dollar is a macro regime unto itself. Dollar strength/weakness impacts:
- Emerging markets (strong dollar = EM stress)
- Commodities (inverse correlation)
- US corporate earnings (exports)
- Risk appetite (AUD is a classic risk proxy)

---

### 4. **Credit (4 tickers)**

- `BAMLH0A0HYM2` (High Yield spread)
- `BAMLC0A0CM` (Investment Grade spread)
- `TEDRATE` (TED spread - bank stress)
- `T10Y3M` (Term spread)

**Why**: Credit spreads widen in stress and tighten in calm/growth. High yield spreads are an **early warning system** for risk-off and recession. TED spread captures banking sector stress.

---

### 5. **Commodities (7 tickers)**

**Precious Metals**
- `GC=F` (Gold), `SI=F` (Silver)

**Why**: Gold is the ultimate risk-off and inflation hedge asset.

**Energy**
- `CL=F` (WTI Crude), `BZ=F` (Brent Crude), `NG=F` (Natural Gas)

**Why**: Oil prices signal inflation and growth expectations. Energy shocks define their own regime (e.g., 1970s stagflation, 2022 energy crisis).

**Industrial Metals**
- `HG=F` (Copper - "Dr. Copper")

**Why**: Copper has predictive power for global growth due to its industrial use.

**Crypto**
- `BTC-USD` (Bitcoin)

**Why**: Bitcoin has emerged as a liquidity and risk proxy. Highly correlated with QE/QT cycles and tech/growth stocks. Acts as "risk-on steroids" in recent cycles.

---

### 6. **Volatility (3 tickers)**

- `^VIX` (CBOE Volatility Index - equity vol)
- `^VVIX` (Vol-of-vol - tail risk)
- `^MOVE` (Bond volatility)

**Why**: VIX is the "fear gauge." Elevated VIX = risk-off regime. VVIX captures tail risk and dealer positioning. MOVE captures bond market stress (critical during rate regime shifts).

---

### 7. **Macro Indicators (11 tickers)**

**Inflation (3 tickers)**
- `CPIAUCSL` (CPI - headline inflation)
- `PCEPI` (PCE - Fed's preferred gauge)
- `DCOILWTICO` (Oil spot price)

**Why**: Realized inflation defines whether we're in inflation shock, disinflation, or stable regimes.

**Growth & Employment (3 tickers)**
- `GDP` (Real GDP)
- `UNRATE` (Unemployment rate)
- `UMCSENT` (Consumer sentiment)

**Why**: Growth and employment are lagging indicators of regime, but confirm regime transitions.

**Leading Indicators (2 tickers)**
- `NAPM` (ISM Manufacturing PMI)
- `USSLIND` (US Leading Index)

**Why Critical**: These are **forward-looking**. PMI below 50 = contraction. Leading index turns before GDP. Essential for detecting regime transitions early.

**Liquidity & Financial Conditions (2 tickers)**
- `NFCI` (Chicago Fed Financial Conditions Index)
- `WALCL` (Fed Balance Sheet)

**Why Critical**: NFCI aggregates credit, leverage, and risk into one index—a direct regime measure. Fed balance sheet defines liquidity regimes (QE = risk-on, QT = risk-off). These are **regime-defining variables**.

**Money Supply (1 ticker)**
- `M2SL` (M2 Money Stock)

**Why**: Liquidity and inflation proxy.

---

## Data Sources

All 59 tickers are available from **free sources**:

- **yfinance** (33 tickers): Equities, FX, commodities, volatility indices, crypto
- **FRED API** (26 tickers): Rates, credit spreads, macro indicators, financial conditions

Both have excellent Python libraries and provide daily or lower frequency data suitable for macro regime detection.

---

## How This Universe Enables Regime Detection

Macro regimes emerge from **cross-asset correlations**. For example:

**RISK_ON Regime:**
- Equities ↑, Credit spreads ↓, VIX ↓
- Growth > Value, High Beta > Low Vol
- AUD/JPY ↑, Dollar ↓
- Commodities ↑, EM outperforms

**RISK_OFF Regime:**
- Equities ↓, Credit spreads ↑, VIX ↑
- Value > Growth, Defensives (XLU) outperform
- Gold ↑, Dollar ↑
- EM underperforms, HY spreads widen

**INFLATION_SHOCK Regime:**
- Commodities ↑↑, Breakeven inflation ↑
- Real rates ↓ (nominal rates lag inflation)
- Gold ↑, Value > Growth
- Fed tightening → Financial conditions tighten (NFCI ↑)

**DISINFLATION/GOLDILOCKS Regime:**
- Equities ↑, Bonds ↑ (falling yields)
- Inflation expectations ↓, Real rates stable/positive
- Credit spreads tight, VIX low
- Everything rallies (stocks, bonds, EM)

By tracking all 59 tickers, the engine can identify which regime is in play based on the **pattern** of cross-asset moves, not just individual tickers.
