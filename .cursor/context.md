# Global Macro Regime Engine — Project Context

## 1. What this project is

This project is a **Global Macro Regime Engine**.

Its purpose is to:
- Ingest macroeconomic and cross-asset time series
- Compute interpretable, cross-asset features
- Infer **market-wide macro regimes** (e.g. RISK_ON, RISK_OFF, INFLATION_SHOCK)
- Store all results in a Postgres database for inspection, comparison, and reuse


---

## 2. Core design philosophy

- **Python does all logic**
  - data cleaning
  - feature engineering
  - regime inference
- **Postgres is passive**
  - stores raw observations
  - stores derived features
  - stores regime outputs
- **SQL contains no business logic**
  - only schema, indexing, and inspection queries

The database exists to:
- persist state between runs
- allow visual inspection via SQLTools
- compare multiple versions of features/regimes

---

## 3. What “regimes” mean in this project

Regimes are **market-wide macro states**, not ticker-specific labels.

Key properties:
- One regime per date per version
- Derived from **cross-asset features**
- Examples: RISK_ON, RISK_OFF, INFLATION_SHOCK, DISINFLATION

Ticker-specific regimes are explicitly *out of scope* unless added later.

---

## 4. Database schema (authoritative)

All tables live in the `MACRO` schema.

### RAW_SERIES
Stores raw macro time series observations.

- One row per (TICKER, DT)
- Append/upsert only
- Considered the immutable “truth”

Columns:
- TICKER
- ASSET_CLASS
- DT
- VALUE
- SOURCE
- INGESTED_AT

### FEATURES
Stores derived features in long form.

- Computed in Python
- Fully regeneratable
- Versioned

Conventions:
- Global features use:
  - TICKER = 'GLOBAL'
  - ASSET_CLASS = 'MACRO'

Columns:
- DT
- TICKER
- ASSET_CLASS
- FEATURE
- VALUE
- VERSION
- COMPUTED_AT

### REGIMES_GLOBAL
Stores market-wide regime labels.

- One row per (DT, VERSION)
- Never overwritten
- Multiple versions can coexist

Columns:
- DT
- REGIME
- SCORE (optional confidence)
- VERSION
- CREATED_AT

---

## 5. Typical workflow

1. Ingest raw data → `RAW_SERIES`
2. Compute features in Python → `FEATURES`
3. Infer regimes in Python → `REGIMES_GLOBAL`
4. Inspect outputs using SQLTools in Cursor
5. Iterate on features / regime logic
6. Store new versions, never overwrite old ones

---

## 6. Conventions used throughout the project

- Tickers are short, stable identifiers (e.g. SPX, DGS10, DXY)
- ASSET_CLASS is one of:
  - EQUITY
  - RATES
  - FX
  - CREDIT
  - COMMODITIES
  - VOL
  - MACRO (for global aggregates)
- VERSION strings are explicit and human-readable:
  - V0_RULES
  - V1_BASELINE
  - V2_HMM
- All timestamps are UTC unless stated otherwise

---

## 7. How Cursor should help

When assisting on this project, Cursor should:
- Prefer **simple, explicit Python**
- Avoid introducing unnecessary abstractions or frameworks
- Avoid pushing SQL-heavy logic
- Keep database interactions transparent and inspectable
- Respect that this is a research engine, not production infrastructure

If there is ambiguity, default to:
- clarity over cleverness
- reproducibility over performance
- explicit versioning over overwriting

---

## 8. Non-goals (important)

This project does NOT aim to:
- build execution systems
- optimise latency
- encode strategy logic in SQL
- prematurely productionise models
- tightly couple models to schema

Those concerns are intentionally deferred.

---

## 9. Mental model to keep in mind

Think of this project as:

> “A macro research notebook, except the notebook is persistent, versioned,
> inspectable, and lives across Python + Postgres.”