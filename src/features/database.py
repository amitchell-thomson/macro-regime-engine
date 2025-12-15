"""
Database operations for feature engineering.
Handles loading raw data and storing computed features.
"""

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from datetime import datetime
from typing import Optional


def get_raw_series(conn, ticker=None, asset_class=None, 
                   start_date=None, end_date=None):
    """
    Load raw series data from database.
    
    Args:
        conn: Database connection
        ticker: Optional ticker filter (single ticker or list)
        asset_class: Optional asset class filter
        start_date: Optional start date (YYYY-MM-DD or date object)
        end_date: Optional end date (YYYY-MM-DD or date object)
    
    Returns:
        DataFrame with columns: ticker, asset_class, dt, value, source
    """
    cursor = conn.cursor()
    
    query = """
        SELECT ticker, asset_class, dt, value, source
        FROM macro.raw_series
        WHERE 1=1
    """
    params = []
    
    if ticker is not None:
        if isinstance(ticker, str):
            query += " AND ticker = %s"
            params.append(ticker)
        elif isinstance(ticker, list):
            query += " AND ticker = ANY(%s)"
            params.append(ticker)
    
    if asset_class is not None:
        query += " AND asset_class = %s"
        params.append(asset_class)
    
    if start_date is not None:
        query += " AND dt >= %s"
        params.append(start_date)
    
    if end_date is not None:
        query += " AND dt <= %s"
        params.append(end_date)
    
    query += " ORDER BY ticker, dt"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    
    if not rows:
        return pd.DataFrame(columns=['ticker', 'asset_class', 'dt', 'value', 'source'])  # type: ignore
    
    # Convert rows to DataFrame - use dict construction to avoid type checker issues
    column_names = ['ticker', 'asset_class', 'dt', 'value', 'source']
    df = pd.DataFrame([dict(zip(column_names, row)) for row in rows])
    df['dt'] = pd.to_datetime(df['dt']).dt.date
    
    return df


def upsert_features(conn, df, version):
    """
    Insert or update features in FEATURES table using bulk operations.
    
    Args:
        conn: Database connection
        df: DataFrame with columns: dt, ticker, asset_class, feature, value
        version: Version string (e.g., 'V1_BASELINE')
    
    Returns:
        Number of rows inserted/updated
    """
    if df.empty:
        return 0
    
    cursor = conn.cursor()
    
    # Prepare data as list of tuples
    computed_at = datetime.now()
    data = [
        (
            row['dt'],
            row['ticker'],
            row['asset_class'],
            row['feature'],
            float(row['value']) if pd.notna(row['value']) else None,
            version,
            computed_at
        )
        for _, row in df.iterrows()
    ]
    
    # Use execute_values for bulk insert with ON CONFLICT
    query = """
        INSERT INTO macro.features 
            (dt, ticker, asset_class, feature, value, version, computed_at)
        VALUES %s
        ON CONFLICT (dt, ticker, feature, version) 
        DO UPDATE SET 
            value = EXCLUDED.value,
            asset_class = EXCLUDED.asset_class,
            computed_at = EXCLUDED.computed_at
    """
    
    # Execute in batches for better memory management
    batch_size = 10000
    row_count = 0
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        execute_values(cursor, query, batch, page_size=1000)
        row_count += len(batch)
        
        # Print progress for large datasets
        if i + batch_size < len(data) and (i + batch_size) % 100000 == 0:
            print(f"  Inserted {i + batch_size:,} / {len(data):,} rows...")
    
    conn.commit()
    cursor.close()
    
    return row_count


def get_features(conn, feature=None, ticker=None, version=None,
                 start_date=None, end_date=None):
    """
    Load features from database.
    
    Args:
        conn: Database connection
        feature: Optional feature name filter
        ticker: Optional ticker filter
        version: Optional version filter
        start_date: Optional start date
        end_date: Optional end date
    
    Returns:
        DataFrame with columns: dt, ticker, asset_class, feature, value, version
    """
    cursor = conn.cursor()
    
    query = """
        SELECT dt, ticker, asset_class, feature, value, version
        FROM macro.features
        WHERE 1=1
    """
    params = []
    
    if feature is not None:
        query += " AND feature = %s"
        params.append(feature)
    
    if ticker is not None:
        query += " AND ticker = %s"
        params.append(ticker)
    
    if version is not None:
        query += " AND version = %s"
        params.append(version)
    
    if start_date is not None:
        query += " AND dt >= %s"
        params.append(start_date)
    
    if end_date is not None:
        query += " AND dt <= %s"
        params.append(end_date)
    
    query += " ORDER BY dt, ticker, feature"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    
    if not rows:
        return pd.DataFrame(columns=['dt', 'ticker', 'asset_class', 'feature', 'value', 'version'])  # type: ignore
    
    # Convert rows to DataFrame - use dict construction to avoid type checker issues
    column_names = ['dt', 'ticker', 'asset_class', 'feature', 'value', 'version']
    df = pd.DataFrame([dict(zip(column_names, row)) for row in rows])
    df['dt'] = pd.to_datetime(df['dt']).dt.date
    
    return df
