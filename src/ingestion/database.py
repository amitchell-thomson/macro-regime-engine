"""
Database operations for ingestion.
Handles connections, inserts, and logging.
"""

import psycopg2
from datetime import datetime

def get_connection(host='localhost', database='macro', 
                   user='macro_user', password=None):
    """
    Create database connection.
    
    Returns:
        psycopg2 connection object
    """
    return psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )


def upsert_raw_series(conn, df, asset_class):
    """
    Insert or update data in raw_series table.
    
    Args:
        conn: Database connection
        df: DataFrame with columns: ticker, dt, value, source
        asset_class: Asset class (EQUITY, RATES, etc.)
    
    Returns:
        Number of rows inserted
    """
    cursor = conn.cursor()
    row_count = 0
    
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO macro.raw_series 
                (ticker, asset_class, dt, value, source, ingested_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, dt) 
            DO UPDATE SET 
                value = EXCLUDED.value,
                ingested_at = EXCLUDED.ingested_at;
        """, (
            row['ticker'],
            asset_class,
            row['dt'],
            row['value'],
            row['source'],
            datetime.now()
        ))
        row_count += 1
    
    conn.commit()
    cursor.close()
    
    return row_count


def close_connection(conn):
    """Close database connection."""
    if conn:
        conn.close()