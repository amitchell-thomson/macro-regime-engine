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


def log_ingestion_run(conn, run_id, status, message=None):
    """
    Log ingestion run to ingest_log table.
    
    Args:
        conn: Database connection
        run_id: Unique identifier for this run
        status: 'running', 'ok', or 'failed'
        message: Optional message
    """
    cursor = conn.cursor()
    
    if status == 'running':
        cursor.execute("""
            INSERT INTO macro.ingest_log 
                (run_id, started_at, status, message)
            VALUES (%s, %s, %s, %s);
        """, (run_id, datetime.now(), status, message))
    else:
        cursor.execute("""
            UPDATE macro.ingest_log
            SET finished_at = %s, status = %s, message = %s
            WHERE run_id = %s;
        """, (datetime.now(), status, message, run_id))
    
    conn.commit()
    cursor.close()


def close_connection(conn):
    """Close database connection."""
    if conn:
        conn.close()