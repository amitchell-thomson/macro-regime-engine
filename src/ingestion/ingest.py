"""
Main ingestion orchestrator.
Coordinates downloading and storing of all universe data.
"""

import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from download import download_ticker
from database import (
    get_connection, 
    upsert_raw_series, 
    log_ingestion_run,
    close_connection
)


def load_universe():
    """Load tickers from universe.yml"""
    universe_path = Path(__file__).parent.parent.parent / 'configs' / 'universe.yml'
    
    with open(universe_path, 'r') as f:
        universe = yaml.safe_load(f)
    
    # Flatten into list
    tickers = []
    for asset_class, ticker_list in universe.items():
        for item in ticker_list:
            tickers.append({
                'ticker': item['ticker'],
                'name': item['name'],
                'source': item['source'],
                'asset_class': asset_class.upper()
            })
    
    return tickers


def ingest_all(start_date='2010-01-01', ticker_filter=None, dry_run=False):
    """
    Main ingestion function.
    
    Args:
        start_date: Start date for historical data
        ticker_filter: If provided, only ingest this ticker
        dry_run: If True, download but don't store in database
    """
    # Load environment
    load_dotenv()
    fred_api_key = os.getenv('FRED_API_KEY')
    db_password = os.getenv('DB_PASSWORD')
    
    # Load universe
    tickers = load_universe()
    
    if ticker_filter:
        tickers = [t for t in tickers if t['ticker'] == ticker_filter]
    
    print(f"Ingesting {len(tickers)} tickers...")
    
    # Connect to database
    conn = None if dry_run else get_connection(password=db_password)
    
    # Start ingestion log
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not dry_run:
        log_ingestion_run(conn, run_id, 'running', f'Ingesting {len(tickers)} tickers')
    
    success_count = 0
    fail_count = 0
    
    # Process each ticker
    for ticker_info in tickers:
        ticker = ticker_info['ticker']
        source = ticker_info['source']
        asset_class = ticker_info['asset_class']
        
        print(f"Processing {ticker} ({asset_class}) from {source}...")
        
        # Download
        df = download_ticker(ticker, source, fred_api_key, start_date)
        
        if df is not None and not df.empty:
            if not dry_run:
                # Store in database
                row_count = upsert_raw_series(conn, df, asset_class)
                print(f"  ✓ Inserted {row_count} rows")
            else:
                print(f"  ✓ Downloaded {len(df)} rows (dry run - not stored)")
            
            success_count += 1
        else:
            print(f"  ✗ Failed to download")
            fail_count += 1
    
    # Finish ingestion log
    if not dry_run:
        log_ingestion_run(
            conn, run_id, 'ok', 
            f'Success: {success_count}, Failed: {fail_count}'
        )
        close_connection(conn)
    
    print(f"\nIngestion complete! Success: {success_count}, Failed: {fail_count}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Ingest macro data')
    parser.add_argument('--start-date', default='2010-01-01', help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--ticker', help='Ingest single ticker only')
    parser.add_argument('--dry-run', action='store_true',
                       help='Download but do not store in database')
    
    args = parser.parse_args()
    
    ingest_all(
        start_date=args.start_date,
        ticker_filter=args.ticker,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()