"""
Download data from yfinance and FRED API.
Returns standardized pandas DataFrames.
"""

import yfinance as yf
import pandas as pd
from fredapi import Fred
from datetime import datetime

def download_yfinance(ticker, start_date='2010-01-01'):
    """
    Download data from Yahoo Finance.
    
    Args:
        ticker: Ticker symbol (e.g., '^GSPC', 'GC=F')
        start_date: Start date for historical data
    
    Returns:
        DataFrame with columns: ticker, dt, value, source
        Or None if download fails
    """
    try:
        data: pd.DataFrame = pd.DataFrame(yf.download(ticker, start=start_date, progress=False))
        
        # Extract Close column and ensure it's 1-dimensional
        close_values = data['Close']
        
        df = pd.DataFrame({
            'ticker': ticker,
            'dt': data.index,
            'value': close_values,
            'source': 'yfinance'
        })
        
        return df
    
    except Exception as e:
        print(f"Error downloading {ticker} from yfinance: {e}")
        return None


def download_fred(ticker, api_key, start_date='2010-01-01'):
    """
    Download data from FRED API.
    
    Args:
        ticker: FRED series ID (e.g., 'DGS10', 'CPIAUCSL')
        api_key: FRED API key
        start_date: Start date for historical data
    
    Returns:
        DataFrame with columns: ticker, dt, value, source
        Or None if download fails
    """
    try:
        fred = Fred(api_key=api_key)
        series = fred.get_series(ticker, observation_start=start_date)
        
        df = pd.DataFrame({
            'ticker': ticker,
            'dt': series.index,
            'value': series.values,
            'source': 'fred'
        })
        
        return df
    
    except Exception as e:
        print(f"Error downloading {ticker} from FRED: {e}")
        return None


def download_ticker(ticker, source, fred_api_key=None, start_date='2010-01-01'):
    """
    Download ticker from appropriate source.
    
    Args:
        ticker: Ticker symbol
        source: 'yfinance' or 'fred'
        fred_api_key: Required if source is 'fred'
        start_date: Start date for historical data
    
    Returns:
        DataFrame or None
    """
    if source == 'yfinance':
        return download_yfinance(ticker, start_date)
    elif source == 'fred':
        return download_fred(ticker, fred_api_key, start_date)
    else:
        print(f"Unknown source: {source}")
        return None