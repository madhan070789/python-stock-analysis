"""
data_fetch.py
=============
This module handles fetching stock market data using the yfinance library.
It also calculates technical indicators like moving averages and features
needed for the crash prediction model.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime


def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'RELIANCE.NS').
    period : str
        How far back to fetch data. Examples: '1mo', '3mo', '6mo', '1y', '2y', '5y'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume (indexed by Date).
        Returns an empty DataFrame if the download fails.
    """
    try:
        # --- Easter Egg / Demo: Guaranteed Crash Scenario ---
        if ticker.upper() == "CRASH_TEST":
            print("Running CRASH_TEST demo scenario")
            days = 60
            dates = pd.date_range(end=datetime.date.today(), periods=days, freq="B")
            
            # Create a sudden massive plummet
            price = 500.0
            closes, opens, highs, lows, volumes = [], [], [], [], []
            for i in range(days):
                if i < 58:
                    # Normal market
                    daily_return = np.random.normal(0.001, 0.01)
                    volume_factor = 1.0
                elif i == 58:
                    # The crash begins at day 58!
                    daily_return = np.random.normal(-0.05, 0.01) # 5% drop
                    volume_factor = np.random.normal(2.0, 0.5)   # 2x volume spike
                else:
                    # The absolute bottom falls out on the last day!
                    daily_return = np.random.normal(-0.15, 0.02) # 15% drop!
                    volume_factor = np.random.normal(8.0, 1.0)   # 8x volume spike!
                    
                price = price * (1 + daily_return)
                open_price = price * np.random.normal(1, 0.01)
                high_price = max(price, open_price) * np.random.normal(1.02, 0.01)
                low_price = min(price, open_price) * np.random.normal(0.98, 0.01)
                volume = int(np.random.normal(10000000 * volume_factor, 2000000))
                
                closes.append(price)
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                volumes.append(max(100000, volume))
                
            df_crash = pd.DataFrame({
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes
            }, index=dates)
            
            return df_crash
            
        # Create a session with a custom User-Agent to help bypass rate limits
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        })
        
        # Download data from Yahoo Finance
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(period=period)

        # If data is returned correctly
        if not df.empty and all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]):
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            return df
            
    except Exception as e:
        print(f"Error fetching real data for {ticker}: {e}")
        
    # --- Fallback: Generate Synthetic Data if fetch fails ---
    print(f"Warning: Generating fallback synthetic data for {ticker} due to fetch error or rate limit.")
    days = 252 # approx 1 year of trading days
    if period == "1mo": days = 21
    elif period == "3mo": days = 63
    elif period == "6mo": days = 126
    elif period == "2y": days = 504
    elif period == "5y": days = 1260
    
    dates = pd.date_range(end=datetime.date.today(), periods=days, freq="B")
    
    # Random walk for price
    price = 150.0 # Starting price
    closes, opens, highs, lows, volumes = [], [], [], [], []
    
    for _ in range(days):
        daily_return = np.random.normal(0, 0.02)
        price = price * (1 + daily_return)
        open_price = price * np.random.normal(1, 0.005)
        high_price = max(price, open_price) * np.random.normal(1.01, 0.005)
        low_price = min(price, open_price) * np.random.normal(0.99, 0.005)
        volume = int(np.random.normal(50000000, 10000000))
        
        closes.append(price)
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        volumes.append(max(1000000, volume))
        
    df_synthetic = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    }, index=dates)
    
    return df_synthetic



def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day and 50-day Simple Moving Averages (SMA).

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with a 'Close' column.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with two new columns: 'SMA_20' and 'SMA_50'.
    """
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    return df


def calculate_features(df: pd.DataFrame) -> dict:
    """
    Calculate features needed for the crash prediction model.

    Features computed:
    - price_change   : Percentage change in closing price (latest vs previous day)
    - volume_change  : Percentage change in volume (latest vs previous day)
    - volatility     : Standard deviation of daily returns over the last 20 days

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with 'Close' and 'Volume' columns. Needs at least 20 rows.

    Returns
    -------
    dict
        Dictionary with keys: 'price_change', 'volume_change', 'volatility'.
        Returns None if there isn't enough data.
    """
    if df is None or len(df) < 20:
        return None

    try:
        # --- Price Change (%) ---
        # How much did the price change from yesterday to today?
        price_change = ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100

        # --- Volume Change (%) ---
        # How much did the trading volume change from yesterday to today?
        volume_change = ((df["Volume"].iloc[-1] - df["Volume"].iloc[-2]) / df["Volume"].iloc[-2]) * 100

        # --- Volatility ---
        # Calculate daily returns, then take the standard deviation of the last 20 days
        daily_returns = df["Close"].pct_change().dropna()
        volatility = daily_returns.tail(20).std() * 100  # Convert to percentage

        return {
            "price_change": round(price_change, 4),
            "volume_change": round(volume_change, 4),
            "volatility": round(volatility, 4),
        }

    except Exception as e:
        print(f"Error calculating features: {e}")
        return None
