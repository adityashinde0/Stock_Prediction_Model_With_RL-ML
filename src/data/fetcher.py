"""
Market Data Ingestion Module
Fetches real-time and historical OHLCV data from Yahoo Finance based on OCR-detected Tickers.
"""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config import DEFAULT_TICKER, DEFAULT_TIMEFRAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MarketDataFetcher")


class MarketDataFetcher:
    """
    Handles robust downloading and formatting of OHLCV financial data.
    """

    @staticmethod
    def fetch_data(
        ticker: str = DEFAULT_TICKER,
        period: str = "2y",
        interval: str = DEFAULT_TIMEFRAME,
    ) -> pd.DataFrame:
        """
        Downloads historical price data and returns a clean, single-level DataFrame
        containing Open, High, Low, Close, Adj Close, and Volume.
        """
        logger.info(f"Fetching market data for [{ticker}] | Period: {period} | Interval: {interval}...")

        try:
            data = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                ignore_tz=True,
                prepost=False,
                auto_adjust=False,  # Retain separate 'Adj Close' column
                progress=False,
            )
        except Exception as e:
            logger.error(f"Failed to download data for {ticker}: {e}")
            return pd.DataFrame()

        # Handle yfinance MultiIndex column behavior
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Validate required columns
        required_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if not required_cols.issubset(set(data.columns)) or len(data) == 0:
            logger.warning(f"Insufficient data returned for {ticker}. Trying default fallback ({DEFAULT_TICKER})...")
            if ticker != DEFAULT_TICKER:
                return MarketDataFetcher.fetch_data(ticker=DEFAULT_TICKER, period=period, interval=interval)
            return pd.DataFrame()

        # Remove duplicate timestamps and sort ascending
        data = data[~data.index.duplicated(keep="first")].sort_index()

        logger.info(f"Successfully loaded {len(data)} rows of market data for {ticker}.")
        return data


# --- Standalone Verification Block ---
if __name__ == "__main__":
    df_test = MarketDataFetcher.fetch_data(ticker="BTC-USD", period="1y", interval="1d")
    print("\n--- Market Data Fetcher Verification ---")
    print(df_test.tail(3))