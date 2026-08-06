"""
Technical Indicator Engineering Module
Calculates Momentum, Trend, Volatility, and Volume indicators for ML/RL feature vectors.
"""

import logging
import pandas as pd
import ta

from src.config import INDICATOR_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TechnicalIndicators")


class TechnicalIndicatorEngine:
    """
    Transforms raw OHLCV market data into rich feature vectors for model inference.
    """

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes EMAs, MACD, RSI, OBV, and Bollinger Bands.
        Returns a cleaned DataFrame with NaN rows removed.
        """
        if df.empty or len(df) < 50:
            logger.warning("DataFrame is too small to calculate technical indicators.")
            return df

        data = df.copy()

        try:
            # --- 1. Exponential Moving Averages (EMA) ---
            data["EMA7"] = ta.trend.EMAIndicator(
                close=data["Adj Close"], window=INDICATOR_PARAMS["ema_fast"], fillna=False
            ).ema_indicator()
            data["EMA21"] = ta.trend.EMAIndicator(
                close=data["Adj Close"], window=INDICATOR_PARAMS["ema_slow"], fillna=False
            ).ema_indicator()
            data["EMA50"] = ta.trend.EMAIndicator(
                close=data["Adj Close"], window=50, fillna=False
            ).ema_indicator()
            data["EMA200"] = ta.trend.EMAIndicator(
                close=data["Adj Close"], window=INDICATOR_PARAMS["ema_trend"], fillna=False
            ).ema_indicator()

            # --- 2. MACD (Moving Average Convergence Divergence) ---
            macd = ta.trend.MACD(
                close=data["Adj Close"],
                window_slow=INDICATOR_PARAMS["macd_slow"],
                window_fast=INDICATOR_PARAMS["macd_fast"],
                window_sign=INDICATOR_PARAMS["macd_signal"],
                fillna=False,
            )
            data["MACD_line"] = macd.macd()
            data["MACD_signal"] = macd.macd_signal()
            data["MACD_diff"] = macd.macd_diff()

            # --- 3. Relative Strength Index (RSI) ---
            data["RSI"] = ta.momentum.RSIIndicator(
                close=data["Adj Close"], window=INDICATOR_PARAMS["rsi_period"], fillna=False
            ).rsi()

            # --- 4. On-Balance Volume (OBV) ---
            data["OBV"] = ta.volume.OnBalanceVolumeIndicator(
                close=data["Adj Close"], volume=data["Volume"], fillna=False
            ).on_balance_volume()

            # --- 5. Bollinger Bands (BB) ---
            bb = ta.volatility.BollingerBands(
                close=data["Adj Close"],
                window=INDICATOR_PARAMS["bollinger_window"],
                window_dev=INDICATOR_PARAMS["bollinger_std"],
                fillna=False,
            )
            data["BBH"] = bb.bollinger_hband_indicator()
            data["BBL"] = bb.bollinger_lband_indicator()
            data["BB_width"] = bb.bollinger_wband()

            # --- 6. Returns & Target Features ---
            data["Return"] = data["Adj Close"].pct_change()

            # Clean warm-up NaN rows caused by long-window indicators (e.g., EMA200)
            clean_data = data.dropna().reset_index(drop=True)
            logger.info(f"Feature Engineering Complete. Shape after cleaning: {clean_data.shape}")
            return clean_data

        except Exception as e:
            logger.error(f"Error computing technical indicators: {e}")
            return df


# --- Standalone Verification Block ---
if __name__ == "__main__":
    from src.data.fetcher import MarketDataFetcher

    raw_df = MarketDataFetcher.fetch_data(ticker="BTC-USD", period="2y", interval="1d")
    processed_df = TechnicalIndicatorEngine.add_all_indicators(raw_df)

    print("\n--- Technical Indicator Engine Verification ---")
    print("Columns available:", list(processed_df.columns))
    print(processed_df[["Adj Close", "EMA7", "RSI", "MACD_diff", "BB_width"]].tail(3))