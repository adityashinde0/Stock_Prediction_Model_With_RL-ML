"""
Technical Indicator Engineering Module
Calculates Momentum, Trend, Volatility, Volume, and Seasonality features
for ML/RL/DL model feature vectors.

Feature set (21 total):
  Trend      : EMA7, EMA21, EMA50, EMA200
  Momentum   : MACD_line, MACD_signal, MACD_diff, RSI
  Volume     : OBV
  Volatility : BBH, BBL, BB_width, RealizedVol, ATR_norm
  Return     : Return, LogReturn
  Regime     : DistEMA200
  Seasonality: DayOfWeek_sin, DayOfWeek_cos, Month_sin, Month_cos
"""

import logging

import numpy as np
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
        Computes all 21 institutional quantitative features and returns a
        cleaned DataFrame with warm-up NaN rows removed.

        New features vs. original:
          LogReturn    : ln(P_t / P_{t-1})  — cleaner distributional properties
          RealizedVol  : 20-day rolling std(LogReturn) × √252 — annualised
          DistEMA200   : (Adj Close - EMA200) / EMA200 — regime distance
          ATR_norm     : ATR(14) / Close — normalised intraday bar expansion
          DayOfWeek_sin/cos : sine/cosine of ISO day-of-week (0=Mon … 6=Sun)
          Month_sin/cos     : sine/cosine of calendar month (1–12)

        Cyclical features are extracted from the DataFrame's DatetimeIndex
        BEFORE reset_index(drop=True) to avoid losing timestamp information.
        """
        if df.empty or len(df) < 50:
            logger.warning("DataFrame is too small to calculate technical indicators.")
            return df

        data = df.copy()

        try:
            # ── 1. Exponential Moving Averages ───────────────────────────────
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

            # ── 2. MACD ──────────────────────────────────────────────────────
            macd = ta.trend.MACD(
                close=data["Adj Close"],
                window_slow=INDICATOR_PARAMS["macd_slow"],
                window_fast=INDICATOR_PARAMS["macd_fast"],
                window_sign=INDICATOR_PARAMS["macd_signal"],
                fillna=False,
            )
            data["MACD_line"]   = macd.macd()
            data["MACD_signal"] = macd.macd_signal()
            data["MACD_diff"]   = macd.macd_diff()

            # ── 3. RSI ───────────────────────────────────────────────────────
            data["RSI"] = ta.momentum.RSIIndicator(
                close=data["Adj Close"],
                window=INDICATOR_PARAMS["rsi_period"],
                fillna=False,
            ).rsi()

            # ── 4. On-Balance Volume ─────────────────────────────────────────
            data["OBV"] = ta.volume.OnBalanceVolumeIndicator(
                close=data["Adj Close"], volume=data["Volume"], fillna=False
            ).on_balance_volume()

            # ── 5. Bollinger Bands ───────────────────────────────────────────
            bb = ta.volatility.BollingerBands(
                close=data["Adj Close"],
                window=INDICATOR_PARAMS["bollinger_window"],
                window_dev=INDICATOR_PARAMS["bollinger_std"],
                fillna=False,
            )
            data["BBH"]      = bb.bollinger_hband_indicator()
            data["BBL"]      = bb.bollinger_lband_indicator()
            data["BB_width"] = bb.bollinger_wband()

            # ── 6. Simple Percentage Return ──────────────────────────────────
            data["Return"] = data["Adj Close"].pct_change()

            # ── 7. Log Return ────────────────────────────────────────────────
            # ln(P_t / P_{t-1}) — better normality, additive across periods
            data["LogReturn"] = np.log(
                data["Adj Close"] / data["Adj Close"].shift(1)
            )

            # ── 8. Historical Realized Volatility (annualised) ───────────────
            # 20-day rolling σ of log returns × √252
            data["RealizedVol"] = (
                data["LogReturn"].rolling(window=20).std() * np.sqrt(252)
            )

            # ── 9. Distance from EMA200 (regime indicator) ───────────────────
            # Positive → price above trend; negative → below trend
            data["DistEMA200"] = (data["Adj Close"] - data["EMA200"]) / data["EMA200"]

            # ── 10. Normalised Average True Range ────────────────────────────
            # ATR(14) / Close → dimensionless intraday volatility measure
            atr = ta.volatility.AverageTrueRange(
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                window=14,
                fillna=False,
            )
            data["ATR"]      = atr.average_true_range()
            data["ATR_norm"] = data["ATR"] / data["Close"]

            # ── 11. Cyclical Date Features (from DatetimeIndex) ──────────────
            # Extracted BEFORE reset_index(drop=True) to retain timestamp info.
            # Sine/cosine encoding avoids discontinuities at day/month boundaries.
            if isinstance(data.index, pd.DatetimeIndex):
                dow   = data.index.dayofweek.astype(float)   # 0=Mon … 6=Sun
                month = data.index.month.astype(float)        # 1 … 12

                data["DayOfWeek_sin"] = np.sin(2.0 * np.pi * dow   / 7.0)
                data["DayOfWeek_cos"] = np.cos(2.0 * np.pi * dow   / 7.0)
                data["Month_sin"]     = np.sin(2.0 * np.pi * month / 12.0)
                data["Month_cos"]     = np.cos(2.0 * np.pi * month / 12.0)
            elif "Date" in data.columns:
                dt    = pd.to_datetime(data["Date"])
                dow   = dt.dt.dayofweek.astype(float)
                month = dt.dt.month.astype(float)

                data["DayOfWeek_sin"] = np.sin(2.0 * np.pi * dow   / 7.0)
                data["DayOfWeek_cos"] = np.cos(2.0 * np.pi * dow   / 7.0)
                data["Month_sin"]     = np.sin(2.0 * np.pi * month / 12.0)
                data["Month_cos"]     = np.cos(2.0 * np.pi * month / 12.0)
            else:
                # Fallback: zero-fill if no datetime information available
                logger.warning(
                    "No DatetimeIndex or Date column found — "
                    "cyclical date features set to 0."
                )
                for col in ["DayOfWeek_sin", "DayOfWeek_cos", "Month_sin", "Month_cos"]:
                    data[col] = 0.0

            # ── 12. Clean warm-up NaN rows (long-window indicators: EMA200, RealizedVol)
            clean_data = data.dropna().reset_index(drop=True)

            # Drop the intermediate ATR column (ATR_norm is the normalised form used)
            if "ATR" in clean_data.columns:
                clean_data = clean_data.drop(columns=["ATR"])

            logger.info(
                f"Feature Engineering Complete. Shape after cleaning: {clean_data.shape}"
            )
            return clean_data

        except Exception as exc:
            logger.error(f"Error computing technical indicators: {exc}")
            return df


# ── Standalone Verification Block ────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.fetcher import MarketDataFetcher

    raw_df       = MarketDataFetcher.fetch_data(ticker="BTC-USD", period="2y", interval="1d")
    processed_df = TechnicalIndicatorEngine.add_all_indicators(raw_df)

    NEW_COLS = [
        "LogReturn", "RealizedVol", "DistEMA200",
        "ATR_norm",
        "DayOfWeek_sin", "DayOfWeek_cos",
        "Month_sin", "Month_cos",
    ]

    print("\n--- Technical Indicator Engine Verification ---")
    print(f"Total columns     : {len(processed_df.columns)}")
    print(f"Total rows        : {len(processed_df)}")
    print(f"New feature cols  : {NEW_COLS}")
    print()
    print(processed_df[["Adj Close", "LogReturn", "RealizedVol", "DistEMA200", "ATR_norm"]].tail(3))
    print()
    print(processed_df[["DayOfWeek_sin", "DayOfWeek_cos", "Month_sin", "Month_cos"]].tail(3))

    # Sanity: all new cols present and non-null in last row
    for col in NEW_COLS:
        assert col in processed_df.columns, f"MISSING: {col}"
        assert not pd.isna(processed_df[col].iloc[-1]), f"NaN in last row: {col}"
    print("\nAll new feature sanity checks PASSED.")