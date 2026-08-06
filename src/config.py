"""
Global Configuration for Stock Vision Predictor
Centralized settings for Data, Indicators, Models, and Backtesting.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_CHARTS_DIR = DATA_DIR / "sample_charts"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Ticker & Market Defaults
DEFAULT_TICKER = "BTC-USD"
FALLBACK_TICKERS = ["BTC-USD", "ETH-USD", "AAPL", "NVDA", "TSLA", "^GSPC", "EURUSD=X"]
DEFAULT_TIMEFRAME = "1d"  # Valid yfinance intervals: 1m, 5m, 15m, 1h, 1d, 1wk

# Indicator Configuration
INDICATOR_PARAMS = {
    "ema_fast": 7,
    "ema_slow": 21,
    "ema_trend": 200,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_window": 20,
    "bollinger_std": 2,
}

# Model Signal Classes
SIGNAL_LABELS = {
    0: "STRONG SELL",
    1: "SELL",
    2: "NEUTRAL / HOLD",
    3: "BUY",
    4: "STRONG BUY"
}

# Backtesting Defaults
INITIAL_CAPITAL = 10000.0
TRANSACTION_FEE = 0.001  # 0.1% per trade