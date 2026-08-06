"""
Unit tests for the TechnicalIndicatorEngine feature pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.data.indicators import TechnicalIndicatorEngine


@pytest.fixture
def mock_ohlcv_dataframe():
    """Generates a synthetic 300-period OHLCV dataset."""
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=300, freq="D")
    base_price = 100.0 + np.cumsum(np.random.normal(0, 1, 300))
    
    df = pd.DataFrame({
        "Open": base_price - 0.5,
        "High": base_price + 1.5,
        "Low": base_price - 1.5,
        "Close": base_price + 0.5,
        "Adj Close": base_price + 0.5,
        "Volume": np.random.randint(100000, 5000000, 300)
    }, index=dates)
    return df


def test_indicator_engineering_output_shape(mock_ohlcv_dataframe):
    """Ensures all required technical indicators are generated and NaNs are removed."""
    processed = TechnicalIndicatorEngine.add_all_indicators(mock_ohlcv_dataframe)
    
    expected_columns = [
        "EMA7", "EMA21", "EMA50", "EMA200",
        "MACD_line", "MACD_signal", "MACD_diff",
        "RSI", "OBV", "BBH", "BBL", "BB_width", "Return"
    ]
    for col in expected_columns:
        assert col in processed.columns
        
    # EMA200 consumes the first 199 rows; ensure remaining rows are complete
    assert len(processed) <= len(mock_ohlcv_dataframe) - 199
    assert processed.isna().sum().sum() == 0