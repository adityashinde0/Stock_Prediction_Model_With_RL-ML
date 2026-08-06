"""
Unit tests for SignalClassifier and PriceTrajectoryForecaster inference engines.
"""

import pandas as pd
import pytest
from src.models.forecaster import PriceTrajectoryForecaster
from src.models.signal_classifier import SignalClassifier


@pytest.fixture
def trained_pipeline_data(mock_ohlcv_dataframe):
    """Returns engineered feature data for testing inference models."""
    from src.data.indicators import TechnicalIndicatorEngine
    return TechnicalIndicatorEngine.add_all_indicators(mock_ohlcv_dataframe)


def test_signal_classifier_probability_bounds(trained_pipeline_data):
    """Verifies that predicted class probabilities are normalized between 0% and 100%."""
    classifier = SignalClassifier()
    classifier.train(trained_pipeline_data)
    result = classifier.predict_signal(trained_pipeline_data)
    
    assert 0 <= result["signal_class"] <= 4
    assert 0.0 <= result["confidence_score"] <= 100.0
    assert "BUY" in result["signal_label"] or "SELL" in result["signal_label"] or "HOLD" in result["signal_label"]


def test_trajectory_forecaster_horizon_integrity(trained_pipeline_data):
    """Ensures multi-output trajectory dates and prices match the requested horizon steps."""
    horizon = 14
    forecaster = PriceTrajectoryForecaster(forecast_horizon=horizon)
    forecaster.train(trained_pipeline_data)
    forecast = forecaster.forecast_trajectory(trained_pipeline_data)
    
    assert len(forecast["forecast_dates"]) == horizon
    assert len(forecast["forecast_prices"]) == horizon
    assert len(forecast["upper_bounds"]) == horizon
    assert len(forecast["lower_bounds"]) == horizon
    # Verify upper volatility band is strictly greater than or equal to the forecast price
    assert all(u >= p for u, p in zip(forecast["upper_bounds"], forecast["forecast_prices"]))