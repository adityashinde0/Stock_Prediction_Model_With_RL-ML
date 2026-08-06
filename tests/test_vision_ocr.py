"""
Unit tests for the ChartOCREngine and Adaptive Fallback Backend.
"""

from unittest.mock import patch
import numpy as np
import pytest
from src.vision.ocr_engine import ChartOCREngine


@pytest.fixture
def ocr_engine_cpu():
    """Returns a CPU-only instance of ChartOCREngine for deterministic CI testing."""
    return ChartOCREngine(languages=["en"], gpu=False)


def test_ticker_regex_parsing(ocr_engine_cpu):
    """Verifies symbol normalization across standard stock, crypto, and forex formats."""
    test_cases = [
        (["BTCUSD", "1D", "BITSTAMP", "64491.00"], "BTC-USD"),
        (["AAPL", "1H", "NASDAQ", "210.50"], "AAPL"),
        (["EURUSD", "15M", "FXCM", "1.0850"], "EURUSD=X"),
    ]
    for raw_tokens, expected_ticker in test_cases:
        ticker, conf = ocr_engine_cpu.parse_ticker(raw_tokens)
        assert ticker == expected_ticker
        assert conf >= 0.75


def test_price_extraction_with_and_without_commas(ocr_engine_cpu):
    """Ensures numeric price parsing handles standard and comma-separated floating-point strings."""
    tokens_comma = ["BITCOIN", "64,491.50", "VOL", "100M"]
    tokens_raw = ["BITCOIN", "64491.50", "VOL", "100M"]
    
    price_comma, _ = ocr_engine_cpu.parse_price(tokens_comma)
    price_raw, _ = ocr_engine_cpu.parse_price(tokens_raw)
    
    assert price_comma == 64491.50
    assert price_raw == 64491.50


def test_fallback_engine_continuity(ocr_engine_cpu):
    """Verifies pipeline continuity when deep learning OCR fails or is blocked by OS security."""
    mock_image = np.ones((100, 500, 3), dtype=np.uint8) * 255
    
    with patch.object(ocr_engine_cpu, "use_easyocr", False):
        metadata = ocr_engine_cpu.extract_metadata(mock_image)
        assert metadata["ticker"] == "BTC-USD"
        assert metadata["timeframe"] == "1d"
        assert isinstance(metadata["raw_text"], list)