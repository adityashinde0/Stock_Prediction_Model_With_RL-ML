"""
Price Trajectory Forecaster Module
Trains a multi-output ensemble regressor to project an N-step forward price trajectory
with upper and lower volatility confidence bounds.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PriceForecaster")


class PriceTrajectoryForecaster:
    """
    Multi-step forecasting engine that predicts forward price curves and confidence intervals.
    """

    def __init__(
        self,
        forecast_horizon: int = 14,
        model_path: Optional[Path] = None,
    ):
        self.forecast_horizon = forecast_horizon
        self.model_path = model_path or (MODELS_DIR / "price_forecaster.pkl")
        self.feature_cols = [
            "EMA7",
            "EMA21",
            "EMA50",
            "EMA200",
            "MACD_line",
            "MACD_signal",
            "MACD_diff",
            "RSI",
            "OBV",
            "BBH",
            "BBL",
            "BB_width",
            "Return",
        ]
        self.model = None

    def create_multi_step_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Creates N-step ahead cumulative relative return targets for each row.
        """
        data = df.copy()
        target_cols = []

        for step in range(1, self.forecast_horizon + 1):
            col_name = f"Target_Step_{step}"
            # Relative cumulative return: (Price_{t+step} - Price_t) / Price_t
            data[col_name] = (
                data["Adj Close"].shift(-step) - data["Adj Close"]
            ) / data["Adj Close"]
            target_cols.append(col_name)

        data = data.dropna().reset_index(drop=True)
        return data, target_cols

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains the MultiOutputRegressor on historical features and saves model weights.
        """
        logger.info(f"Preparing training data for {self.forecast_horizon}-step trajectory forecast...")
        data, target_cols = self.create_multi_step_targets(df)

        missing_cols = [c for c in self.feature_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        X = data[self.feature_cols]
        y = data[target_cols]

        logger.info("Training MultiOutput RandomForest Regressor...")
        base_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model = MultiOutputRegressor(base_regressor)
        self.model.fit(X, y)

        score = self.model.score(X, y)
        logger.info(f"Forecaster Training Complete | Training Multi-Output R^2 Score: {score:.4f}")

        self.save_model()
        return {"r2_score": round(score, 4), "horizon": self.forecast_horizon}

    def save_model(self):
        """
        Persists trained forecaster to disk.
        """
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "features": self.feature_cols,
                    "horizon": self.forecast_horizon,
                },
                f,
            )
        logger.info(f"Saved price forecaster to -> {self.model_path}")

    def load_model(self) -> bool:
        """
        Loads saved model weights from disk if available.
        """
        if not self.model_path.exists():
            logger.warning(f"No trained forecaster found at {self.model_path}.")
            return False

        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.feature_cols = data["features"]
                self.forecast_horizon = data["horizon"]
            logger.info("Price forecaster loaded successfully from disk.")
            return True
        except Exception as e:
            logger.error(f"Failed to load forecaster: {e}")
            return False

    def forecast_trajectory(
        self, df: pd.DataFrame, history_window: int = 60
    ) -> Dict[str, Any]:
        """
        Projects future price trajectory anchored to the last observed market price
        and includes upper/lower confidence bounds.
        """
        if self.model is None:
            if not self.load_model():
                logger.warning("No forecaster loaded. Automatically training on provided data...")
                self.train(df)

        latest_row = df.iloc[[-1]]
        latest_features = latest_row[self.feature_cols]
        last_observed_price = float(latest_row["Adj Close"].values[0])

        # Predict forward cumulative return percentages
        predicted_returns = self.model.predict(latest_features)[0]

        # Convert relative return trajectory back into actual dollar prices
        forecast_prices = [
            round(last_observed_price * (1.0 + float(ret)), 2)
            for ret in predicted_returns
        ]

        # Robust Date / Timestamp Extraction
        if isinstance(df.index, pd.DatetimeIndex):
            last_timestamp = df.index[-1]
            time_delta = df.index[-1] - df.index[-2] if len(df) > 1 else pd.Timedelta(days=1)
            hist_slice = df.tail(history_window)
            historical_dates = [d.strftime("%Y-%m-%d %H:%M") for d in hist_slice.index]
        elif "Date" in df.columns:
            date_col = pd.to_datetime(df["Date"])
            last_timestamp = date_col.iloc[-1]
            time_delta = date_col.iloc[-1] - date_col.iloc[-2] if len(df) > 1 else pd.Timedelta(days=1)
            hist_slice = df.tail(history_window)
            historical_dates = [d.strftime("%Y-%m-%d %H:%M") for d in pd.to_datetime(hist_slice["Date"])]
        else:
            last_timestamp = pd.Timestamp.now()
            time_delta = pd.Timedelta(days=1)
            historical_dates = [f"Step_{i}" for i in range(min(history_window, len(df)))]

        forecast_dates = [
            (last_timestamp + (time_delta * step)).strftime("%Y-%m-%d %H:%M")
            for step in range(1, self.forecast_horizon + 1)
        ]

        # Calculate dynamic volatility ribbon using recent Bollinger Band width
        recent_bb_width = float(latest_row["BB_width"].values[0]) / 100.0
        volatility_scale = max(0.015, recent_bb_width * 0.5)

        upper_bounds = [
            round(p * (1.0 + (volatility_scale * np.sqrt(i + 1))), 2)
            for i, p in enumerate(forecast_prices)
        ]
        lower_bounds = [
            round(p * (1.0 - (volatility_scale * np.sqrt(i + 1))), 2)
            for i, p in enumerate(forecast_prices)
        ]

        hist_slice = df.tail(history_window)
        historical_prices = [round(float(p), 2) for p in hist_slice["Adj Close"]]

        return {
            "last_observed_price": round(last_observed_price, 2),
            "historical_dates": historical_dates,
            "historical_prices": historical_prices,
            "forecast_dates": forecast_dates,
            "forecast_prices": forecast_prices,
            "upper_bounds": upper_bounds,
            "lower_bounds": lower_bounds,
            "horizon_steps": self.forecast_horizon,
        }


# --- Standalone Verification Block ---
if __name__ == "__main__":
    from src.data.fetcher import MarketDataFetcher
    from src.data.indicators import TechnicalIndicatorEngine

    logger.info("Fetching verification dataset for forecaster...")
    raw_df = MarketDataFetcher.fetch_data(ticker="BTC-USD", period="2y", interval="1d")
    processed_df = TechnicalIndicatorEngine.add_all_indicators(raw_df)

    forecaster = PriceTrajectoryForecaster(forecast_horizon=14)
    forecaster.train(processed_df)

    trajectory = forecaster.forecast_trajectory(processed_df)
    print("\n--- Price Trajectory Forecaster Verification Result ---")
    print(f"Last Actual Price  : ${trajectory['last_observed_price']}")
    print(f"Forecast Horizon   : {trajectory['horizon_steps']} periods ahead")
    print(f"Next 3 Dates Ahead : {trajectory['forecast_dates'][:3]}")
    print(f"Next 3 Prices Ahead: ${trajectory['forecast_prices'][:3]}")
    print(f"Upper Vol Band     : ${trajectory['upper_bounds'][:3]}")
    print(f"Lower Vol Band     : ${trajectory['lower_bounds'][:3]}")