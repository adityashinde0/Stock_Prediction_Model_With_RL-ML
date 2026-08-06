"""
Signal Classification Model
Trains a supervised ensemble classifier on technical indicator features to predict 5-class trading signals.
Outputs: Strong Sell (0), Sell (1), Neutral (2), Buy (3), Strong Buy (4) + Confidence Score.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split  # Corrected import

from src.config import MODELS_DIR, SIGNAL_LABELS

# Adaptive Backend: Attempt to import XGBoost
XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignalClassifier")


class SignalClassifier:
    """
    Supervised classification engine for predicting multi-class trading action signals.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (MODELS_DIR / "signal_classifier.pkl")
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

    def create_target_labels(self, df: pd.DataFrame, forward_periods: int = 5) -> pd.DataFrame:
        """
        Generates 5-class target labels based on forward N-period returns using quantile binning.
        """
        data = df.copy()
        # Calculate future return over forward_periods
        data["Forward_Return"] = data["Adj Close"].shift(-forward_periods) / data["Adj Close"] - 1.0
        data = data.dropna().reset_index(drop=True)

        # Quantile thresholds for 5 distinct trading actions
        q15 = data["Forward_Return"].quantile(0.15)
        q40 = data["Forward_Return"].quantile(0.40)
        q60 = data["Forward_Return"].quantile(0.60)
        q85 = data["Forward_Return"].quantile(0.85)

        def classify_return(ret: float) -> int:
            if ret <= q15:
                return 0  # STRONG SELL
            elif ret <= q40:
                return 1  # SELL
            elif ret <= q60:
                return 2  # NEUTRAL / HOLD
            elif ret <= q85:
                return 3  # BUY
            else:
                return 4  # STRONG BUY

        data["Target"] = data["Forward_Return"].apply(classify_return)
        return data

    def train(self, df: pd.DataFrame, forward_periods: int = 5) -> Dict[str, Any]:
        """
        Trains the classifier on historical feature vectors and saves model weights.
        """
        logger.info("Generating target labels and preparing training dataset...")
        data = self.create_target_labels(df, forward_periods=forward_periods)

        # Validate feature presence
        missing_cols = [c for c in self.feature_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        X = data[self.feature_cols]
        y = data["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Keep time-series order
        )

        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost Classifier...")
            self.model = XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                eval_metric="mlogloss",
            )
        else:
            logger.info("XGBoost unavailable. Training RandomForest Classifier...")
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
            )

        self.model.fit(X_train, y_train)

        # Evaluate on out-of-sample test split
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        logger.info(f"Model Training Complete | Out-of-Sample Accuracy: {acc:.2%}")

        # Save weights to disk
        self.save_model()

        return {"accuracy": acc, "classification_report": report}

    def save_model(self):
        """
        Persists trained classifier to disk.
        """
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "features": self.feature_cols}, f)
        logger.info(f"Saved signal classifier to -> {self.model_path}")

    def load_model(self) -> bool:
        """
        Loads saved model weights from disk if available.
        """
        if not self.model_path.exists():
            logger.warning(f"No trained model found at {self.model_path}.")
            return False

        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.feature_cols = data["features"]
            logger.info("Model loaded successfully from disk.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predicts the trading signal and confidence score for the latest market timestamp.
        """
        if self.model is None:
            if not self.load_model():
                logger.warning("No model loaded. Automatically training on provided data...")
                self.train(df)

        latest_features = df[self.feature_cols].iloc[[-1]]
        
        # Predict class and probability distribution
        pred_class = int(self.model.predict(latest_features)[0])
        probs = self.model.predict_proba(latest_features)[0]
        confidence = float(np.max(probs)) * 100.0

        signal_label = SIGNAL_LABELS.get(pred_class, "NEUTRAL / HOLD")

        return {
            "signal_class": pred_class,
            "signal_label": signal_label,
            "confidence_score": round(confidence, 2),
            "class_probabilities": {
                SIGNAL_LABELS[k]: round(float(v) * 100.0, 2) for k, v in enumerate(probs)
            },
        }


# --- Standalone Verification Block ---
if __name__ == "__main__":
    from src.data.fetcher import MarketDataFetcher
    from src.data.indicators import TechnicalIndicatorEngine

    logger.info("Fetching verification dataset...")
    raw_df = MarketDataFetcher.fetch_data(ticker="BTC-USD", period="2y", interval="1d")
    processed_df = TechnicalIndicatorEngine.add_all_indicators(raw_df)

    classifier = SignalClassifier()
    train_metrics = classifier.train(processed_df)

    latest_signal = classifier.predict_signal(processed_df)
    print("\n--- Signal Classifier Verification Result ---")
    print(f"Predicted Signal   : {latest_signal['signal_label']}")
    print(f"Confidence Score   : {latest_signal['confidence_score']}%")
    print(f"Class Probabilities: {latest_signal['class_probabilities']}")