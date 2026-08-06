"""
Statistical Accuracy Evaluator Module
Calculates Directional Accuracy (DA), Mean Absolute Percentage Error (MAPE), and RMSE.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AccuracyEvaluator")


class AccuracyEvaluator:
    """
    Evaluates forecasting accuracy and directional correctness for time-series predictions.
    """

    @staticmethod
    def evaluate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates percentage of time the prediction correctly guesses UP vs. DOWN movement.
        """
        if len(y_true) < 2 or len(y_pred) < 2:
            return 0.0

        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)

        # True if both moved in the same direction (both positive or both negative)
        correct_directions = (true_diff * pred_diff) > 0
        directional_accuracy = float(np.mean(correct_directions)) * 100.0
        return round(directional_accuracy, 2)

    @staticmethod
    def evaluate_forecast(y_true: List[float], y_pred: List[float]) -> Dict[str, Any]:
        """
        Calculates RMSE, MAPE, and Directional Accuracy between true and predicted price curves.
        """
        true_arr = np.array(y_true, dtype=float)
        pred_arr = np.array(y_pred, dtype=float)

        if len(true_arr) == 0 or len(pred_arr) == 0 or len(true_arr) != len(pred_arr):
            logger.warning("Invalid arrays passed to evaluate_forecast.")
            return {"rmse": 0.0, "mape": 0.0, "directional_accuracy": 0.0}

        rmse = float(np.sqrt(mean_squared_error(true_arr, pred_arr)))
        mape = float(mean_absolute_percentage_error(true_arr, pred_arr)) * 100.0
        da = AccuracyEvaluator.evaluate_directional_accuracy(true_arr, pred_arr)

        return {
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "directional_accuracy": da,
        }


# --- Standalone Verification Block ---
if __name__ == "__main__":
    # Simulated true price path vs predicted price path
    mock_true = [100.0, 102.0, 101.0, 105.0, 108.0, 107.0]
    mock_pred = [100.0, 103.0, 100.5, 106.0, 109.0, 106.5]

    metrics = AccuracyEvaluator.evaluate_forecast(mock_true, mock_pred)
    print("\n--- Statistical Accuracy Evaluator Verification ---")
    print(f"Directional Accuracy : {metrics['directional_accuracy']}%")
    print(f"MAPE (Error %)       : {metrics['mape']}%")
    print(f"RMSE ($ Error)       : ${metrics['rmse']}")