"""
Price Trajectory Forecaster Module
Adaptive deep learning forecaster using a Bidirectional LSTM with
Temporal Self-Attention and Monte Carlo Dropout for uncertainty estimation.
Falls back gracefully to a scikit-learn MultiOutputRegressor ensemble
if PyTorch training fails or is unavailable on the host machine.

Input feature set (21 features — must match TechnicalIndicatorEngine output):
  Trend      : EMA7, EMA21, EMA50, EMA200
  Momentum   : MACD_line, MACD_signal, MACD_diff, RSI
  Volume     : OBV
  Volatility : BBH, BBL, BB_width, RealizedVol, ATR_norm
  Return     : Return, LogReturn
  Regime     : DistEMA200
  Seasonality: DayOfWeek_sin, DayOfWeek_cos, Month_sin, Month_cos
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR

# ── PyTorch Adaptive Import ──────────────────────────────────────────────────
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PriceForecaster")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — PyTorch Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class _BiLSTMAttentionForecaster(nn.Module):
    """
    Bidirectional LSTM with a lightweight Temporal Self-Attention layer.

    Architecture (forward pass):
        Input   : (batch, seq_len, n_features)
        BiLSTM  : hidden_size=128, num_layers=2, dropout=0.3
                  → (batch, seq_len, 256)
        Attention: Linear(256→1) + softmax over time dimension
                  → context vector (batch, 256)
        Head    : Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→horizon)
        Output  : (batch, horizon)  — cumulative relative returns
    """

    def __init__(
        self,
        n_features: int = 13,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        forecast_horizon: int = 14,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            # Dropout between LSTM layers (only valid when num_layers > 1)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_size = hidden_size * 2  # bidirectional doubles feature dim

        # Temporal self-attention: learn one scalar score per timestep
        self.attention = nn.Linear(lstm_out_size, 1)

        # Prediction head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_out_size, 128)
        self.fc2 = nn.Linear(128, forecast_horizon)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)                      # (batch, seq_len, 256)

        # Attention: produce a context vector weighted over the time axis
        attn_scores = self.attention(lstm_out)           # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)     # (batch, seq_len, 1)
        context = (lstm_out * attn_weights).sum(dim=1)   # (batch, 256)

        # Head
        out = F.relu(self.fc1(context))                  # (batch, 128)
        out = self.dropout(out)
        return self.fc2(out)                             # (batch, horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Main Forecaster Class  (public API identical to original)
# ─────────────────────────────────────────────────────────────────────────────

class PriceTrajectoryForecaster:
    """
    Adaptive multi-step forecasting engine.

    Primary backend : PyTorch BiLSTM + Temporal Self-Attention + MC Dropout.
    Fallback backend: scikit-learn MultiOutputRegressor (RandomForest).

    The public API surface (class name, __init__ params, all public method
    signatures, and the forecast_trajectory() output dict schema) is identical
    to the original implementation — app.py requires zero changes.
    """

    # ── Training hyper-parameters ──────────────────────────────────────────
    SEQ_LEN      = 60     # input context window (timesteps)
    BATCH_SIZE   = 32
    MAX_EPOCHS   = 150
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 15     # early-stopping patience (validation epochs)
    MC_PASSES    = 20     # Monte Carlo dropout forward passes for uncertainty
    SIGMA_SCALE  = 1.5    # confidence band width = mean ± SIGMA_SCALE × std

    def __init__(
        self,
        forecast_horizon: int = 14,
        model_path: Optional[Path] = None,
    ):
        self.forecast_horizon = forecast_horizon
        # DL checkpoint (.pth) — primary save target
        self.model_path = model_path or (MODELS_DIR / "dl_forecaster.pth")
        # Legacy sklearn pickle (.pkl) — fallback load path
        self.sklearn_path = MODELS_DIR / "price_forecaster.pkl"

        self.feature_cols: List[str] = [
            # Trend
            "EMA7", "EMA21", "EMA50", "EMA200",
            # Momentum
            "MACD_line", "MACD_signal", "MACD_diff", "RSI",
            # Volume
            "OBV",
            # Volatility
            "BBH", "BBL", "BB_width", "RealizedVol", "ATR_norm",
            # Return
            "Return", "LogReturn",
            # Regime
            "DistEMA200",
            # Seasonality (cyclical sine/cosine encoding)
            "DayOfWeek_sin", "DayOfWeek_cos", "Month_sin", "Month_cos",
        ]

        # Populated at train / load time
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self._backend: str = "none"         # "dl" | "sklearn" | "none"
        self._device: Optional[Any] = None  # torch.device when DL active

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_device(self) -> "torch.device":  # noqa: F821
        """Auto-selects CUDA → MPS → CPU."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2 — Sequence Builder
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sequences(
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a flat (N, n_features) feature matrix into overlapping
        (N-seq_len, seq_len, n_features) input sequences with aligned
        (N-seq_len, horizon) targets indexed at the end of each window.
        """
        X_seqs, y_seqs = [], []
        for i in range(seq_len, len(X)):
            X_seqs.append(X[i - seq_len : i])
            y_seqs.append(y[i - 1])   # target aligned to window's final step
        return (
            np.array(X_seqs, dtype=np.float32),
            np.array(y_seqs, dtype=np.float32),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3 — Deep Learning Training Loop
    # ─────────────────────────────────────────────────────────────────────────

    def _train_dl_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains the BiLSTM+Attention model with:
          - AdamW optimizer (lr=1e-3, weight_decay=1e-4)
          - HuberLoss (delta=0.02) — robust to return outlier spikes
          - CosineAnnealingLR scheduler (T_max=50, eta_min=1e-5)
          - Early stopping (patience=15 validation epochs)
          - Gradient clipping (max_norm=1.0)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available.")

        self._device = self._get_device()
        logger.info(f"Training DL Forecaster on device: {self._device}")

        # ── Build multi-step targets ──
        data, target_cols = self.create_multi_step_targets(df)
        missing = [c for c in self.feature_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X_raw = data[self.feature_cols].values.astype(np.float32)
        y_raw = data[target_cols].values.astype(np.float32)

        # ── Fit StandardScaler, build sequences ──
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        X_seqs, y_seqs = self._build_sequences(X_scaled, y_raw, self.SEQ_LEN)
        n_total = len(X_seqs)
        if n_total < 10:
            raise ValueError(
                f"Too few training sequences ({n_total}). Need ≥ 10. "
                "Fetch more historical data or reduce SEQ_LEN."
            )

        # ── Chronological 80/20 split (no shuffle — preserve time order) ──
        split = int(n_total * 0.8)
        X_train, X_val = X_seqs[:split], X_seqs[split:]
        y_train, y_val = y_seqs[:split], y_seqs[split:]

        train_ds = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=False
        )
        X_val_t = torch.from_numpy(X_val).to(self._device)
        y_val_t = torch.from_numpy(y_val).to(self._device)

        # ── Instantiate model ──
        n_features = X_seqs.shape[2]
        net = _BiLSTMAttentionForecaster(
            n_features=n_features,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            forecast_horizon=self.forecast_horizon,
        ).to(self._device)

        optimizer  = AdamW(net.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        scheduler  = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        criterion  = nn.HuberLoss(delta=0.02)

        best_val_loss: float = float("inf")
        best_state: Optional[Dict] = None
        patience_counter: int = 0

        logger.info(
            f"  Sequences: {n_total} total | Train: {len(X_train)} | Val: {len(X_val)} | "
            f"Features: {n_features} | Horizon: {self.forecast_horizon}"
        )

        # ── Training loop ──
        for epoch in range(1, self.MAX_EPOCHS + 1):
            net.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                pred = net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            # ── Validation ──
            net.eval()
            with torch.no_grad():
                val_pred = net(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Deep copy weights to CPU for safe restoration
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 25 == 0 or epoch == 1:
                avg_train = epoch_loss / max(len(train_loader), 1)
                logger.info(
                    f"  Epoch {epoch:3d}/{self.MAX_EPOCHS} | "
                    f"Train: {avg_train:.6f} | Val: {val_loss:.6f} | "
                    f"Best: {best_val_loss:.6f} | Patience: {patience_counter}/{self.PATIENCE}"
                )

            if patience_counter >= self.PATIENCE:
                logger.info(
                    f"  Early stopping triggered at epoch {epoch} "
                    f"(patience={self.PATIENCE}, best val={best_val_loss:.6f})."
                )
                break

        # ── Restore best checkpoint ──
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

        self.model   = net
        self._backend = "dl"
        logger.info(f"DL Training complete. Best validation loss: {best_val_loss:.6f}")
        return {"backend": "dl", "best_val_loss": round(best_val_loss, 6)}

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4 — Monte Carlo Dropout Inference
    # ─────────────────────────────────────────────────────────────────────────

    def _mc_dropout_predict(
        self,
        x_seq: "torch.Tensor",  # noqa: F821
        n_passes: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs n_passes stochastic forward passes with dropout enabled
        (model.train() temporarily activates dropout at inference time).

        Returns
        -------
        mean_returns  : (horizon,) mean predicted cumulative relative returns
        upper_returns : mean + SIGMA_SCALE × std
        lower_returns : mean − SIGMA_SCALE × std
        """
        assert TORCH_AVAILABLE and self.model is not None
        x_seq = x_seq.to(self._device)

        preds = []
        # Enable dropout by switching to train mode temporarily
        self.model.train()
        with torch.no_grad():
            for _ in range(n_passes):
                out = self.model(x_seq)          # (1, horizon)
                preds.append(out.cpu().numpy())
        self.model.eval()

        stacked = np.stack(preds, axis=0)        # (n_passes, 1, horizon)
        mean_r  = stacked.mean(axis=0)[0]        # (horizon,)
        std_r   = stacked.std(axis=0)[0]         # (horizon,)

        upper_r = mean_r + self.SIGMA_SCALE * std_r
        lower_r = mean_r - self.SIGMA_SCALE * std_r
        return mean_r, upper_r, lower_r

    # ─────────────────────────────────────────────────────────────────────────
    # Sklearn Fallback Training (Phase 5 — internal)
    # ─────────────────────────────────────────────────────────────────────────

    def _train_sklearn_fallback(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains the legacy MultiOutputRegressor (RandomForest) and persists
        it to the legacy .pkl path. Used when DL training fails or PyTorch
        is unavailable.
        """
        logger.info("Training sklearn MultiOutputRegressor fallback...")
        data, target_cols = self.create_multi_step_targets(df)
        X = data[self.feature_cols]
        y = data[target_cols]

        base_reg = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        sklearn_model = MultiOutputRegressor(base_reg)
        sklearn_model.fit(X, y)
        score = sklearn_model.score(X, y)

        self.model    = sklearn_model
        self._backend = "sklearn"
        logger.info(f"sklearn fallback training complete. Training R²={score:.4f}")

        # Persist immediately
        with open(self.sklearn_path, "wb") as f:
            pickle.dump(
                {
                    "model": sklearn_model,
                    "features": self.feature_cols,
                    "horizon": self.forecast_horizon,
                },
                f,
            )
        logger.info(f"sklearn fallback saved → {self.sklearn_path}")
        return {"backend": "sklearn", "r2_score": round(score, 4)}

    # ─────────────────────────────────────────────────────────────────────────
    # Private inference helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_dl(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds a SEQ_LEN-step input sequence from the tail of df,
        normalises with the fitted scaler, and runs MC Dropout prediction.
        Pads with zeros on the left if fewer than SEQ_LEN rows exist.
        """
        assert self.scaler is not None, "Scaler must be fitted before DL inference."

        n   = len(df)
        win = min(self.SEQ_LEN, n)
        X_raw    = df[self.feature_cols].values[-win:].astype(np.float32)
        X_scaled = self.scaler.transform(X_raw)

        # Zero-pad on the left if history is shorter than the context window
        if win < self.SEQ_LEN:
            pad      = np.zeros((self.SEQ_LEN - win, X_scaled.shape[1]), dtype=np.float32)
            X_scaled = np.vstack([pad, X_scaled])

        # Shape: (1, SEQ_LEN, n_features)
        x_seq = torch.from_numpy(X_scaled[np.newaxis, :, :])
        return self._mc_dropout_predict(x_seq, n_passes=self.MC_PASSES)

    def _predict_sklearn(self, latest_row: pd.DataFrame) -> np.ndarray:
        """Single-row sklearn prediction → 1D return array (length = horizon)."""
        return self.model.predict(latest_row[self.feature_cols])[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — unchanged signatures
    # ─────────────────────────────────────────────────────────────────────────

    def create_multi_step_targets(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Creates N-step ahead cumulative relative return targets for each row.
        (Unchanged from original implementation.)
        """
        data = df.copy()
        target_cols = []
        for step in range(1, self.forecast_horizon + 1):
            col_name = f"Target_Step_{step}"
            data[col_name] = (
                data["Adj Close"].shift(-step) - data["Adj Close"]
            ) / data["Adj Close"]
            target_cols.append(col_name)
        data = data.dropna().reset_index(drop=True)
        return data, target_cols

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Entry point for training. Attempts DL (BiLSTM+Attention) first;
        on any failure gracefully falls back to sklearn MultiOutputRegressor.
        """
        logger.info(
            f"Starting adaptive forecaster training "
            f"(horizon={self.forecast_horizon}, PyTorch={TORCH_AVAILABLE})..."
        )
        if TORCH_AVAILABLE:
            try:
                metrics = self._train_dl_model(df)
                self.save_model()
                return metrics
            except Exception as exc:
                logger.warning(
                    f"DL training failed: {exc}  →  switching to sklearn fallback."
                )

        return self._train_sklearn_fallback(df)

    def save_model(self):
        """
        Persists the DL model state dict + fitted scaler to disk as a .pth file.
        (sklearn fallback persistence is handled inside _train_sklearn_fallback.)
        """
        if self._backend == "dl" and TORCH_AVAILABLE and self.model is not None:
            payload = {
                "backend": "dl",
                "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                "scaler":     self.scaler,
                "features":   self.feature_cols,
                "horizon":    self.forecast_horizon,
                "n_features": len(self.feature_cols),
                "model_config": {
                    "hidden_size": 128,
                    "num_layers":  2,
                    "dropout":     0.3,
                },
            }
            torch.save(payload, self.model_path)
            logger.info(f"DL forecaster saved → {self.model_path}")
        else:
            logger.info("save_model: sklearn model already persisted.")

    def load_model(self) -> bool:
        """
        Load priority:
          1. DL checkpoint (.pth) — BiLSTM+Attention
          2. Legacy sklearn pickle (.pkl) — RandomForest fallback
        Returns True on success, False if nothing found.
        """
        # ── 1. Try DL .pth ─────────────────────────────────────────────────
        if self.model_path.exists() and TORCH_AVAILABLE:
            try:
                # weights_only=False required because scaler (non-tensor) is embedded
                payload = torch.load(
                    self.model_path, map_location="cpu", weights_only=False
                )
                if payload.get("backend") == "dl":
                    cfg = payload["model_config"]
                    net = _BiLSTMAttentionForecaster(
                        n_features=payload["n_features"],
                        hidden_size=cfg["hidden_size"],
                        num_layers=cfg["num_layers"],
                        dropout=cfg["dropout"],
                        forecast_horizon=payload["horizon"],
                    )
                    net.load_state_dict(payload["state_dict"])
                    net.eval()

                    self._device       = self._get_device()
                    self.model         = net.to(self._device)
                    self.scaler        = payload["scaler"]
                    self.feature_cols  = payload["features"]
                    self.forecast_horizon = payload["horizon"]
                    self._backend      = "dl"
                    logger.info("DL forecaster (BiLSTM+Attention) loaded successfully.")
                    return True
            except Exception as exc:
                logger.warning(f"DL model load failed ({exc}). Trying sklearn fallback...")

        # ── 2. Try legacy sklearn .pkl ──────────────────────────────────────
        if self.sklearn_path.exists():
            try:
                with open(self.sklearn_path, "rb") as f:
                    data = pickle.load(f)
                self.model            = data["model"]
                self.feature_cols     = data["features"]
                self.forecast_horizon = data["horizon"]
                self._backend         = "sklearn"
                logger.info("sklearn fallback forecaster loaded from disk.")
                return True
            except Exception as exc:
                logger.error(f"sklearn fallback load failed: {exc}")

        logger.warning("No trained forecaster found on disk.")
        return False

    def forecast_trajectory(
        self, df: pd.DataFrame, history_window: int = 60
    ) -> Dict[str, Any]:
        """
        Projects the future price trajectory anchored to the last observed price.

        Output dict schema (identical to original — zero breaking changes):
        {
            "last_observed_price" : float,
            "historical_dates"    : List[str],
            "historical_prices"   : List[float],
            "forecast_dates"      : List[str],
            "forecast_prices"     : List[float],
            "upper_bounds"        : List[float],
            "lower_bounds"        : List[float],
            "horizon_steps"       : int,
        }

        DL path   → MC Dropout 1.5σ empirical bounds (20 passes).
        sklearn path → legacy Bollinger Band width approximation.
        """
        # ── Ensure model is ready ──────────────────────────────────────────
        if self.model is None:
            if not self.load_model():
                logger.warning("No forecaster loaded — training on provided data...")
                self.train(df)

        latest_row           = df.iloc[[-1]]
        last_observed_price  = float(latest_row["Adj Close"].values[0])

        # ── Route inference ────────────────────────────────────────────────
        if self._backend == "dl" and TORCH_AVAILABLE:
            mean_returns, upper_returns, lower_returns = self._predict_dl(df)
        else:
            mean_returns = self._predict_sklearn(latest_row)
            # Legacy BB-width volatility approximation for sklearn path
            recent_bb_width  = float(latest_row["BB_width"].values[0]) / 100.0
            volatility_scale = max(0.015, recent_bb_width * 0.5)
            upper_returns    = np.array([
                r + volatility_scale * np.sqrt(i + 1)
                for i, r in enumerate(mean_returns)
            ])
            lower_returns    = np.array([
                r - volatility_scale * np.sqrt(i + 1)
                for i, r in enumerate(mean_returns)
            ])

        # ── Convert relative returns → dollar prices ───────────────────────
        forecast_prices = [
            round(last_observed_price * (1.0 + float(r)), 2) for r in mean_returns
        ]
        upper_bounds = [
            round(last_observed_price * (1.0 + float(r)), 2) for r in upper_returns
        ]
        lower_bounds = [
            round(last_observed_price * (1.0 + float(r)), 2) for r in lower_returns
        ]

        # ── Date / timestamp extraction (unchanged logic) ──────────────────
        if isinstance(df.index, pd.DatetimeIndex):
            last_timestamp  = df.index[-1]
            time_delta      = df.index[-1] - df.index[-2] if len(df) > 1 else pd.Timedelta(days=1)
            hist_slice      = df.tail(history_window)
            historical_dates = [d.strftime("%Y-%m-%d %H:%M") for d in hist_slice.index]
        elif "Date" in df.columns:
            date_col         = pd.to_datetime(df["Date"])
            last_timestamp   = date_col.iloc[-1]
            time_delta       = (
                date_col.iloc[-1] - date_col.iloc[-2] if len(df) > 1 else pd.Timedelta(days=1)
            )
            hist_slice       = df.tail(history_window)
            historical_dates = [
                d.strftime("%Y-%m-%d %H:%M") for d in pd.to_datetime(hist_slice["Date"])
            ]
        else:
            last_timestamp  = pd.Timestamp.now()
            time_delta      = pd.Timedelta(days=1)
            hist_len        = min(history_window, len(df))
            historical_dates = [
                (last_timestamp - (time_delta * (hist_len - 1 - i))).strftime("%Y-%m-%d %H:%M")
                for i in range(hist_len)
            ]

        forecast_dates = [
            (last_timestamp + (time_delta * step)).strftime("%Y-%m-%d %H:%M")
            for step in range(1, self.forecast_horizon + 1)
        ]

        hist_slice       = df.tail(history_window)
        historical_prices = [round(float(p), 2) for p in hist_slice["Adj Close"]]

        return {
            "last_observed_price": round(last_observed_price, 2),
            "historical_dates":    historical_dates,
            "historical_prices":   historical_prices,
            "forecast_dates":      forecast_dates,
            "forecast_prices":     forecast_prices,
            "upper_bounds":        upper_bounds,
            "lower_bounds":        lower_bounds,
            "horizon_steps":       self.forecast_horizon,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Verification Block
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.fetcher import MarketDataFetcher
    from src.data.indicators import TechnicalIndicatorEngine

    logger.info("Fetching BTC-USD 2y/1d dataset for DL forecaster verification...")
    raw_df       = MarketDataFetcher.fetch_data(ticker="BTC-USD", period="2y", interval="1d")
    processed_df = TechnicalIndicatorEngine.add_all_indicators(raw_df)

    fc      = PriceTrajectoryForecaster(forecast_horizon=14)
    metrics = fc.train(processed_df)
    logger.info(f"Training metrics: {metrics}")

    traj = fc.forecast_trajectory(processed_df)

    print("\n--- DL Price Trajectory Forecaster Verification ---")
    print(f"Backend            : {fc._backend}")
    print(f"Last Actual Price  : ${traj['last_observed_price']:,.2f}")
    print(f"Forecast Horizon   : {traj['horizon_steps']} periods ahead")
    print(f"Next 3 Dates Ahead : {traj['forecast_dates'][:3]}")
    print(f"Next 3 Prices     : {traj['forecast_prices'][:3]}")
    print(f"Upper Band (1.5σ)  : {traj['upper_bounds'][:3]}")
    print(f"Lower Band (1.5σ)  : {traj['lower_bounds'][:3]}")

    # Sanity checks
    assert len(traj["forecast_prices"]) == 14,  "FAIL: forecast_prices must have 14 elements"
    assert len(traj["upper_bounds"])    == 14,  "FAIL: upper_bounds must have 14 elements"
    assert len(traj["lower_bounds"])    == 14,  "FAIL: lower_bounds must have 14 elements"
    assert all(
        traj["upper_bounds"][i] >= traj["forecast_prices"][i] for i in range(14)
    ), "FAIL: upper_bounds must be >= forecast_prices"
    assert all(
        traj["lower_bounds"][i] <= traj["forecast_prices"][i] for i in range(14)
    ), "FAIL: lower_bounds must be <= forecast_prices"
    print("\nAll sanity checks PASSED.")