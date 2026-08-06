# 📈 Stock Vision Predictor - Multi-Modal AI Trading Assistant

[![CI/CD Testing Pipeline](https://github.com/adityashinde0/Stock_Prediction_Model_With_RL-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/adityashinde0/Stock_Prediction_Model_With_RL-ML/actions)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio_4.0%2B-orange.svg)](https://gradio.app/)

An open-source, multi-modal machine learning system that bridges **Computer Vision** with **Quantitative Financial Forecasting**.

By dragging and dropping a candlestick chart screenshot from TradingView or MetaTrader, the system extracts chart metadata via GPU-accelerated OCR, synchronizes real-time institutional market data, engineers multi-period technical indicators, and executes supervised ensemble models to output discrete trading signals and a **14-period future price trajectory** with dynamic volatility confidence intervals.

---

# 🏗️ System Architecture & Dataflow

```text
[ User Uploads Screenshot ]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ 1. VISION & OCR SUBSYSTEM (`src/vision/ocr_engine.py`) │
│    - Preprocesses header crop via OpenCV               │
│    - Extracts Ticker (e.g., BTC-USD) & Timeframe       │
│    - Resilient Fallback Backend for OS DLL Blocker     │
└─────────────────────────┬──────────────────────────────┘
                          │ Extracted Ticker & Interval
                          ▼
┌────────────────────────────────────────────────────────┐
│ 2. INGESTION & FEATURES (`src/data/indicators.py`)     │
│    - Fetches live OHLCV history via `yfinance`         │
│    - Computes EMA7/21/50/200, MACD, RSI, OBV, BB       │
└─────────────────────────┬──────────────────────────────┘
                          │ 13-Dimensional Feature Vector
                          ▼
┌────────────────────────────────────────────────────────┐
│ 3. ML INFERENCE ENGINE (`src/models/`)                 │
│    - SignalClassifier: Quantile target classification  │
│      (Strong Buy / Buy / Hold / Sell / Strong Sell)    │
│    - PriceTrajectoryForecaster: MultiOutputRegressor   │
│      projections across t+1 to t+14 forward periods    │
└─────────────────────────┬──────────────────────────────┘
                          │ Signals, Forecasts, & Vol Bounds
                          ▼
┌────────────────────────────────────────────────────────┐
│ 4. GRADIO INTERACTIVE DASHBOARD (`app.py`)             │
│    - Displays color-coded Signal Badge & Confidence %  │
│    - Renders HTML5 Plotly chart with ±1.5σ Ribbon      │
│    - Evaluates historical Sharpe Ratio & Max Drawdown  │
└────────────────────────────────────────────────────────┘
```

---

# 🧮 Mathematical Formulation

## 1. Multi-Step Forward Return Targets

To generalize across assets with different price scales (approximately **$10 to $60,000**), the trajectory forecaster predicts cumulative relative returns rather than absolute dollar values.

### Forward Return

```math
R_{t+k} = \frac{P_{t+k} - P_t}{P_t}
```

Where:

- \(P_t\) = Closing price at time **t**
- \(k \in \{1,2,\dots,14\}\)

### Price Reconstruction

```math
\hat{P}_{t+k}=P_t\left(1+\hat{R}_{t+k}\right)
```

---

## 2. Dynamic Volatility Confidence Ribbon

The upper and lower confidence intervals dynamically scale according to the square root of time and the recent Bollinger Band width.

### Volatility Scale

```math
\sigma_{\text{scale}}
=
\max
\left(
0.015,
\frac{\text{BB}_{\text{width}}}{200}
\right)
```

### Upper Confidence Band

```math
\text{Upper Band}_i
=
\hat{P}_{t+i}
\left(
1+\sigma_{\text{scale}}\sqrt{i}
\right)
```

### Lower Confidence Band

```math
\text{Lower Band}_i
=
\hat{P}_{t+i}
\left(
1-\sigma_{\text{scale}}\sqrt{i}
\right)
```

---

# 🚀 Quickstart Installation & Deployment

## 1. Clone Repository

```bash
git clone https://github.com/YourUsername/Stock-Vision-Predictor.git
cd Stock-Vision-Predictor
```

---

## 2. Create Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install GPU-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
pip install -r requirements.txt
```

---

## 4. Launch the Application

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:7860
```

Upload a TradingView or MetaTrader screenshot and click **Analyze Chart & Forecast Trajectory**.

---

# 🧪 Run Automated Tests

```bash
pytest tests/ -v
```

---

# 📄 License

This project is released under the **MIT License**.

See the **LICENSE** file for complete details.

---

# ✅ Pre-Release Verification & Git Hygiene Checklist

Before pushing to GitHub, complete the following validation steps.

## 1. Dependency Lock Verification

- Ensure `requirements.txt` contains no local paths.
- Ensure package versions are pinned appropriately.
- Verify all `scikit-learn` imports use standard syntax.

---

## 2. Cold-Start Model Weight Handling

Model files (`*.pkl`) are excluded via `.gitignore`.

If pretrained models are unavailable:

- SignalClassifier automatically trains.
- PriceTrajectoryForecaster automatically trains.

Training occurs on first inference using live market data.

---

## 3. Local CI Verification

Run:

```powershell
pytest tests/ -v
```

Ensure all tests pass before publishing.

---

# 🚀 Git Initialization & First Commit

```bash
git init

git add .

git commit -m "feat: initial open-source release of Stock Vision Predictor multi-modal pipeline"

git branch -M main

git remote add origin https://github.com/YourUsername/Stock-Vision-Predictor.git

git push -u origin main
```
