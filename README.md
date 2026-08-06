---
title: Stock Prediction Model With RL-ML
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

<div align="center">
  <h1>📈 Stock Prediction Model with RL & ML</h1>
  <p>
    <b>An open-source, multi-modal machine learning system bridging Computer Vision with Quantitative Financial Forecasting.</b>
  </p>

[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio_4.0%2B-orange.svg)](https://gradio.app/)
[![CI](https://github.com/adityashinde0/Stock_Prediction_Model_With_RL-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/adityashinde0/Stock_Prediction_Model_With_RL-ML/actions/workflows/ci.yml)

</div>

---

## 🚀 Overview

This repository provides an institutional-grade quantitative forecasting and analysis dashboard. By simply uploading a candlestick chart screenshot from TradingView or MetaTrader, the system:
1. **Extracts Metadata** via GPU-accelerated OCR (identifying Ticker and Timeframe).
2. **Synchronizes Market Data** in real-time via `yfinance`.
3. **Engineers 21 Technical Features** including Log Returns, Realized Volatility, ATR, EMA/MACD, and Cyclical Seasonality.
4. **Predicts 5-Class Trading Signals** (Strong Sell to Strong Buy) using a Random Forest ensemble.
5. **Forecasts 14-Period Price Trajectories** using a PyTorch Bidirectional LSTM with Temporal Self-Attention and Monte Carlo Dropout for uncertainty estimation.
6. **Renders a Bloomberg-style Terminal UI** via Gradio & Plotly, complete with candlestick charts, RSI subplots, Stop-Loss/Take-Profit overlays, and risk management metrics.

---

## 🧠 Core Architecture

### 1. Vision & OCR Subsystem (`src/vision/ocr_engine.py`)
- Uses `EasyOCR` and `OpenCV` to preprocess the chart header.
- Automatically extracts the Ticker symbol (e.g., `BTC-USD`) and Timeframe (e.g., `1d`, `1h`).
- Includes a robust sanity checker to warn against massive OCR price divergences.

### 2. Quantitative Feature Engineering (`src/data/indicators.py`)
Transforms raw OHLCV market data into rich, 21-dimensional feature vectors:
- **Trend**: EMA (7, 21, 50, 200), Distance from EMA200
- **Momentum**: MACD (Line, Signal, Diff), RSI (14)
- **Volatility**: Bollinger Bands, Normalized ATR, 20-Day Annualized Realized Volatility
- **Returns**: Simple Returns, Log Returns ($\ln(P_t / P_{t-1})$)
- **Seasonality**: Sine/Cosine Cyclical Encoding of Day-of-Week and Month.

### 3. Deep Learning Inference Engine (`src/models/`)
- **SignalClassifier**: Quantile-based target classification (Strong Buy / Buy / Hold / Sell / Strong Sell) using Scikit-Learn `RandomForestClassifier` / `XGBoost`.
- **PriceTrajectoryForecaster**: An advanced PyTorch `nn.Module` featuring a **Bidirectional LSTM with Temporal Self-Attention**.
  - Includes **Monte Carlo Dropout** inference (running 20 stochastic forward passes) to empirically derive $1.5\sigma$ confidence upper and lower bounds.
  - Automatically falls back to a Scikit-Learn `MultiOutputRegressor` if PyTorch execution fails.

### 4. Interactive Dashboard (`app.py`)
- Built using **Gradio Blocks** for a clean, responsive web interface.
- Dynamic **Plotly** integration rendering professional `go.Candlestick` charts, Probability Fan forecasts (using `tonexty` fills), and dual-subplot RSI layouts.
- Auto-calculates Risk/Reward ratios based on derived Stop-Loss and Take-Profit bounds.

---

## 🧮 Mathematical Formulation

### Deep Learning Uncertainty (MC Dropout)
Instead of relying on standard deviation approximations, the $1.5\sigma$ confidence bands are derived organically from the neural network's epistemic uncertainty during inference:
```python
# 20 stochastic forward passes with active dropout
mean_returns = mean_over_passes(model(x))
std_returns  = std_over_passes(model(x))

upper_bounds = mean_returns + 1.5 * std_returns
lower_bounds = mean_returns - 1.5 * std_returns
```

### Institutional Feature: Annualized Realized Volatility
Calculates the rolling 20-day standard deviation of log returns, annualized:
```math
\sigma_{\text{realized}} = \sqrt{252} \times \sqrt{\frac{1}{19} \sum_{i=1}^{20} (R_i - \bar{R})^2}
```
Where $R_i = \ln(P_t / P_{t-1})$

---

## 🛠️ Quickstart Installation

### 1. Clone the Repository
```bash
git clone https://github.com/adityashinde0/Stock_Prediction_Model_With_RL-ML.git
cd Stock_Prediction_Model_With_RL-ML
```

### 2. Create a Virtual Environment
**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (GPU recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install -r requirements.txt
```
*(Note for Linux users: Ensure you have `libgl1`, `libglib2.0-0`, and `libgomp1` installed via `apt-get` for OpenCV headless support).*

### 4. Launch the Dashboard
```bash
python app.py
```
Open your browser to `http://127.0.0.1:7860`. The models will automatically fetch market data and train themselves on the first run.

---

## 🧪 CI/CD & Testing
This repository includes a comprehensive GitHub Actions workflow (`.github/workflows/ci.yml`) that validates the entire pipeline (OCR, Feature Engineering, Signal Classification, and DL Forecasting) against Python 3.10, 3.11, and 3.12 matrices. 

To run the integration tests locally:
```bash
pytest tests/ -v
```

---

## 🤝 Contributing
This is an open-source project. Contributions, issues, and feature requests are highly welcome! 
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
