"""
Financial Strategy Backtester Module
Simulates trading based on model signals and calculates Sharpe Ratio, Max Drawdown, and Cumulative Returns.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config import INITIAL_CAPITAL, TRANSACTION_FEE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StrategyBacktester")


class StrategyBacktester:
    """
    Backtests trading signals against historical market prices and computes risk-adjusted metrics.
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL, fee: float = TRANSACTION_FEE):
        self.initial_capital = initial_capital
        self.fee = fee

    def run_backtest(self, df: pd.DataFrame, signal_column: str = "Signal") -> Dict[str, Any]:
        """
        Runs a simulation over historical data where:
          - BUY (3) / STRONG BUY (4) -> Enter 100% Long
          - SELL (1) / STRONG SELL (0) -> Exit to 100% Cash
          - NEUTRAL (2) -> Hold current position
        """
        data = df.copy().reset_index(drop=True)
        if signal_column not in data.columns:
            raise ValueError(f"Column '{signal_column}' not found in DataFrame for backtesting.")

        cash = self.initial_capital
        shares = 0.0
        portfolio_values = []
        position = 0  # 0 = Cash, 1 = Long

        for i, row in data.iterrows():
            price = float(row["Adj Close"])
            signal = int(row[signal_column])

            # Buy Signal: Move from Cash to Long
            if signal in [3, 4] and position == 0:
                shares = (cash * (1.0 - self.fee)) / price
                cash = 0.0
                position = 1
            # Sell Signal: Move from Long to Cash
            elif signal in [0, 1] and position == 1:
                cash = shares * price * (1.0 - self.fee)
                shares = 0.0
                position = 0

            # Calculate daily total equity
            current_value = cash + (shares * price)
            portfolio_values.append(current_value)

        data["Portfolio_Value"] = portfolio_values
        data["Strategy_Return"] = data["Portfolio_Value"].pct_change().fillna(0.0)
        data["Benchmark_Value"] = (
            self.initial_capital * (data["Adj Close"] / data["Adj Close"].iloc[0])
        )

        return self.calculate_metrics(data)

    def calculate_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes financial KPIs: Total Return, Benchmark Return, Sharpe Ratio, and Max Drawdown.
        """
        final_value = float(data["Portfolio_Value"].iloc[-1])
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100.0

        benchmark_final = float(data["Benchmark_Value"].iloc[-1])
        benchmark_return = ((benchmark_final - self.initial_capital) / self.initial_capital) * 100.0

        # Annualized Sharpe Ratio (assuming 252 trading days for daily intervals)
        mean_ret = np.mean(data["Strategy_Return"])
        std_ret = np.std(data["Strategy_Return"])
        sharpe_ratio = 0.0
        if std_ret > 0:
            sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(252))

        # Maximum Drawdown (MDD)
        rolling_max = data["Portfolio_Value"].cummax()
        drawdown = (data["Portfolio_Value"] - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min()) * 100.0

        return {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": round(final_value, 2),
            "strategy_total_return_pct": round(total_return, 2),
            "benchmark_hold_return_pct": round(benchmark_return, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
        }


# --- Standalone Verification Block ---
if __name__ == "__main__":
    from src.data.fetcher import MarketDataFetcher
    from src.data.indicators import TechnicalIndicatorEngine
    from src.models.signal_classifier import SignalClassifier

    logger.info("Running backtest verification on BTC-USD...")
    raw_df = MarketDataFetcher.fetch_data(ticker="BTC-USD", period="2y", interval="1d")
    processed_df = TechnicalIndicatorEngine.add_all_indicators(raw_df)

    classifier = SignalClassifier()
    classifier.train(processed_df)

    # Generate historical signals for backtesting
    processed_df = classifier.create_target_labels(processed_df)
    processed_df.rename(columns={"Target": "Signal"}, inplace=True)

    backtester = StrategyBacktester()
    results = backtester.run_backtest(processed_df, signal_column="Signal")

    print("\n--- Financial Strategy Backtester Verification Result ---")
    print(f"Initial Capital         : ${results['initial_capital']}")
    print(f"Final Portfolio Value   : ${results['final_portfolio_value']}")
    print(f"Strategy Total Return   : {results['strategy_total_return_pct']}%")
    print(f"Buy-and-Hold Benchmark  : {results['benchmark_hold_return_pct']}%")
    print(f"Annualized Sharpe Ratio : {results['sharpe_ratio']}")
    print(f"Maximum Drawdown        : {results['max_drawdown_pct']}%")