import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load everything
df = pd.read_csv("data/sp500_regimes.csv", index_col="Date", parse_dates=True)
model = pickle.load(open("models/hmm_model.pkl", "rb"))

regime_info = {
    0: "Calm Bull",
    1: "Crisis",
    2: "Bear / High Vol",
    3: "Strong Bull",
    4: "Neutral",
}

# Current regime
current_regime = df["regime"].iloc[-1]
print("=" * 50)
print("MARKET REGIME DETECTION — PROJECT SUMMARY")
print("=" * 50)
print(f"\nData: S&P 500, {df.index[0].date()} to {df.index[-1].date()}")
print(f"Total trading days: {len(df)}")
print(f"Features used: log_return, vol_21d, vol_ratio, rsi, ma_distance")
print(f"Model: Gaussian HMM with {model.n_components} states")
print(f"\nCurrent regime: {regime_info[current_regime]} (Regime {current_regime})")

# Regime summary table
print(f"\n{'Regime':<20} {'Days':>6} {'%':>6} {'Return':>10} {'Vol':>8} {'RSI':>6}")
print("-" * 60)
for r in sorted(df["regime"].unique()):
    subset = df[df["regime"] == r]
    label = regime_info[r]
    days = len(subset)
    pct = days / len(df) * 100
    ret = subset["log_return"].mean() * 252 * 100
    vol = subset["vol_21d"].mean() * 100
    rsi = subset["rsi"].mean()
    print(f"{label:<20} {days:>6} {pct:>5.1f}% {ret:>+9.1f}% {vol:>7.1f}% {rsi:>5.1f}")

# Files created
print(f"\n{'='*50}")
print("FILES CREATED")
print(f"{'='*50}")
print("  data/sp500_raw.csv           — Raw price data")
print("  data/sp500_with_returns.csv   — With log returns")
print("  data/sp500_with_volatility.csv — With rolling volatility")
print("  data/sp500_features.csv       — Full feature matrix")
print("  data/sp500_regimes.csv        — With regime labels")
print("  models/hmm_model.pkl          — Trained HMM model")
print("  models/scaler.pkl             — Feature scaler")
print("  results/sp500_price.png       — Price chart")
print("  results/log_returns.png       — Returns analysis")
print("  results/rolling_volatility.png — Volatility chart")
print("  results/regime_chart.png      — Regime visualization")
print("  results/backtest.png          — Strategy performance")