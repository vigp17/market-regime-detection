import pandas as pd
import numpy as np

# Load data with volatility
df = pd.read_csv("data/sp500_with_volatility.csv", index_col="Date", parse_dates=True)

# Feature 1: Log return (already have it)

# Feature 2: 21-day rolling volatility (already have it)

# Feature 3: Volatility ratio (short-term vs long-term)
# Values > 1 mean vol is increasing, < 1 means calming down
df["vol_ratio"] = df["vol_5d"] / df["vol_21d"]

# Feature 4: RSI (Relative Strength Index)
# Measures if market is overbought (>70) or oversold (<30)
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0.0)
loss = (-delta).where(delta < 0, 0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["rsi"] = 100 - (100 / (1 + rs))

# Feature 5: Distance from 20-day moving average (normalized)
# Positive = price above trend, negative = below
ma_20 = df["Close"].rolling(20).mean()
df["ma_distance"] = (df["Close"] - ma_20) / ma_20

# Drop NaN rows
df = df.dropna()

# Select our features
feature_cols = ["log_return", "vol_21d", "vol_ratio", "rsi", "ma_distance"]

print("Feature matrix ready!")
print(f"Shape: {len(df)} rows x {len(feature_cols)} features")
print(f"\nFeature summary:")
print(df[feature_cols].describe().round(4))

# Save
df.to_csv("data/sp500_features.csv")
print("\nSaved to data/sp500_features.csv")