import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data with returns
df = pd.read_csv("data/sp500_with_returns.csv", index_col="Date", parse_dates=True)

# Rolling volatility at different windows (annualized)
df["vol_5d"] = df["log_return"].rolling(5).std() * np.sqrt(252)
df["vol_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)
df["vol_63d"] = df["log_return"].rolling(63).std() * np.sqrt(252)

# Drop NaN rows
df = df.dropna()

# Save
df.to_csv("data/sp500_with_volatility.csv")

# Print stats
print("21-day rolling volatility stats (annualized):")
print(df["vol_21d"].describe())
print(f"\nCalm period example (2017 avg): {df.loc['2017', 'vol_21d'].mean():.4f}")
print(f"Crisis period example (2020 avg): {df.loc['2020', 'vol_21d'].mean():.4f}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Price
axes[0].plot(df.index, df["Close"], color="black", linewidth=0.8)
axes[0].set_title("S&P 500 Price", fontsize=13)
axes[0].set_ylabel("Price")

# Rolling volatility
axes[1].plot(df.index, df["vol_5d"], alpha=0.4, label="5-day", linewidth=0.6)
axes[1].plot(df.index, df["vol_21d"], alpha=0.8, label="21-day", linewidth=1)
axes[1].plot(df.index, df["vol_63d"], alpha=0.8, label="63-day", linewidth=1)
axes[1].set_title("Rolling Volatility (Annualized)", fontsize=13)
axes[1].set_ylabel("Volatility")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/rolling_volatility.png", dpi=150)
plt.show()
print("\nChart saved to results/rolling_volatility.png")