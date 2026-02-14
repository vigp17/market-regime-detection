import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/sp500_raw.csv", index_col="Date", parse_dates=True)

# Compute log returns
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

# Drop the first row (NaN from shift)
df = df.dropna()

# Print basic stats
print("Log return statistics:")
print(df["log_return"].describe())
print(f"\nMean daily return: {df['log_return'].mean():.6f}")
print(f"Daily volatility:  {df['log_return'].std():.6f}")
print(f"Worst day:         {df['log_return'].min():.4f} on {df['log_return'].idxmin().date()}")
print(f"Best day:          {df['log_return'].max():.4f} on {df['log_return'].idxmax().date()}")

# Save updated data
df.to_csv("data/sp500_with_returns.csv")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Daily returns over time
axes[0].plot(df.index, df["log_return"], color="steelblue", linewidth=0.4)
axes[0].set_title("Daily Log Returns", fontsize=13)
axes[0].set_ylabel("Log Return")
axes[0].axhline(y=0, color="black", linewidth=0.5)

# Histogram of returns
axes[1].hist(df["log_return"], bins=100, color="steelblue", edgecolor="white")
axes[1].set_title("Distribution of Log Returns", fontsize=13)
axes[1].set_xlabel("Log Return")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("results/log_returns.png", dpi=150)
plt.show()
print("\nChart saved to results/log_returns.png")