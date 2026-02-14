import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data/sp500_raw.csv", index_col="Date", parse_dates=True)

# Basic info
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# Plot the closing price
plt.figure(figsize=(14, 6))
plt.plot(df["Close"], color="black", linewidth=0.8)
plt.title("S&P 500 (2005 - Present)", fontsize=14)
plt.ylabel("Price")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig("results/sp500_price.png", dpi=150)
plt.show()
print("\nChart saved to results/sp500_price.png")
