import yfinance as yf
import pandas as pd

# Download S&P 500 data (2005 to today)
print("Downloading S&P 500 data...")
sp500 = yf.download("^GSPC", start="2005-01-01")

# Flatten multi-level columns if present
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)

# Save to CSV
sp500.to_csv("data/sp500_raw.csv")
print(f"Saved {len(sp500)} rows to data/sp500_raw.csv")
print(sp500.tail())