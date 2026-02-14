import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data with regimes
df = pd.read_csv("data/sp500_regimes.csv", index_col="Date", parse_dates=True)

# Strategy: Adjust allocation based on regime
# Calm Bull (0): 100% invested
# Crisis (1): 0% invested (go to cash)
# Bear/High Vol (2): 30% invested
# Strong Bull (3): 100% invested
# Neutral (4): 70% invested

allocation = {0: 1.0, 1: 0.0, 2: 0.3, 3: 1.0, 4: 0.7}

df["strategy_return"] = df["regime"].map(allocation).shift(1) * df["log_return"]

# Shift(1) is critical — we use YESTERDAY's regime to decide TODAY's allocation
# This avoids look-ahead bias

# Cumulative returns
df["buy_hold_cumulative"] = df["log_return"].cumsum().apply(np.exp)
df["strategy_cumulative"] = df["strategy_return"].cumsum().apply(np.exp)

# Performance metrics
def calc_metrics(returns, label):
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol
    cumulative = returns.cumsum().apply(np.exp)
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    print(f"\n{label}:")
    print(f"  Annual return:   {annual_return*100:.1f}%")
    print(f"  Annual vol:      {annual_vol*100:.1f}%")
    print(f"  Sharpe ratio:    {sharpe:.2f}")
    print(f"  Max drawdown:    {max_drawdown*100:.1f}%")

print("=" * 50)
print("STRATEGY vs BUY-AND-HOLD")
print("=" * 50)

calc_metrics(df["log_return"].dropna(), "Buy & Hold")
calc_metrics(df["strategy_return"].dropna(), "Regime Strategy")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])

ax1.plot(df.index, df["buy_hold_cumulative"], label="Buy & Hold", color="black", linewidth=1)
ax1.plot(df.index, df["strategy_cumulative"], label="Regime Strategy", color="#2ecc71", linewidth=1)
ax1.set_title("Regime-Aware Strategy vs Buy & Hold", fontsize=14, fontweight="bold")
ax1.set_ylabel("Growth of $1", fontsize=12)
ax1.legend(fontsize=11)

# Drawdown comparison
bh_cum = df["buy_hold_cumulative"]
st_cum = df["strategy_cumulative"]
bh_dd = (bh_cum - bh_cum.cummax()) / bh_cum.cummax()
st_dd = (st_cum - st_cum.cummax()) / st_cum.cummax()

ax2.fill_between(df.index, bh_dd, 0, alpha=0.3, color="black", label="Buy & Hold")
ax2.fill_between(df.index, st_dd, 0, alpha=0.3, color="#2ecc71", label="Regime Strategy")
ax2.set_title("Drawdowns", fontsize=13)
ax2.set_ylabel("Drawdown", fontsize=12)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig("results/backtest.png", dpi=150)
plt.show()
print("\nSaved to results/backtest.png")