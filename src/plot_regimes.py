import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# Load data with regimes
df = pd.read_csv("data/sp500_regimes.csv", index_col="Date", parse_dates=True)

# Regime labels and colors
regime_info = {
    0: ("Calm Bull", "#2ecc71"),
    1: ("Crisis", "#e74c3c"),
    2: ("Bear / High Vol", "#e67e22"),
    3: ("Strong Bull", "#3498db"),
    4: ("Neutral", "#95a5a6"),
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), height_ratios=[3, 1], sharex=True)

# Top: Price with regime backgrounds
ax1.plot(df.index, df["Close"], color="black", linewidth=0.7)
ax1.set_ylabel("Price", fontsize=12)
ax1.set_title("S&P 500 — Market Regimes Detected by Hidden Markov Model", fontsize=14, fontweight="bold")

for i in range(len(df) - 1):
    regime = df["regime"].iloc[i]
    color = regime_info[regime][1]
    ax1.axvspan(df.index[i], df.index[i + 1], alpha=0.3, color=color, linewidth=0)

legend_elements = [
    Patch(facecolor=color, alpha=0.5, label=label)
    for regime, (label, color) in sorted(regime_info.items())
]
ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)

# Bottom: Regime timeline
for regime, (label, color) in regime_info.items():
    mask = df["regime"] == regime
    ax2.fill_between(df.index, 0, 1, where=mask, color=color, alpha=0.7)

ax2.set_ylabel("Regime", fontsize=12)
ax2.set_yticks([])
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
plt.savefig("results/regime_chart.png", dpi=150)
plt.show()
print("Saved to results/regime_chart.png")