import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load features
df = pd.read_csv("data/sp500_features.csv", index_col="Date", parse_dates=True)
feature_cols = ["log_return", "vol_21d", "vol_ratio", "rsi", "ma_distance"]

# Scale features (HMM works better with normalized data)
scaler = StandardScaler()
X = scaler.fit_transform(df[feature_cols])

# Try different number of states and pick the best using BIC
print("Finding optimal number of regimes...\n")
results = {}

for n_states in [2, 3, 4, 5]:
    best_score = -np.inf
    best_model = None

    # Run multiple times (EM can get stuck in local optima)
    for attempt in range(10):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=attempt
        )
        model.fit(X)
        score = model.score(X)

        if score > best_score:
            best_score = score
            best_model = model

    # BIC = -2 * log_likelihood + num_params * log(n_samples)
    n_params = n_states * (n_states - 1) + n_states * len(feature_cols) + n_states * len(feature_cols) * (len(feature_cols) + 1) // 2
    bic = -2 * best_score + n_params * np.log(len(X))

    results[n_states] = {"model": best_model, "score": best_score, "bic": bic}
    print(f"  {n_states} states → Log-likelihood: {best_score:.0f}, BIC: {bic:.0f}")

# Pick the model with lowest BIC
best_n = min(results, key=lambda k: results[k]["bic"])
best_model = results[best_n]["model"]
print(f"\nBest model: {best_n} regimes (lowest BIC)")

# Decode regimes
regimes = best_model.predict(X)
df["regime"] = regimes

# Print regime characteristics
print(f"\n{'='*60}")
print("REGIME CHARACTERISTICS")
print(f"{'='*60}")
for regime in sorted(df["regime"].unique()):
    subset = df[df["regime"] == regime]
    print(f"\nRegime {regime} ({len(subset)} days, {len(subset)/len(df)*100:.1f}%)")
    print(f"  Avg daily return:  {subset['log_return'].mean()*252:.1f}% annualized")
    print(f"  Avg volatility:    {subset['vol_21d'].mean()*100:.1f}%")
    print(f"  Avg RSI:           {subset['rsi'].mean():.1f}")
    print(f"  Avg MA distance:   {subset['ma_distance'].mean()*100:.2f}%")

# Print transition matrix
print(f"\nTransition matrix:")
print(pd.DataFrame(
    best_model.transmat_.round(3),
    index=[f"From {i}" for i in range(best_n)],
    columns=[f"To {i}" for i in range(best_n)]
))

# Save
df.to_csv("data/sp500_regimes.csv")
pickle.dump(best_model, open("models/hmm_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
print("\nSaved regimes to data/sp500_regimes.csv")
print("Saved model to models/hmm_model.pkl")