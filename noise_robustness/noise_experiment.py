"""
Noise Robustness Experiment
============================
Question: if real sensor data is noisy, how much does that hurt the model,
and which filter recovers the most performance?

Setup, per test row:
  1. Take the TRUE (clean) 12-feature vector for that grid state.
  2. Simulate W=10 sequential noisy sensor readings of that same true state
     (Gaussian noise + occasional outlier spikes, at several severity levels).
  3. Feed the model:
       (a) the TRUE clean vector           -> upper bound (best possible)
       (b) ONE raw noisy reading, no filter -> worst case / naive baseline
       (c) a moving average of the 10 noisy readings
       (d) a median filter of the 10 noisy readings
       (e) z-score outlier rejection + mean of the remainder
       (f) a scalar Kalman filter run sequentially over the 10 readings
  4. Compare test AUC for each -> tells us how much each filter recovers.

This uses the already-trained cost-sensitive SVM (the stronger model from
the main pipeline) as the fixed classifier; only the INPUT processing changes.
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

FEATURES = ["tau1", "tau2", "tau3", "tau4",
            "p1", "p2", "p3", "p4",
            "g1", "g2", "g3", "g4"]

df = pd.read_csv("Data_for_UCI_named.csv")
X = df[FEATURES].copy()
y = (df["stabf"] == "unstable").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# -----------------------------------------------------------------------
# Train the model once on CLEAN data (as in the main project)
# -----------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                class_weight={0: 1, 1: 5}, random_state=RANDOM_STATE)
svm_model.fit(X_train_s, y_train)

baseline_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test_s)[:, 1])
print(f"Baseline (clean data) test AUC: {baseline_auc:.4f}")

feature_std = X_train.std().values  # per-feature scale, for setting realistic noise magnitude

# -----------------------------------------------------------------------
# Noise simulation
# -----------------------------------------------------------------------
W = 10  # simulated sequential noisy readings per test row

def simulate_noisy_readings(x_true, noise_frac, outlier_rate=0.06, outlier_mult=6.0):
    """
    x_true: (n_samples, n_features) clean values
    Returns: (n_samples, W, n_features) simulated noisy readings
    """
    n_samples, n_feat = x_true.shape
    noise_std = noise_frac * feature_std  # per-feature std
    readings = np.repeat(x_true[:, None, :], W, axis=1).astype(float)
    gaussian_noise = rng.normal(0, noise_std, size=(n_samples, W, n_feat))
    readings += gaussian_noise
    # inject occasional outlier spikes (sensor glitches)
    outlier_mask = rng.random((n_samples, W, n_feat)) < outlier_rate
    spike = rng.normal(0, noise_std * outlier_mult, size=(n_samples, W, n_feat))
    readings += outlier_mask * spike
    return readings


def filter_none(readings):
    return readings[:, 0, :]  # just the first raw noisy reading


def filter_moving_average(readings):
    return readings.mean(axis=1)


def filter_median(readings):
    return np.median(readings, axis=1)


def filter_zscore_then_mean(readings, z_thresh=3.0):
    mean = readings.mean(axis=1, keepdims=True)
    std = readings.std(axis=1, keepdims=True) + 1e-9
    z = np.abs((readings - mean) / std)
    mask = z <= z_thresh
    masked = np.where(mask, readings, np.nan)
    out = np.nanmean(masked, axis=1)
    out = np.where(np.isnan(out), readings.mean(axis=1), out)
    return out


def filter_kalman(readings, process_var=1e-4, meas_var_frac=None):
    """
    Scalar Kalman filter run independently per feature, per sample,
    estimating a (near-)constant true value from W sequential noisy readings.
    """
    n_samples, w, n_feat = readings.shape
    if meas_var_frac is None:
        meas_var_frac = 0.5  # assumed measurement noise variance scale
    R = (meas_var_frac * feature_std) ** 2  # measurement variance per feature
    Q = process_var * (feature_std ** 2)     # process variance per feature

    x_hat = readings[:, 0, :].copy()
    P = np.tile(feature_std ** 2, (n_samples, 1))  # initial uncertainty

    for t in range(1, w):
        P = P + Q  # predict
        K = P / (P + R)  # Kalman gain
        z = readings[:, t, :]
        x_hat = x_hat + K * (z - x_hat)  # update
        P = (1 - K) * P
    return x_hat


def evaluate(x_estimated, y_true, model, scaler):
    x_scaled = scaler.transform(pd.DataFrame(x_estimated, columns=FEATURES))
    prob = model.predict_proba(x_scaled)[:, 1]
    return roc_auc_score(y_true, prob)


# -----------------------------------------------------------------------
# Run across several noise severities
# -----------------------------------------------------------------------
noise_levels = [0.05, 0.10, 0.20, 0.35, 0.50]  # fraction of feature std as Gaussian noise std
methods = {
    "No filter (raw noisy reading)": filter_none,
    "Moving average (10 readings)": filter_moving_average,
    "Median filter (10 readings)": filter_median,
    "Z-score reject + mean": filter_zscore_then_mean,
    "Kalman filter": filter_kalman,
}

results = {name: [] for name in methods}
x_true_test = X_test.values

for nl in noise_levels:
    readings = simulate_noisy_readings(x_true_test, noise_frac=nl)
    for name, fn in methods.items():
        x_est = fn(readings)
        auc = evaluate(x_est, y_test.values, svm_model, scaler)
        results[name].append(auc)
        print(f"noise={nl:.2f}  {name:35s} AUC={auc:.4f}")

# clean upper bound (repeated for the plot as a flat reference line)
clean_auc = baseline_auc

# -----------------------------------------------------------------------
# Save results + plot
# -----------------------------------------------------------------------
summary = {
    "clean_baseline_auc": float(clean_auc),
    "noise_levels": noise_levels,
    "results": {k: [float(v) for v in vals] for k, vals in results.items()},
}
with open("noise_experiment_results.json", "w") as f:
    json.dump(summary, f, indent=2)

plt.figure(figsize=(8, 5.5))
colors = {
    "No filter (raw noisy reading)": "#C1121F",
    "Moving average (10 readings)": "#8E44AD",
    "Median filter (10 readings)": "#2980B9",
    "Z-score reject + mean": "#F39C12",
    "Kalman filter": "#1B7A43",
}
for name, vals in results.items():
    plt.plot(noise_levels, vals, marker="o", label=name, color=colors[name], lw=2)
plt.axhline(clean_auc, color="gray", ls="--", lw=1.5, label="Clean data (upper bound)")
plt.xlabel("Noise severity (fraction of feature std-dev)")
plt.ylabel("Test AUC")
plt.title("Model Performance Under Sensor Noise, With and Without Filtering")
plt.legend(fontsize=9, loc="lower left")
plt.tight_layout()
plt.savefig("noise_robustness_comparison.png", dpi=150)
plt.close()

# Print final ranking at the highest noise level (worst case = most informative)
print("\n=== Ranking at highest noise level tested ({:.2f}) ===".format(noise_levels[-1]))
final_scores = sorted(((name, vals[-1]) for name, vals in results.items()), key=lambda x: -x[1])
for name, score in final_scores:
    print(f"{name:35s} AUC={score:.4f}")

print("\nSaved noise_experiment_results.json and noise_robustness_comparison.png")
