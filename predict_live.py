"""
predict_live.py — The Integrated Prediction Pipeline
=======================================================
This is the piece that actually STITCHES the project together:

    raw noisy sensor readings
            |
            v
    median_filter.py            <- chosen after testing 5 filters
            |                       (see noise_robustness/noise_experiment.py)
            v
    trained SVM (pipeline.py)   <- cost-sensitive, tuned via CV
            |
            v
    cost-optimal threshold      <- from metrics_summary.json
            |
            v
    STABLE / UNSTABLE decision

Everything up to and including the decision is 100% deterministic code —
no LLM, no agent, fast enough for a real-time path. The detective_agent/
sits AFTER this: only once a prediction is later confirmed wrong does it
investigate, using this exact same median_filter() function to check
whether noise contributed to the error.

Run this file directly for a demo using one simulated noisy grid reading.
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from median_filter import median_filter

RANDOM_STATE = 42
FEATURES = ["tau1", "tau2", "tau3", "tau4",
            "p1", "p2", "p3", "p4",
            "g1", "g2", "g3", "g4"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "Data_for_UCI_named.csv")
METRICS_PATH = os.path.join(SCRIPT_DIR, "metrics_summary.json")


def train_svm():
    """Same training setup as pipeline.py — kept here in one function so
    predict_live.py can run standalone. In a real deployment this would
    load a saved/pickled model instead of retraining every time."""
    df = pd.read_csv(CSV_PATH)
    X = df[FEATURES].copy()
    y = (df["stabf"] == "unstable").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(X_train)
    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                    class_weight={0: 1, 1: 5}, random_state=RANDOM_STATE)
    svm_model.fit(scaler.transform(X_train), y_train)
    return svm_model, scaler


def load_cost_optimal_threshold(default=0.5):
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        # metrics_summary.json stores the XGBoost threshold; SVM's own
        # threshold was computed the same way in pipeline.py. Falling back
        # to 0.5 if the key isn't present keeps this script runnable on
        # its own.
        return metrics.get("svm", {}).get("cost_optimal_threshold", default)
    return default


def predict_from_noisy_window(svm_model, scaler, noisy_readings, threshold):
    """
    noisy_readings: np.ndarray, shape (window_size, n_features) — a short
    history of raw noisy sensor readings for ONE grid state.
    """
    filtered = median_filter(noisy_readings[None, :, :])  # -> (1, n_features)
    filtered_df = pd.DataFrame(filtered, columns=FEATURES)
    prob_unstable = svm_model.predict_proba(scaler.transform(filtered_df))[0, 1]
    decision = "UNSTABLE" if prob_unstable >= threshold else "STABLE"
    return {
        "filtered_reading": {f: float(v) for f, v in zip(FEATURES, filtered[0])},
        "probability_unstable": float(prob_unstable),
        "threshold_used": float(threshold),
        "decision": decision,
    }


if __name__ == "__main__":
    print("Training SVM on clean historical data...")
    svm_model, scaler = train_svm()
    threshold = load_cost_optimal_threshold()
    print(f"Using cost-optimal decision threshold: {threshold:.3f}\n")

    # --- Simulate one real-time window of noisy sensor readings ---
    rng = np.random.default_rng(7)
    true_state = np.array([2.1, 3.4, 8.7, 4.9,      # tau1-4 (an unstable-leaning state)
                            3.9, -1.3, -1.2, -1.4,    # p1-4
                            0.75, 0.62, 0.58, 0.71])  # g1-4
    feature_std = np.array([2.7] * 4 + [0.7] * 4 + [0.27] * 4)  # rough dataset-wide scale

    window_size = 10
    noise = rng.normal(0, 0.2 * feature_std, size=(window_size, len(FEATURES)))
    noisy_readings = true_state[None, :] + noise
    # inject one glitch reading
    noisy_readings[3] += rng.normal(0, 1.5, size=len(FEATURES))

    result = predict_from_noisy_window(svm_model, scaler, noisy_readings, threshold)

    print("Raw reading #1 (what a 'no filter' system would use):")
    print("  ", dict(zip(FEATURES, np.round(noisy_readings[0], 3))))
    print("\nMedian-filtered estimate (what actually gets predicted on):")
    print("  ", {k: round(v, 3) for k, v in result["filtered_reading"].items()})
    print(f"\nP(unstable) = {result['probability_unstable']:.3f}")
    print(f"Decision threshold = {result['threshold_used']:.3f}")
    print(f"FINAL DECISION: {result['decision']}")

    with open(os.path.join(SCRIPT_DIR, "predict_live_example_output.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved predict_live_example_output.json")
