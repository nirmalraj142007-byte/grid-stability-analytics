"""
orchestrator.py — Connects predict_live.py and detective_agent/ automatically
================================================================================
This is the missing link discussed in review: right now predict_live.py and
detective_agent.py are two separate scripts you run by hand. This script
simulates a continuous stream of grid readings, and AUTOMATICALLY invokes
the detective agent the moment a prediction is later found to be wrong.

Simulates the real-world timing gap honestly:
  1. A noisy reading comes in -> median filter -> SVM predicts NOW.
  2. Some time later, the TRUE outcome becomes known (in this simulation,
     that's the real label from the test set -- in a real deployment this
     would come from actual grid monitoring, which does not exist in this
     project; see README.md).
  3. The orchestrator compares prediction vs. actual outcome.
  4. On a mismatch (false negative or false positive), it automatically
     calls the detective agent's LangGraph pipeline on that exact case --
     no human has to notice the error or manually run detective_agent.py.
  5. The agent's case-file report is saved and printed.

This keeps the same separation of concerns as the rest of the project:
the prediction path (steps 1) is fast, deterministic code. The agent
(step 4) only runs after an error is CONFIRMED, and only ever produces a
report for a human -- it still never touches the live prediction.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "detective_agent"))

from median_filter import median_filter
from predict_live import train_svm, load_cost_optimal_threshold, FEATURES
from detective_agent import train_models, build_graph, HAS_ANTHROPIC_KEY, HAS_GROQ_KEY

CASE_FILES_DIR = os.path.join(SCRIPT_DIR, "case_files")
os.makedirs(CASE_FILES_DIR, exist_ok=True)


def simulate_noisy_window(true_row, feature_std, rng, window=10):
    noise = rng.normal(0, 0.2 * feature_std, size=(window, len(FEATURES)))
    readings = true_row[None, :] + noise
    if rng.random() < 0.15:  # occasional glitch, matching noise_experiment.py's rate
        readings[rng.integers(0, window)] += rng.normal(0, 1.2 * feature_std, size=len(FEATURES))
    return readings


def run_stream(n_samples=30, seed=11):
    print("Training SVM (prediction path) and XGBoost+SHAP (investigation path)...\n")
    svm_model, scaler = train_svm()
    threshold = load_cost_optimal_threshold()

    # detective agent's own model bundle (SVM + XGBoost + SHAP explainer +
    # the exact X_test/y_test it investigates against)
    agent_models = train_models()
    graph = build_graph(agent_models, _select_llm())

    feature_std = agent_models["X_test"][FEATURES].std().values
    rng = np.random.default_rng(seed)

    print(f"Streaming {n_samples} simulated grid readings "
          f"(cost-optimal decision threshold = {threshold:.3f})\n")
    print(f"{'#':<4}{'Predicted':<12}{'Actual (later)':<16}{'Result':<12}")
    print("-" * 50)

    triggered = 0
    for i in range(n_samples):
        true_row = agent_models["X_test"].iloc[i][FEATURES].values.astype(float)
        actual_label = "unstable" if agent_models["y_test"].iloc[i] == 1 else "stable"

        # --- STEP 1: prediction happens now, from noisy filtered input ---
        noisy_window = simulate_noisy_window(true_row, feature_std, rng)
        filtered = median_filter(noisy_window[None, :, :])
        filtered_df = pd.DataFrame(filtered, columns=FEATURES)
        prob = svm_model.predict_proba(scaler.transform(filtered_df))[0, 1]
        predicted_label = "unstable" if prob >= threshold else "stable"

        # --- STEP 2: (simulated) time passes, true outcome becomes known ---
        is_wrong = predicted_label != actual_label
        result = "MISMATCH" if is_wrong else "correct"
        print(f"{i:<4}{predicted_label:<12}{actual_label:<16}{result:<12}")

        # --- STEP 3 + 4: automatic trigger on confirmed error ---
        if is_wrong:
            triggered += 1
            print(f"     -> Prediction did not match outcome. Automatically invoking detective agent...")
            case = graph.invoke({
                "sample_idx": i,
                "external_prediction": {
                    "predicted_label": predicted_label,
                    "predicted_probability_unstable": float(prob),
                },
                "external_filtered_reading": {f: float(v) for f, v in zip(FEATURES, filtered[0])},
            })
            out_path = os.path.join(CASE_FILES_DIR, f"case_{i:03d}.json")
            with open(out_path, "w") as f:
                json.dump({k: v for k, v in case.items()}, f, indent=2, default=str)
            print(f"     -> Case file written: case_files/case_{i:03d}.json")
            print(f"     -> Verdict: {case['error_type']}\n")

    print("-" * 50)
    print(f"Stream complete. {triggered}/{n_samples} predictions were wrong and "
          f"automatically investigated.")
    print(f"Case files saved in: {CASE_FILES_DIR}/")


def _select_llm():
    if HAS_ANTHROPIC_KEY:
        from langchain_anthropic import ChatAnthropic
        print("Using Claude for agent investigations.")
        return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1000)
    if HAS_GROQ_KEY:
        from langchain_groq import ChatGroq
        print("Using Groq for agent investigations.")
        return ChatGroq(model="llama-3.3-70b-versatile", max_tokens=1000)
    print("No ANTHROPIC_API_KEY or GROQ_API_KEY set -- agent will use its deterministic fallback.")
    return None


if __name__ == "__main__":
    run_stream(n_samples=30)
