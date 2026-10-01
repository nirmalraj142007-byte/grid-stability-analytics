"""
The Detective Agent
=====================
Investigates a SPECIFIC wrong prediction (a false negative or false positive)
made by the grid-stability classifier, and writes a plain-language "case file"
explaining why the model got it wrong.

Architecture (LangGraph state machine):

    [gather_evidence]  -- deterministic, no LLM --
            |             pulls: prediction, confidence, ground truth,
            |             SHAP contributions, raw-vs-filtered sensor readings
            v
    [investigate]       -- LLM reasoning step --
            |             the LLM reads the evidence and reasons about
            |             the most likely root cause
            v
    [write_report]      -- LLM writing step --
                          turns the investigation into a short, readable
                          case-file report for a human reviewer

This mirrors the design principle from the rest of the project: the
real-time, safety-critical path (filtering + prediction) is 100%
deterministic code. The agent sits AFTER that path, investigating
confirmed errors after the fact -- it never touches the live prediction
or retrains anything itself. A human reads its report and decides what,
if anything, to do next.

Requires an ANTHROPIC_API_KEY or GROQ_API_KEY environment variable to use
a real LLM (Anthropic is preferred if both are set). Without either, this
script still runs end-to-end using a deterministic template in place of
the LLM calls, so the graph structure and evidence-gathering logic can be
demonstrated/tested without live API access.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import TypedDict, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
import shap

from langgraph.graph import StateGraph, END

# The detective agent uses the SAME median filter chosen in the noise-
# robustness experiment (see ../noise_robustness/noise_experiment.py) —
# not a re-implementation. This keeps the "was this reading noisy"
# check consistent with what the live prediction pipeline would actually use.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from median_filter import median_filter

HAS_ANTHROPIC_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_GROQ_KEY = bool(os.environ.get("GROQ_API_KEY"))
HAS_API_KEY = HAS_ANTHROPIC_KEY or HAS_GROQ_KEY

if HAS_ANTHROPIC_KEY:
    from langchain_anthropic import ChatAnthropic
if HAS_GROQ_KEY:
    from langchain_groq import ChatGroq
if HAS_API_KEY:
    from langchain_core.messages import SystemMessage, HumanMessage

RANDOM_STATE = 42
FEATURES = ["tau1", "tau2", "tau3", "tau4",
            "p1", "p2", "p3", "p4",
            "g1", "g2", "g3", "g4"]

# =========================================================================
# 1. TRAIN THE MODELS  (same setup as pipeline.py — SVM is the classifier,
#    XGBoost is used only as the fast SHAP explainer, exactly as in the
#    main project)
# =========================================================================

def train_models():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "Data_for_UCI_named.csv")
    df = pd.read_csv(csv_path)
    X = df[FEATURES].copy()
    y = (df["stabf"] == "unstable").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                    class_weight={0: 1, 1: 5}, random_state=RANDOM_STATE)
    svm_model.fit(X_train_s, y_train)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=6,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(xgb_model)

    return {
        "svm_model": svm_model, "scaler": scaler, "xgb_model": xgb_model,
        "explainer": explainer, "X_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }


# =========================================================================
# 2. TOOLS — deterministic evidence-gathering functions the agent calls.
#    None of these involve an LLM; they're plain code, same as the rest
#    of the project.
# =========================================================================

def get_prediction_and_confidence(models, idx):
    row = models["X_test"].iloc[[idx]]
    row_s = models["scaler"].transform(row)
    prob = models["svm_model"].predict_proba(row_s)[0, 1]
    pred_label = "unstable" if prob >= 0.5 else "stable"
    return {"predicted_label": pred_label, "predicted_probability_unstable": float(prob)}


def get_ground_truth(models, idx):
    actual = int(models["y_test"].iloc[idx])
    return {"actual_label": "unstable" if actual == 1 else "stable"}


def get_shap_explanation(models, idx, top_k=4):
    row = models["X_test"].iloc[[idx]]
    shap_values = models["explainer"].shap_values(row)[0]
    contributions = list(zip(FEATURES, row.iloc[0].values, shap_values))
    contributions.sort(key=lambda t: -abs(t[2]))
    top = contributions[:top_k]
    return [{"feature": f, "value": float(v), "shap_contribution": float(s)} for f, v, s in top]


def check_sensor_quality(models, idx, rng_seed=None):
    """
    Simulates what a real sensor-quality check would look like: takes the
    row's true values, simulates a noisy 10-reading window (same model as
    noise_experiment.py), and compares the raw single reading against the
    median-filtered estimate. A big raw-vs-filtered gap on any feature is
    a signal that noise or a sensor glitch may have contributed to the
    error.
    """
    rng = np.random.default_rng(rng_seed if rng_seed is not None else idx)
    row = models["X_test"].iloc[idx][FEATURES].values.astype(float)
    feature_std = models["X_test"][FEATURES].std().values

    W = 10
    noise = rng.normal(0, 0.15 * feature_std, size=(W, len(FEATURES)))
    readings = row[None, :] + noise
    # 6% chance of an outlier spike per reading, matching noise_experiment.py
    outlier_mask = rng.random((W, len(FEATURES))) < 0.06
    spike = rng.normal(0, 0.9 * feature_std, size=(W, len(FEATURES)))
    readings += outlier_mask * spike

    raw_reading = readings[0]
    filtered = median_filter(readings[None, :, :])[0]  # (1, W, n_features) -> (n_features,)
    gap = np.abs(raw_reading - filtered)
    worst_idx = np.argsort(-gap)[:3]
    flags = [{"feature": FEATURES[i], "raw": float(raw_reading[i]),
              "filtered": float(filtered[i]), "gap": float(gap[i])} for i in worst_idx]
    return {"largest_raw_vs_filtered_gaps": flags,
            "any_outlier_injected_in_window": bool(outlier_mask.any())}


# =========================================================================
# 3. LANGGRAPH STATE + NODES
# =========================================================================

class CaseState(TypedDict):
    sample_idx: int
    prediction: dict
    ground_truth: dict
    error_type: str
    shap_evidence: list
    sensor_evidence: dict
    investigation: Optional[str]
    report: Optional[str]
    # Optional: the actual prediction + noisy/filtered reading that triggered
    # this investigation (e.g. from orchestrator.py). If NOT provided, this
    # node falls back to recomputing a prediction from the clean test-set
    # row -- which is only appropriate when running detective_agent.py
    # standalone against a known historical test case, not when investigating
    # a live, noise-driven prediction someone else already made.
    external_prediction: Optional[dict]
    external_filtered_reading: Optional[dict]


def node_gather_evidence(state: CaseState, models) -> CaseState:
    idx = state["sample_idx"]
    ground_truth = get_ground_truth(models, idx)

    external_pred = state.get("external_prediction")
    external_reading = state.get("external_filtered_reading")

    if external_pred is not None:
        # Investigate the ACTUAL event that triggered this case -- the real
        # prediction made on the real (noisy, filtered) reading at the time,
        # not a fresh recomputation from clean historical data.
        prediction = external_pred
        shap_row = pd.DataFrame([external_reading], columns=FEATURES)
        shap_values = models["explainer"].shap_values(shap_row)[0]
        contributions = sorted(
            zip(FEATURES, shap_row.iloc[0].values, shap_values),
            key=lambda t: -abs(t[2])
        )[:4]
        state["shap_evidence"] = [
            {"feature": f, "value": float(v), "shap_contribution": float(s)}
            for f, v, s in contributions
        ]
    else:
        prediction = get_prediction_and_confidence(models, idx)
        state["shap_evidence"] = get_shap_explanation(models, idx)

    if prediction["predicted_label"] == "stable" and ground_truth["actual_label"] == "unstable":
        error_type = "False Negative (missed instability)"
    elif prediction["predicted_label"] == "unstable" and ground_truth["actual_label"] == "stable":
        error_type = "False Positive (false alarm)"
    else:
        error_type = "Correct prediction (no error to investigate)"

    state["prediction"] = prediction
    state["ground_truth"] = ground_truth
    state["error_type"] = error_type
    state["sensor_evidence"] = check_sensor_quality(models, idx)
    return state


def node_investigate(state: CaseState, llm) -> CaseState:
    evidence_summary = json.dumps({
        "prediction": state["prediction"],
        "ground_truth": state["ground_truth"],
        "error_type": state["error_type"],
        "top_shap_contributions": state["shap_evidence"],
        "sensor_quality_check": state["sensor_evidence"],
    }, indent=2)

    if llm is not None:
        messages = [
            SystemMessage(content=(
                "You are a detective investigating why a machine learning grid-stability "
                "classifier made a wrong prediction. You are given hard evidence: the "
                "model's prediction, the actual outcome, which input features drove the "
                "prediction (SHAP values), and a sensor-quality check comparing a raw "
                "reading to a noise-filtered estimate. Reason step by step about the most "
                "likely root cause. Do not speculate wildly -- ground every claim in the "
                "evidence provided. Be concise, 4-6 sentences."
            )),
            HumanMessage(content=f"Case evidence:\n{evidence_summary}\n\nWhat most likely caused this error?"),
        ]
        response = llm.invoke(messages)
        state["investigation"] = response.content
    else:
        # Deterministic fallback (no API key) -- rule-based, not an LLM,
        # but demonstrates the same evidence -> reasoning structure.
        top_feat = state["shap_evidence"][0]
        gaps = state["sensor_evidence"]["largest_raw_vs_filtered_gaps"]
        biggest_gap = max(gaps, key=lambda g: g["gap"])
        state["investigation"] = (
            f"[Deterministic fallback -- no ANTHROPIC_API_KEY or GROQ_API_KEY set] "
            f"The model's prediction was most heavily driven by '{top_feat['feature']}' "
            f"(value={top_feat['value']:.3f}, SHAP contribution={top_feat['shap_contribution']:.3f}). "
            f"The sensor-quality check found its largest raw-vs-filtered discrepancy on "
            f"'{biggest_gap['feature']}' (gap={biggest_gap['gap']:.3f}), "
            f"{'with an outlier spike present in the simulated reading window' if state['sensor_evidence']['any_outlier_injected_in_window'] else 'with no outlier spike detected in this window'}. "
            f"This suggests the error is plausibly linked to noise on the dominant feature "
            f"driving the prediction, though this is a rule-based summary, not an LLM-reasoned conclusion."
        )
    return state


def node_write_report(state: CaseState, llm) -> CaseState:
    if llm is not None:
        messages = [
            SystemMessage(content=(
                "You write short, clear incident reports for non-technical grid operators. "
                "Given a detective's investigation notes, write a case-file style report: "
                "a one-line verdict, then 3 short bullet points of supporting evidence, then "
                "one recommended next action for a human reviewer. No jargon without a plain-"
                "language explanation. Keep it under 150 words."
            )),
            HumanMessage(content=f"Investigation notes:\n{state['investigation']}\n\nEvidence:\n"
                                  f"Prediction: {state['prediction']}\n"
                                  f"Ground truth: {state['ground_truth']}\n"
                                  f"Error type: {state['error_type']}"),
        ]
        response = llm.invoke(messages)
        state["report"] = response.content
    else:
        state["report"] = (
            f"CASE FILE — Sample #{state['sample_idx']}\n"
            f"Verdict: {state['error_type']}\n\n"
            f"Evidence:\n"
            f"- Model predicted '{state['prediction']['predicted_label']}' "
            f"(confidence: {state['prediction']['predicted_probability_unstable']:.1%} unstable)\n"
            f"- Actual outcome was '{state['ground_truth']['actual_label']}'\n"
            f"- Top driving feature: {state['shap_evidence'][0]['feature']}\n\n"
            f"Investigation notes:\n{state['investigation']}\n\n"
            f"Recommended next action: Human reviewer to confirm sensor data quality for the "
            f"flagged feature(s) before deciding whether retraining is warranted."
        )
    return state


# =========================================================================
# 4. BUILD THE GRAPH
# =========================================================================

def build_graph(models, llm):
    graph = StateGraph(CaseState)
    graph.add_node("gather_evidence", lambda s: node_gather_evidence(s, models))
    graph.add_node("investigate", lambda s: node_investigate(s, llm))
    graph.add_node("write_report", lambda s: node_write_report(s, llm))

    graph.set_entry_point("gather_evidence")
    graph.add_edge("gather_evidence", "investigate")
    graph.add_edge("investigate", "write_report")
    graph.add_edge("write_report", END)

    return graph.compile()


# =========================================================================
# 5. FIND A REAL WRONG PREDICTION TO INVESTIGATE, AND RUN THE AGENT
# =========================================================================

def find_a_wrong_prediction(models, prefer="false_negative"):
    X_test_s = models["scaler"].transform(models["X_test"])
    probs = models["svm_model"].predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    actual = models["y_test"].values

    if prefer == "false_negative":
        mask = (preds == 0) & (actual == 1)
    else:
        mask = (preds == 1) & (actual == 0)

    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return int(np.where(preds != actual)[0][0])
    return int(idxs[0])


if __name__ == "__main__":
    print("Training models...")
    models = train_models()

    if HAS_ANTHROPIC_KEY:
        print("ANTHROPIC_API_KEY found -- using Claude (claude-sonnet-4-6) for investigation.\n")
        llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1000)
    elif HAS_GROQ_KEY:
        print("GROQ_API_KEY found -- using Groq (llama-3.3-70b-versatile) for investigation.\n")
        llm = ChatGroq(model="llama-3.3-70b-versatile", max_tokens=1000)
    else:
        print("No ANTHROPIC_API_KEY or GROQ_API_KEY set -- running with deterministic fallback reasoning.")
        print("(Set one of those environment variables and rerun to see the real LLM-powered investigation.)\n")
        llm = None

    case_idx = find_a_wrong_prediction(models, prefer="false_negative")
    print(f"Investigating test-sample #{case_idx} (a confirmed false negative)...\n")

    graph = build_graph(models, llm)
    result = graph.invoke({"sample_idx": case_idx})

    print("=" * 70)
    print("CASE FILE")
    print("=" * 70)
    print(result["report"])
    print("=" * 70)

    with open("case_file_output.json", "w") as f:
        json.dump({k: v for k, v in result.items()}, f, indent=2, default=str)
    print("\nFull case file (with all evidence) saved to case_file_output.json")
