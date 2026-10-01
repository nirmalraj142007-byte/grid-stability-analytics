"""
central_service.py — "Electricity Department" central server
=================================================================
This is the CENTRAL service in the edge/central split discussed in review:
prediction stays local at each substation (substation_service.py); this
service only aggregates results across substations, detects mismatches,
and triggers the detective agent -- exactly the orchestrator.py logic,
now generalized across MULTIPLE independent substation servers instead
of one in-process model.

Run substations first, then this:
    SUBSTATION_ID=SUB-A PORT=8001 python3 substation_service.py &
    SUBSTATION_ID=SUB-B PORT=8002 python3 substation_service.py &
    SUBSTATION_ID=SUB-C PORT=8003 python3 substation_service.py &
    python3 central_service.py

Then open http://localhost:8000/ for the dashboard, or POST /api/run-cycle
to simulate a batch of readings flowing through all substations.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "detective_agent"))

from predict_live import FEATURES
from detective_agent import train_models, build_graph, HAS_ANTHROPIC_KEY, HAS_GROQ_KEY

SUBSTATIONS = [
    {"id": "SUB-A", "url": "http://localhost:8001"},
    {"id": "SUB-B", "url": "http://localhost:8002"},
    {"id": "SUB-C", "url": "http://localhost:8003"},
]

app = FastAPI(title="Electricity Department -- Central Monitoring")

print("[central] Training investigation models (SVM + XGBoost + SHAP)...")
_agent_models = train_models()


def _select_llm():
    if HAS_ANTHROPIC_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1000)
    if HAS_GROQ_KEY:
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", max_tokens=1000)
    return None


_graph = build_graph(_agent_models, _select_llm())
_feature_std = _agent_models["X_test"][FEATURES].std().values

# In-memory state -- this is a demo/simulation server, not persistent storage
STATE = {
    "stream_log": [],   # every prediction made, across all substations
    "case_files": [],   # every mismatch investigated by the agent
    "next_idx": 0,       # cursor into the test set, simulating a continuous feed
}


def simulate_noisy_window(true_row, rng, window=10):
    noise = rng.normal(0, 0.2 * _feature_std, size=(window, len(FEATURES)))
    readings = true_row[None, :] + noise
    if rng.random() < 0.15:
        readings[rng.integers(0, window)] += rng.normal(0, 1.2 * _feature_std, size=len(FEATURES))
    return readings


class RunCycleRequest(BaseModel):
    n: int = 10
    seed: int = 0


@app.get("/api/health")
def health():
    return {"status": "ok", "substations": [s["id"] for s in SUBSTATIONS]}


@app.get("/api/substation-status")
def substation_status():
    results = []
    for sub in SUBSTATIONS:
        try:
            r = httpx.get(f"{sub['url']}/health", timeout=2.0)
            results.append({**sub, "reachable": True, **r.json()})
        except Exception as e:
            results.append({**sub, "reachable": False, "error": str(e)})
    return results


@app.post("/api/run-cycle")
def run_cycle(req: RunCycleRequest):
    rng = np.random.default_rng(req.seed)
    triggered = []

    for _ in range(req.n):
        idx = STATE["next_idx"] % len(_agent_models["X_test"])
        STATE["next_idx"] += 1

        true_row = _agent_models["X_test"].iloc[idx][FEATURES].values.astype(float)
        actual_label = "unstable" if _agent_models["y_test"].iloc[idx] == 1 else "stable"

        # round-robin across substations, simulating readings arriving from
        # different sub-distribution units
        sub = SUBSTATIONS[idx % len(SUBSTATIONS)]

        noisy_window = simulate_noisy_window(true_row, rng)

        try:
            resp = httpx.post(f"{sub['url']}/predict",
                               json={"readings": noisy_window.tolist()}, timeout=5.0)
            resp.raise_for_status()
            pred = resp.json()
        except Exception as e:
            STATE["stream_log"].append({
                "idx": idx, "substation_id": sub["id"], "error": str(e),
                "actual_label": actual_label,
            })
            continue

        is_wrong = pred["predicted_label"] != actual_label
        entry = {
            "idx": idx,
            "substation_id": sub["id"],
            "predicted_label": pred["predicted_label"],
            "probability_unstable": pred["probability_unstable"],
            "actual_label": actual_label,
            "mismatch": is_wrong,
        }
        STATE["stream_log"].append(entry)

        if is_wrong:
            case = _graph.invoke({
                "sample_idx": idx,
                "external_prediction": {
                    "predicted_label": pred["predicted_label"],
                    "predicted_probability_unstable": pred["probability_unstable"],
                },
                "external_filtered_reading": pred["filtered_reading"],
            })
            case_record = {
                "case_id": f"case_{idx:03d}_{sub['id']}",
                "substation_id": sub["id"],
                **{k: v for k, v in case.items()},
            }
            STATE["case_files"].append(case_record)
            triggered.append(case_record["case_id"])

    return {"ran": req.n, "triggered_investigations": triggered,
            "total_stream_log": len(STATE["stream_log"]),
            "total_case_files": len(STATE["case_files"])}


@app.get("/api/status")
def status():
    total = len(STATE["stream_log"])
    mismatches = sum(1 for e in STATE["stream_log"] if e.get("mismatch"))
    return JSONResponse({
        "substations": SUBSTATIONS,
        "total_predictions": total,
        "total_mismatches": mismatches,
        "accuracy": round(1 - mismatches / total, 4) if total else None,
        "stream_log": STATE["stream_log"][-100:],  # most recent 100
        "case_files": STATE["case_files"],
    })


@app.get("/", response_class=HTMLResponse)
def dashboard():
    dashboard_path = os.path.join(SCRIPT_DIR, "dashboard.html")
    with open(dashboard_path, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
