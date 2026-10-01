"""
substation_service.py — one sub-distribution unit, running as its own server
================================================================================
This is the EDGE service: exactly the prediction path from predict_live.py
(median_filter -> trained SVM -> cost-optimal threshold), wrapped as its own
FastAPI server so it can run independently, on its own port, as its own
process -- one per real sub-distribution unit.

Run several of these (different SUBSTATION_ID / PORT) to simulate multiple
substations, each with its own local model instance, matching the edge
architecture discussed in review: prediction stays local and fast; only
the investigation layer (detective_agent/) is centralized.

Usage:
    SUBSTATION_ID=SUB-A PORT=8001 python3 substation_service.py
    SUBSTATION_ID=SUB-B PORT=8002 python3 substation_service.py
    SUBSTATION_ID=SUB-C PORT=8003 python3 substation_service.py
"""

import os
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from median_filter import median_filter
from predict_live import train_svm, load_cost_optimal_threshold, FEATURES

SUBSTATION_ID = os.environ.get("SUBSTATION_ID", "SUB-A")
PORT = int(os.environ.get("PORT", "8001"))

app = FastAPI(title=f"Substation {SUBSTATION_ID}")

# Trained once at startup -- this is the LOCAL model instance for this unit.
# In a real deployment each substation would have its own model artifact,
# possibly retrained/calibrated on that unit's own historical data.
print(f"[{SUBSTATION_ID}] Training local SVM...")
_svm_model, _scaler = train_svm()
_threshold = load_cost_optimal_threshold()
print(f"[{SUBSTATION_ID}] Ready. Cost-optimal threshold = {_threshold:.3f}")


class NoisyWindow(BaseModel):
    # window_size x 12 raw noisy readings for this one grid state
    readings: List[List[float]]


@app.get("/health")
def health():
    return {"substation_id": SUBSTATION_ID, "status": "ok", "threshold": _threshold}


@app.post("/predict")
def predict(payload: NoisyWindow):
    readings = np.array(payload.readings)  # (window, 12)
    filtered = median_filter(readings[None, :, :])  # -> (1, 12)
    filtered_df = pd.DataFrame(filtered, columns=FEATURES)
    prob = float(_svm_model.predict_proba(_scaler.transform(filtered_df))[0, 1])
    predicted_label = "unstable" if prob >= _threshold else "stable"
    return {
        "substation_id": SUBSTATION_ID,
        "predicted_label": predicted_label,
        "probability_unstable": prob,
        "threshold_used": _threshold,
        "filtered_reading": {f: float(v) for f, v in zip(FEATURES, filtered[0])},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
