"""
Tests central_service.py's core logic WITHOUT needing real running substation
servers -- httpx.post is monkeypatched to return realistic predictions
directly, so we can verify the aggregation / mismatch-detection / agent-
triggering logic in isolation, fast and deterministically.

(This project's actual multi-server deployment is still meant to be run for
real -- see README.md -- this test just validates the logic without
depending on this sandbox's networking, which proved unreliable for
long-running multi-process tests.)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "detective_agent"))

import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

print("Importing central_service (this trains the SVM+XGBoost+SHAP models)...")
import central_service as cs
from median_filter import median_filter
from predict_live import FEATURES

# --- Fake a substation's /predict response, using central's OWN trained SVM
#     as a stand-in for "the substation's local model" (in reality each
#     substation trains its own copy of the same model -- functionally
#     equivalent for this logic test) ---
_svm_model, _scaler = None, None


def fake_train_svm():
    global _svm_model, _scaler
    if _svm_model is None:
        from predict_live import train_svm
        _svm_model, _scaler = train_svm()
    return _svm_model, _scaler


def fake_post(url, json=None, timeout=None):
    svm, scaler = fake_train_svm()
    readings = np.array(json["readings"])
    filtered = median_filter(readings[None, :, :])
    filtered_df = pd.DataFrame(filtered, columns=FEATURES)
    prob = float(svm.predict_proba(scaler.transform(filtered_df))[0, 1])
    label = "unstable" if prob >= 0.11 else "stable"

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "predicted_label": label,
                "probability_unstable": prob,
                "filtered_reading": {f: float(v) for f, v in zip(FEATURES, filtered[0])},
            }
    return FakeResponse()


print("\n=== Testing /api/health ===")
from fastapi.testclient import TestClient
client = TestClient(cs.app)
r = client.get("/api/health")
print(r.status_code, r.json())
assert r.status_code == 200

print("\n=== Running a mocked cycle of 20 readings ===")
with patch("central_service.httpx.post", side_effect=fake_post):
    r = client.post("/api/run-cycle", json={"n": 20, "seed": 5})
print(r.status_code, json.dumps(r.json(), indent=2))
assert r.status_code == 200
result = r.json()

print("\n=== Checking /api/status ===")
r = client.get("/api/status")
status = r.json()
print(f"Total predictions: {status['total_predictions']}")
print(f"Total mismatches: {status['total_mismatches']}")
print(f"Case files generated: {len(status['case_files'])}")

# --- The critical correctness check from the earlier bug fix: every
#     triggered case must actually report a real error type, never
#     "Correct prediction" (that would mean the agent investigated a
#     DIFFERENT event than the one that actually triggered it) ---
bad_cases = [c for c in status["case_files"] if c["error_type"] == "Correct prediction (no error to investigate)"]
assert len(bad_cases) == 0, f"BUG: {len(bad_cases)} case(s) contradict their own trigger!"
assert len(status["case_files"]) == status["total_mismatches"], "every mismatch should have a case file"

# --- Also confirm the case investigates the ACTUAL triggering evidence,
#     not a silently-recomputed clean version ---
for c in status["case_files"]:
    assert c["prediction"]["predicted_label"] != c["ground_truth"]["actual_label"], \
        "case's own recorded prediction should disagree with ground truth"

print("\nALL CENTRAL SERVICE LOGIC TESTS PASSED")
print(f"({len(status['case_files'])} mismatches correctly triggered and internally consistent)")
