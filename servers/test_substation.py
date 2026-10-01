import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SUBSTATION_ID"] = "SUB-A"
os.environ["PORT"] = "8001"

from fastapi.testclient import TestClient
from substation_service import app

client = TestClient(app)

print("=== /health ===")
r = client.get("/health")
print(r.status_code, r.json())
assert r.status_code == 200
assert r.json()["substation_id"] == "SUB-A"

print("\n=== /predict ===")
import numpy as np
rng = np.random.default_rng(0)
base = np.array([3, 3, 3, 3, 3.5, -1.1, -1.1, -1.3, 0.5, 0.5, 0.5, 0.5])
window = (base[None, :] + rng.normal(0, 0.3, (10, 12))).tolist()
r = client.post("/predict", json={"readings": window})
print(r.status_code, json.dumps(r.json(), indent=2))
assert r.status_code == 200
body = r.json()
assert body["predicted_label"] in ("stable", "unstable")
assert "filtered_reading" in body
assert len(body["filtered_reading"]) == 12

print("\nALL SUBSTATION SERVICE TESTS PASSED")
