# Multi-Substation Deployment (FastAPI)

This turns the project from standalone scripts into the edge/central
architecture discussed in review: each sub-distribution unit runs its own
prediction server; a central "electricity department" service aggregates
results, detects mismatches, and triggers the investigation agent.

```
substation_service.py  (run 3x, one per unit, different ports)
        |
        |  POST /predict  {readings: [[...12 values...] x10]}
        v
central_service.py  (aggregates, detects mismatches, triggers agent,
                      serves the dashboard)
        |
        v
dashboard.html  (open http://localhost:8000/ in a browser)
```

## How to run it

```bash
pip install fastapi uvicorn httpx

# Terminal 1, 2, 3 -- one per substation
SUBSTATION_ID=SUB-A PORT=8001 python3 substation_service.py
SUBSTATION_ID=SUB-B PORT=8002 python3 substation_service.py
SUBSTATION_ID=SUB-C PORT=8003 python3 substation_service.py

# Terminal 4 -- the central service
python3 central_service.py
```

Then open **http://localhost:8000/** for the dashboard. Click "Run 10
simulated readings" to feed a batch through all three substations — the
dashboard updates with the prediction stream and, on any mismatch, the
investigated case file.

You can also drive it directly:

```bash
curl -X POST http://localhost:8000/api/run-cycle \
     -H "Content-Type: application/json" -d '{"n": 10, "seed": 1}'
curl http://localhost:8000/api/status
```

## How this was tested — read this before presenting it as "tested"

Running 4 long-lived servers simultaneously and hitting them with `curl`
turned out to be unreliable in the sandbox this was built in — background
processes were killed between tool calls, and networking to `localhost`
was inconsistent. Rather than claim something I couldn't actually verify,
testing was done a different, still-rigorous way:

- **`test_substation.py`** — uses FastAPI's `TestClient` to call
  `substation_service.py`'s real `/health` and `/predict` endpoints
  in-process (same code path as a real HTTP request, no networking layer
  needed). **Passed.**
- **`test_central_logic.py`** — same approach for `central_service.py`,
  with the substations' HTTP responses mocked (since 3 separate live
  servers couldn't be kept running in this sandbox). Verifies the full
  aggregation → mismatch detection → agent trigger pipeline, and
  specifically re-checks the evidence-consistency bug fixed earlier (no
  triggered case should ever report "Correct prediction"). **Passed, 0
  inconsistent cases.**

**What this means, honestly**: the code for every individual endpoint and
the full aggregation logic has been verified to work correctly. What was
NOT verified in this environment is the actual live network communication
between 4 genuinely separate, simultaneously-running OS processes over
real HTTP — that's the one part you should test yourself after unzipping
this, before presenting it as a fully live-tested system. Run the 4
commands above in 4 terminals and confirm the dashboard populates as
expected.
