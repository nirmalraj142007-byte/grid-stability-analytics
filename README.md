# Grid Stability Analytics — Full Project

Predictive load-shedding & grid stability classification, on the UCI
Electrical Grid Stability Simulated Dataset — cost-sensitive ML, SHAP
explainability, noise robustness testing, and an LLM-based error
investigator.

## What's actually in here, and how it connects

```
grid_stability/
│
├── Data_for_UCI_named.csv          Source data (10,000 simulated grid states)
├── pipeline.py                     STAGE 1 — train the model
│                                   Cost-sensitive XGBoost + SVM, threshold
│                                   tuning, SHAP explainability. Produces
│                                   metrics_summary.json and all the
│                                   shap_*.png / *.json outputs below it.
│
├── median_filter.py                STAGE 2 — clean noisy input
│                                   The filter chosen in noise_robustness/
│                                   after testing 5 candidates. Used by
│                                   BOTH predict_live.py and detective_agent/.
│
├── predict_live.py                 STAGE 3 — the actual integrated pipeline
│                                   noisy readings -> median_filter() ->
│                                   trained SVM -> cost-optimal threshold
│                                   (read from metrics_summary.json) ->
│                                   STABLE / UNSTABLE decision.
│                                   THIS is what a real deployment would run.
│
├── orchestrator.py                  STAGE 3.5 — connects prediction to investigation
│                                    Simulates a stream of readings, and the
│                                    moment a prediction is later found wrong,
│                                    AUTOMATICALLY invokes detective_agent/ on
│                                    that exact case — no human has to notice
│                                    the error first. See "the automatic
│                                    trigger" note below for what this does
│                                    and does not simulate honestly.
│
├── noise_robustness/                Evidence for why median_filter.py
│   ├── noise_experiment.py          was chosen over 4 alternatives
│   ├── noise_experiment_results.json (moving average, z-score reject,
│   └── noise_robustness_comparison.png  Kalman filter, no filter at all).
│
└── detective_agent/                 STAGE 4 — after-the-fact investigation
    ├── detective_agent.py           LangGraph agent that investigates a
    │                                CONFIRMED wrong prediction: pulls SHAP
    │                                evidence + re-runs median_filter() on
    │                                that case to check for sensor noise,
    │                                then an LLM reasons about the likely
    │                                root cause and writes a plain-language
    │                                case file for a human reviewer.
    └── SKILLS.md                    Honest spec of what the agent can and
                                     cannot do — read this before presenting
                                     the agent as more capable than it is.
│
└── servers/                         STAGE 5 — multi-substation deployment
    ├── substation_service.py        FastAPI edge service — one per real
    │                                sub-distribution unit, runs the
    │                                prediction path locally (fast, no
    │                                dependency on the central service).
    ├── central_service.py           FastAPI central service — the
    │                                "electricity department" view.
    │                                Aggregates predictions across all
    │                                substations, detects mismatches,
    │                                auto-triggers detective_agent/, and
    │                                serves the dashboard.
    ├── dashboard.html                Browser dashboard for the central
    │                                service — substation health, live
    │                                prediction stream, investigated cases.
    ├── test_substation.py            In-process endpoint tests (see
    ├── test_central_logic.py         servers/README.md for why, and what
    │                                was/wasn't verified live).
    └── README.md                     How to run the 4-server deployment,
                                     plus an honest note on testing limits.
```

## The automatic trigger — what's real and what's simulated

`orchestrator.py` genuinely connects `predict_live.py`'s prediction path to
`detective_agent/`'s investigation path — a wrong prediction really does
auto-trigger the agent, with the agent investigating the SAME noisy
reading and prediction that actually caused the error (not a fresh,
cleaner recomputation — an earlier version of this had that bug; it's
fixed now, see the code comments in `node_gather_evidence`).

**What's simulated, honestly:** the step where "time passes and the true
outcome becomes known." In `orchestrator.py`, that outcome is the already-
known test-set label, revealed immediately. In a real deployment, this
would require an actual delayed feedback channel from live grid
monitoring — confirming, days or minutes later, whether the grid really
did become unstable — and that channel does not exist in this project.

## The design principle, in one sentence

**Everything on the real-time decision path (`predict_live.py`, using
`median_filter.py` and the trained model) is deterministic, fast code.
The LLM/agent (`detective_agent/`) sits entirely AFTER that path, only
runs on confirmed errors, and only ever produces a report for a human —
it never touches the live prediction, never retrains anything, and never
acts autonomously.**

This mirrors the reasoning already used elsewhere in this line of work:
deterministic logic for anything safety- or time-critical, AI layered on
top for explanation and investigation where judgment genuinely helps.

## How to run each stage

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn langchain langgraph langchain-anthropic

# Stage 1 — train the model, generate SHAP analysis (takes ~30-60s)
python3 pipeline.py

# Stage 2 — see the noise-robustness comparison that justified the filter choice
python3 noise_robustness/noise_experiment.py

# Stage 3 — the actual integrated prediction pipeline (run this to see the
# full noisy-input -> filter -> model -> decision flow in one place)
python3 predict_live.py

# Stage 3.5 — the full connected demo: stream of readings, automatic
# agent trigger on wrong predictions, case files written to case_files/
python3 orchestrator.py

# Stage 4 — investigate a single wrong prediction standalone (works without
# an API key using a deterministic fallback; set ANTHROPIC_API_KEY or
# GROQ_API_KEY for the real LLM version — see detective_agent/SKILLS.md)
export ANTHROPIC_API_KEY="your-key-here"   # or GROQ_API_KEY
cd detective_agent && python3 detective_agent.py

# Stage 5 — the full multi-substation deployment with a live dashboard
# (see servers/README.md for details, including honest testing notes)
cd ../servers
pip install fastapi uvicorn httpx
SUBSTATION_ID=SUB-A PORT=8001 python3 substation_service.py &
SUBSTATION_ID=SUB-B PORT=8002 python3 substation_service.py &
SUBSTATION_ID=SUB-C PORT=8003 python3 substation_service.py &
python3 central_service.py
# then open http://localhost:8000/
```

## What this project honestly is, and isn't

**Is:** a trained, validated, cost-sensitive classifier with explainability
(SHAP), a tested noise-handling layer (median filter, benchmarked against
4 alternatives), an integrated end-to-end prediction script, and a working
LangGraph agent for post-hoc error investigation.

**Isn't:** connected to any real grid, SCADA system, or physical sensor —
`predict_live.py` uses a simulated noisy reading as a demo. It also does
not shed load or take any physical action — it classifies and explains,
nothing more. Both are genuine next phases of work, not things this
project claims to already do.
