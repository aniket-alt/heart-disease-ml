Heart Disease Risk — CRISP-DM End-to-End (EDA → Model → API → Monitoring)

Goal. Build and deploy a calibrated binary classifier that predicts whether a patient has heart disease at assessment time. The system is designed as clinical decision support (triage), not a stand-alone diagnosis.

⚠️ Medical disclaimer: This is a research/education project. Outputs are not medical advice and must not be used for clinical decisions without proper validation and oversight.

Dataset

Source: UCI Heart Disease (Cleveland + extensions), commonly distributed as heart_disease_uci.csv.

Target: Presence of heart disease (binary). If multi-class (0–4 severity), we binarize >0 → 1.

Typical columns (may vary slightly by file):

Numeric: age, trestbps, chol, thalach, oldpeak

Categorical/binary: sex, cp, fbs, restecg, exang, slope, ca, thal

Known quirks:

Placeholder/invalid values treated as missing: trestbps ≤ 0, chol ≤ 0, oldpeak < 0.

High missingness in pathway features: ca (~66%), thal (~53%), slope (~34%) depending on the variant.

Project Workflow (CRISP-DM)
1) Business Understanding

Use case: rule-out support; prioritize further testing/referral.

Success: AUROC ≥ 0.88; operate at Sensitivity ≥ 0.90, good calibration (low Brier/ECE), high NPV.

Fairness: report Sens/Spec/PPV/NPV by sex and age bands; flag gaps >5–7 pp.

2) Data Understanding (EDA)

Prevalence (example from our run): ~55% positive overall.

By sex (example): Female ~26%, Male ~63% disease rate → monitor group metrics.

Strong signals: cp, thalach, oldpeak, exang, and (when available) ca, thal, slope.

Artifacts produced:

Missingness & range checks

Univariate distributions & prevalence bars

Correlations & Mutual Information ranking

Orientation baseline (LogReg) metrics

3) Data Preparation

Numeric: median impute + missing-indicator.

Categorical: mode impute + One-Hot (version-safe).

Two feature variants:

full (includes ca, thal, slope)

no_pathway (drops them; more robust when these are sparse/unavailable)

4) Modeling

Models: Logistic Regression, Random Forest, XGBoost (optional).

CV: randomized search (5-fold, AUROC).

Isotonic calibration on validation (prefit).

Operating points decided on validation:

Sensitivity ≥ 0.90 (maximize specificity subject to that)

Cost-minimizing (example: FN=10, FP=2)

5) Evaluation

Metrics: AUROC, AUPRC, Brier, ECE, calibration slope/intercept.

Plots: ROC, PR, reliability diagram, decision curve.

Slices: performance by sex and age bands at chosen threshold.

Artifacts:

overall_metrics.csv, operating_points.csv, fairness_slices_at_sens.csv

plots/{roc.png, pr.png, reliability.png, decision_curve.png}

model_card.md

Example results from our run (your results may vary):

AUROC ~ 0.93 (LogReg baseline on holdout)

PR AUC ~ 0.92, Brier ~ 0.11

Sensitivity-first threshold ≈ 0.57 with high NPV

6) Deployment (FastAPI)

Artifacts (Drive-friendly):

serve/model/calibrated_pipeline.joblib

serve/model/serve_config.json ← holds threshold

serve/model/features.json

serve/app.py

Endpoints:

GET /health → {status, model, variant}

POST /predict → {probs[], preds[], threshold_used, meta}

POST /predict_proba (same payload; returns probs & decision)

Colab-stable launch: copies app.py + model/ to /content/serve_runtime, runs Uvicorn on localhost (single worker, pure-Python loop for stability).

7) Monitoring & Maintenance

Request logging: JSONL (serve/model/logs/prediction_log.jsonl).

Baseline: training-set distributions → monitoring/baseline.json.

Daily monitor: drift (PSI), predicted positive rate, fairness proxies; writes day-stamped reports + alerts.

Retraining triggers: PSI ≥ 0.25, AUROC drop ≥ 0.03, Sens drop ≥ 5 pp, fairness gap ≥ 7 pp, schema/missingness spikes.

Repository Structure (suggested)
.
├── notebooks/
│   ├── phase2_eda.ipynb
│   ├── phase3_prep.ipynb
│   ├── phase4_modeling.ipynb
│   ├── phase5_evaluation.ipynb
│   └── phase7_monitoring.ipynb
├── serve/
│   ├── app.py
│   └── model/
│       ├── calibrated_pipeline.joblib
│       ├── serve_config.json
│       └── features.json
├── heart_phase5_evaluation/
│   ├── overall_metrics.csv
│   ├── operating_points.csv
│   ├── fairness_slices_at_sens.csv
│   └── plots/ (roc.png, pr.png, reliability.png, decision_curve.png)
├── monitoring/
│   ├── baseline.json
│   └── YYYY-MM-DD/ (summary.csv, drift_psi.csv, fairness_proxy.csv, alerts.txt)
├── Dockerfile
├── requirements.txt
└── README.md


.gitignore (recommended):

# large binaries & temp
*.joblib
*.zip
*.log
__pycache__/
.ipynb_checkpoints/
serve/model/logs/
*.csv

How to Run (Colab)

Mount Drive and ensure your artifacts live under:
/content/drive/My Drive/Heart/serve/model/

Start the server (stable, non-blocking):

We used a “Cell 9” that copies to /content/serve_runtime and launches on 127.0.0.1:8091.

Test:

import requests, json
BASE = "http://127.0.0.1:8091"
print(requests.get(f"{BASE}/health").json())
payload = {"rows":[{"age":57,"sex":1,"cp":2,"trestbps":140,"chol":260,"fbs":0,
                    "restecg":1,"thalach":150,"exang":0,"oldpeak":1.2,"slope":1,"ca":0,"thal":2}]}
print(requests.post(f"{BASE}/predict", json=payload).json())

API Contract

Request

{
  "rows": [ { "<feature>": <value>, ... } ],
  "threshold": 0.72   // optional override
}


Response

{
  "probs": [0.82, ...],
  "preds": [1, ...],
  "threshold_used": 0.72,
  "meta": {"variant":"no_pathway", "model":"logreg"}
}

Key Graphs & Where They Live

heart_phase5_evaluation/plots/roc.png — Discrimination across thresholds

heart_phase5_evaluation/plots/pr.png — Precision–Recall (good for imbalanced data)

heart_phase5_evaluation/plots/reliability.png — Calibration quality

heart_phase5_evaluation/plots/decision_curve.png — Net benefit vs threshold

Insights (from our run)

Simple, regularized Logistic Regression already performs strongly (AUROC ≈ 0.93); Random Forest or XGBoost may add a bit.

Calibration is crucial: isotonic improves probability quality and decision curves.

Pathway features (ca, thal, slope) carry signal but are often missing; keeping a no_pathway variant boosts robustness.

Large sex prevalence gap → always report slice metrics and watch fairness deltas in monitoring.
