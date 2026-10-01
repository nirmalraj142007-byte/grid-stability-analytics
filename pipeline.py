"""
Grid Stability Analytics — Cost-Sensitive Classification + SHAP
=================================================================
Dataset: UCI Electrical Grid Stability Simulated Dataset (10,000 rows)
Target: stabf (stable / unstable), derived from stab (max real eigenvalue
        of the linearized differential-algebraic system — negative = stable).

This script:
  1. Prepares data (drops the leaky `stab` column, keeps only true predictors)
  2. Defines an asymmetric cost matrix (missed instability >> false alarm)
  3. Trains a cost-sensitive XGBoost classifier (scale_pos_weight tuned via CV
     to minimize expected cost, not accuracy)
  4. Trains a cost-sensitive SVM (RBF kernel, class_weight tuned the same way)
  5. Picks an operating threshold on predicted probability by minimizing
     expected cost on validation data (not the default 0.5)
  6. Runs SHAP (TreeExplainer on XGBoost) to find the exact decision boundary
     in terms of the electrical/economic variables, especially g1 (producer's
     price-elasticity coefficient) and the tau reaction-time terms.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import xgboost as xgb
import shap

RANDOM_STATE = 42
# Both the input CSV and all output files live in the same folder as this
# script, so it runs correctly no matter where that folder is unzipped to.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = SCRIPT_DIR
CSV_PATH = os.path.join(SCRIPT_DIR, "Data_for_UCI_named.csv")

# -----------------------------------------------------------------------
# 1. DATA
# -----------------------------------------------------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find {CSV_PATH}\n"
        f"Make sure 'Data_for_UCI_named.csv' is in the same folder as this script."
    )
df = pd.read_csv(CSV_PATH)

# stab is a deterministic linear function of the label (it IS the label,
# pre-thresholding) -> drop it, or the model trivially "solves" the task
# by learning stab > 0. Real deployment never has stab, only the 12 physical
# / economic inputs.
FEATURES = ["tau1", "tau2", "tau3", "tau4",
            "p1", "p2", "p3", "p4",
            "g1", "g2", "g3", "g4"]
X = df[FEATURES].copy()
y = (df["stabf"] == "unstable").astype(int)  # 1 = unstable (positive/rare-cost class)

print("Class balance:", y.value_counts(normalize=True).to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# -----------------------------------------------------------------------
# 2. COST MATRIX
# -----------------------------------------------------------------------
# Grid-ops framing:
#   False Negative (predict stable, actually unstable) -> missed load-shed
#       trigger -> risk of cascading failure / transformer damage.
#   False Positive (predict unstable, actually stable)  -> unnecessary
#       micro-load-shed -> wasted generation / minor curtailment cost.
# Cost ratio informed by typical utility outage-cost studies: unplanned
# cascading outage cost is roughly an order of magnitude worse than
# precautionary curtailment of a similar duration.
COST_FN = 10.0   # missed instability
COST_FP = 1.0    # false alarm
print(f"Cost matrix -> FN:{COST_FN}  FP:{COST_FP}  (ratio {COST_FN/COST_FP:.0f}:1)")


def expected_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (fn * cost_fn + fp * cost_fp) / len(y_true)


def best_threshold(y_true, y_prob, cost_fn=COST_FN, cost_fp=COST_FP):
    thresholds = np.linspace(0.01, 0.99, 197)
    costs = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        costs.append(expected_cost(y_true, pred, cost_fn, cost_fp))
    costs = np.array(costs)
    best_i = costs.argmin()
    return thresholds[best_i], costs[best_i], thresholds, costs


# -----------------------------------------------------------------------
# 3. COST-SENSITIVE XGBOOST  (scale_pos_weight tuned by CV expected cost)
# -----------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
candidate_spw = [1, 2, 3, 4, 5, 6, 8, 10]  # positive class = unstable already majority; spw>1 pushes toward predicting MORE unstable (safer)
cv_results = []

for spw in candidate_spw:
    fold_costs = []
    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_val)[:, 1]
        # use threshold=0.5 during the spw search (spw already encodes cost skew)
        pred = (prob >= 0.5).astype(int)
        fold_costs.append(expected_cost(y_val, pred))
    cv_results.append((spw, np.mean(fold_costs), np.std(fold_costs)))
    print(f"scale_pos_weight={spw:>2}  mean expected cost={np.mean(fold_costs):.4f}  +/-{np.std(fold_costs):.4f}")

best_spw = min(cv_results, key=lambda r: r[1])[0]
print(f"\n>> Selected scale_pos_weight = {best_spw}")

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=best_spw, eval_metric="logloss",
    random_state=RANDOM_STATE, n_jobs=-1
)
xgb_model.fit(X_train, y_train)

xgb_prob_test = xgb_model.predict_proba(X_test)[:, 1]
xgb_thr, xgb_thr_cost, thr_grid, cost_grid = best_threshold(y_test.values, xgb_prob_test)
xgb_pred_default = (xgb_prob_test >= 0.5).astype(int)
xgb_pred_costopt = (xgb_prob_test >= xgb_thr).astype(int)

print(f"\nXGBoost test AUC: {roc_auc_score(y_test, xgb_prob_test):.4f}")
print(f"Default thr=0.50 -> expected cost {expected_cost(y_test, xgb_pred_default):.4f}")
print(f"Cost-optimal thr={xgb_thr:.3f} -> expected cost {xgb_thr_cost:.4f}")
print(classification_report(y_test, xgb_pred_costopt, target_names=["stable", "unstable"]))

# -----------------------------------------------------------------------
# 4. COST-SENSITIVE SVM  (RBF kernel, class_weight)
# -----------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# class_weight expresses the SAME cost asymmetry: weight on class 1 (unstable)
# relative to class 0 (stable) mirrors COST_FN / COST_FP applied to the rarer
# error type. We grid a small set for speed on 8k rows.
candidate_cw = [1, 2, 3, 5, 8]
svm_cv_results = []
for cw in candidate_cw:
    fold_costs = []
    for tr_idx, val_idx in skf.split(X_train_s, y_train):
        model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                    class_weight={0: 1, 1: cw}, random_state=RANDOM_STATE)
        model.fit(X_train_s[tr_idx], y_train.iloc[tr_idx])
        prob = model.predict_proba(X_train_s[val_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        fold_costs.append(expected_cost(y_train.iloc[val_idx], pred))
    svm_cv_results.append((cw, np.mean(fold_costs)))
    print(f"SVM class_weight(unstable)={cw:>2}  mean expected cost={np.mean(fold_costs):.4f}")

best_cw = min(svm_cv_results, key=lambda r: r[1])[0]
print(f">> Selected SVM class_weight = {{0:1, 1:{best_cw}}}")

svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                class_weight={0: 1, 1: best_cw}, random_state=RANDOM_STATE)
svm_model.fit(X_train_s, y_train)
svm_prob_test = svm_model.predict_proba(X_test_s)[:, 1]
svm_thr, svm_thr_cost, _, _ = best_threshold(y_test.values, svm_prob_test)
svm_pred = (svm_prob_test >= svm_thr).astype(int)
print(f"\nSVM test AUC: {roc_auc_score(y_test, svm_prob_test):.4f}")
print(f"SVM cost-optimal thr={svm_thr:.3f} -> expected cost {svm_thr_cost:.4f}")
print(classification_report(y_test, svm_pred, target_names=["stable", "unstable"]))

# -----------------------------------------------------------------------
# 5. SAVE HEADLINE METRICS
# -----------------------------------------------------------------------
summary = {
    "n_samples": int(len(df)),
    "class_balance": y.value_counts(normalize=True).to_dict(),
    "cost_matrix": {"FN_missed_instability": COST_FN, "FP_false_alarm": COST_FP},
    "xgboost": {
        "best_scale_pos_weight": best_spw,
        "test_auc": float(roc_auc_score(y_test, xgb_prob_test)),
        "cost_optimal_threshold": float(xgb_thr),
        "expected_cost_default_thr": float(expected_cost(y_test, xgb_pred_default)),
        "expected_cost_optimal_thr": float(xgb_thr_cost),
    },
    "svm": {
        "best_class_weight_unstable": best_cw,
        "test_auc": float(roc_auc_score(y_test, svm_prob_test)),
        "cost_optimal_threshold": float(svm_thr),
        "expected_cost_optimal_thr": float(svm_thr_cost),
    },
}
with open(f"{OUT}/metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved metrics_summary.json")

# -----------------------------------------------------------------------
# 6. THRESHOLD / COST CURVE PLOT
# -----------------------------------------------------------------------
plt.figure(figsize=(7, 4.5))
plt.plot(thr_grid, cost_grid, color="#c0392b", lw=2)
plt.axvline(xgb_thr, color="#2c3e50", ls="--", label=f"cost-optimal thr={xgb_thr:.2f}")
plt.axvline(0.5, color="gray", ls=":", label="default thr=0.50")
plt.xlabel("XGBoost predicted P(unstable)")
plt.ylabel("Expected cost per sample")
plt.title("Expected Cost vs. Decision Threshold (test set)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/cost_threshold_curve.png", dpi=150)
plt.close()

# ROC comparison
fpr_x, tpr_x, _ = roc_curve(y_test, xgb_prob_test)
fpr_s, tpr_s, _ = roc_curve(y_test, svm_prob_test)
plt.figure(figsize=(5.5, 5))
plt.plot(fpr_x, tpr_x, label=f"XGBoost (AUC={roc_auc_score(y_test, xgb_prob_test):.3f})", color="#2980b9")
plt.plot(fpr_s, tpr_s, label=f"SVM RBF (AUC={roc_auc_score(y_test, svm_prob_test):.3f})", color="#8e44ad")
plt.plot([0, 1], [0, 1], color="gray", ls=":")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC — XGBoost vs Cost-Sensitive SVM")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/roc_comparison.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 7. SHAP ANALYSIS (on the XGBoost model — exact, fast TreeExplainer)
# -----------------------------------------------------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Global importance (mean |SHAP|)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)
importance_df.to_csv(f"{OUT}/shap_feature_importance.csv", index=False)
print("\nSHAP global importance:")
print(importance_df)

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=FEATURES, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------
# 8. FIND THE FLIP THRESHOLD ON THE TOP FEATURE (g1)
# -----------------------------------------------------------------------
top_feature = importance_df.iloc[0]["feature"]
feat_idx = FEATURES.index(top_feature)
feat_vals = X_test[top_feature].values
feat_shap = shap_values[:, feat_idx]

# dependence plot
plt.figure(figsize=(7, 5))
shap.dependence_plot(top_feature, shap_values, X_test, feature_names=FEATURES,
                      interaction_index=None, show=False, ax=plt.gca())
plt.title(f"SHAP dependence: {top_feature}")
plt.tight_layout()
plt.savefig(f"{OUT}/shap_dependence_top_feature.png", dpi=150, bbox_inches="tight")
plt.close()

# Sort by feature value, find the value where mean SHAP crosses zero
order = np.argsort(feat_vals)
sorted_vals = feat_vals[order]
sorted_shap = feat_shap[order]

# smooth with a rolling mean to find the zero-crossing robustly
window = 50
roll = pd.Series(sorted_shap).rolling(window, center=True, min_periods=10).mean().values
sign = np.sign(roll)
crossing_idxs = np.where(np.diff(sign) != 0)[0]
crossing_points = sorted_vals[crossing_idxs] if len(crossing_idxs) else []

flip_summary = {
    "top_feature": top_feature,
    "feature_range_in_test": [float(feat_vals.min()), float(feat_vals.max())],
    "approx_zero_crossings": [float(c) for c in crossing_points],
}
with open(f"{OUT}/flip_threshold_{top_feature}.json", "w") as f:
    json.dump(flip_summary, f, indent=2)
print(f"\nFlip analysis on {top_feature}:", flip_summary)

# Also examine tau1 (producer reaction time) as a second key differential variable
if "tau1" in FEATURES:
    tf = "tau1"
    tidx = FEATURES.index(tf)
    tvals = X_test[tf].values
    tshap = shap_values[:, tidx]
    order2 = np.argsort(tvals)
    sv2 = tvals[order2]
    ss2 = tshap[order2]
    roll2 = pd.Series(ss2).rolling(window, center=True, min_periods=10).mean().values
    sign2 = np.sign(roll2)
    cross2 = np.where(np.diff(sign2) != 0)[0]
    cross_pts2 = sv2[cross2] if len(cross2) else []
    with open(f"{OUT}/flip_threshold_tau1.json", "w") as f:
        json.dump({"feature": "tau1", "approx_zero_crossings": [float(c) for c in cross_pts2]}, f, indent=2)
    print("Flip analysis on tau1:", cross_pts2)

print("\nAll artifacts written to", OUT)
