#!/usr/bin/env python3
"""V2 training — optimized for speed. Uses subsample for CV, full data for final fit."""

import os, sys, time, json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load features
print("Loading features...")
feature_df = pd.read_csv(os.path.join(DATA_DIR, "processed_features_v2.csv"), low_memory=False)
feature_df["date"] = pd.to_datetime(feature_df["date"])

from src.features import get_feature_columns
feature_cols = get_feature_columns()

# Filter
key_features = ["elo_diff", "surface_elo_diff", "rank_diff"]
mask = feature_df[key_features].notna().all(axis=1)
df = feature_df[mask].copy()
df[feature_cols] = df[feature_cols].fillna(0)

train = df[(df["date"] >= "2005-01-01") & (df["date"] < "2024-01-01")]
test = df[df["date"] >= "2024-01-01"]

X_train = train[feature_cols].values.astype(np.float32)
y_train = train["target"].values.astype(int)
X_test = test[feature_cols].values.astype(np.float32)
y_test = test["target"].values.astype(int)

print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")

# Use last 20k samples for faster CV (still respects temporal order)
CV_SIZE = 20000
X_cv = X_train[-CV_SIZE:]
y_cv = y_train[-CV_SIZE:]
tscv = TimeSeriesSplit(n_splits=3)

results = []

# ---- XGBoost Optuna ----
print(f"\n=== XGBoost Optuna (50 trials, CV on last {CV_SIZE} samples) ===")

def xgb_objective(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0, 2.0),
    }
    scores = []
    for ti, vi in tscv.split(X_cv):
        m = XGBClassifier(**p, random_state=42, eval_metric="logloss", n_jobs=2,
                          early_stopping_rounds=20, tree_method="hist")
        m.fit(X_cv[ti], y_cv[ti], eval_set=[(X_cv[vi], y_cv[vi])], verbose=False)
        scores.append(accuracy_score(y_cv[vi], m.predict(X_cv[vi])))
    return np.mean(scores)

study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
t0 = time.time()
study_xgb.optimize(xgb_objective, n_trials=50, timeout=180)
print(f"  Completed {len(study_xgb.trials)} trials in {time.time()-t0:.0f}s")
print(f"  Best CV: {study_xgb.best_value:.4f}")
print(f"  Best params: {study_xgb.best_params}")

# Final model on FULL training data
xgb_best = XGBClassifier(**study_xgb.best_params, random_state=42, eval_metric="logloss",
                          n_jobs=-1, tree_method="hist")
xgb_best.fit(X_train, y_train)

y_pred = xgb_best.predict(X_test)
y_prob = xgb_best.predict_proba(X_test)[:, 1]
xgb_acc = accuracy_score(y_test, y_pred)
xgb_auc = roc_auc_score(y_test, y_prob)
xgb_ll = log_loss(y_test, y_prob)
print(f"  TEST — Accuracy: {xgb_acc:.4f}, AUC: {xgb_auc:.4f}, LogLoss: {xgb_ll:.4f}")
results.append({"name": "XGBoost (Optuna)", "accuracy": xgb_acc, "roc_auc": xgb_auc, "log_loss": xgb_ll})

# ---- LightGBM Optuna ----
print(f"\n=== LightGBM Optuna (50 trials) ===")

def lgb_objective(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 15, 80),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
    }
    scores = []
    for ti, vi in tscv.split(X_cv):
        m = lgb.LGBMClassifier(**p, random_state=42, verbose=-1, n_jobs=2)
        m.fit(X_cv[ti], y_cv[ti], eval_set=[(X_cv[vi], y_cv[vi])],
              callbacks=[lgb.early_stopping(20, verbose=False)])
        scores.append(accuracy_score(y_cv[vi], m.predict(X_cv[vi])))
    return np.mean(scores)

study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
t0 = time.time()
study_lgb.optimize(lgb_objective, n_trials=50, timeout=180)
print(f"  Completed {len(study_lgb.trials)} trials in {time.time()-t0:.0f}s")
print(f"  Best CV: {study_lgb.best_value:.4f}")
print(f"  Best params: {study_lgb.best_params}")

lgb_best = lgb.LGBMClassifier(**study_lgb.best_params, random_state=42, verbose=-1, n_jobs=-1)
lgb_best.fit(X_train, y_train)

y_pred = lgb_best.predict(X_test)
y_prob = lgb_best.predict_proba(X_test)[:, 1]
lgb_acc = accuracy_score(y_test, y_pred)
lgb_auc = roc_auc_score(y_test, y_prob)
lgb_ll = log_loss(y_test, y_prob)
print(f"  TEST — Accuracy: {lgb_acc:.4f}, AUC: {lgb_auc:.4f}, LogLoss: {lgb_ll:.4f}")
results.append({"name": "LightGBM (Optuna)", "accuracy": lgb_acc, "roc_auc": lgb_auc, "log_loss": lgb_ll})

# ---- Stacking Ensemble ----
print("\n=== Stacking Ensemble (XGBoost + LightGBM → LogisticRegression) ===")
stack = StackingClassifier(
    estimators=[("xgb", xgb_best), ("lgb", lgb_best)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3, passthrough=True, n_jobs=-1,
)
t0 = time.time()
stack.fit(X_train, y_train)
print(f"  Fit time: {time.time()-t0:.0f}s")

y_pred = stack.predict(X_test)
y_prob = stack.predict_proba(X_test)[:, 1]
st_acc = accuracy_score(y_test, y_pred)
st_auc = roc_auc_score(y_test, y_prob)
st_ll = log_loss(y_test, y_prob)
print(f"  TEST — Accuracy: {st_acc:.4f}, AUC: {st_auc:.4f}, LogLoss: {st_ll:.4f}")
results.append({"name": "Stacking Ensemble", "accuracy": st_acc, "roc_auc": st_auc, "log_loss": st_ll})

# ---- Save everything ----
best_result = max(results, key=lambda x: x["accuracy"])
best_models = {"XGBoost (Optuna)": xgb_best, "LightGBM (Optuna)": lgb_best, "Stacking Ensemble": stack}
best_model = best_models[best_result["name"]]

joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
joblib.dump(xgb_best, os.path.join(MODELS_DIR, "xgboost_optuna.joblib"))
joblib.dump(lgb_best, os.path.join(MODELS_DIR, "lightgbm_optuna.joblib"))
joblib.dump(stack, os.path.join(MODELS_DIR, "stacking_ensemble.joblib"))
joblib.dump(study_xgb, os.path.join(MODELS_DIR, "xgb_optuna_study.joblib"))
joblib.dump(study_lgb, os.path.join(MODELS_DIR, "lgb_optuna_study.joblib"))

with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_cols, f)

# Feature importance
fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb_best.feature_importances_,
}).sort_values("importance", ascending=False)
fi.to_csv(os.path.join(OUTPUTS_DIR, "feature_importance_v2.csv"), index=False)

# Save comparison
pd.DataFrame(results).to_csv(os.path.join(OUTPUTS_DIR, "model_comparison_v2.csv"), index=False)

# Save report
with open(os.path.join(OUTPUTS_DIR, "training_report_v2.txt"), "w") as f:
    f.write("=" * 60 + "\n")
    f.write("TENNIS MATCH PREDICTION V2 - TRAINING REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training: 2005-2023 ({len(X_train)} matches)\n")
    f.write(f"Test: 2024 ({len(X_test)} matches)\n")
    f.write(f"Features: {len(feature_cols)} (64 total)\n\n")
    f.write("V2 IMPROVEMENTS:\n")
    f.write("  - Dynamic K-factor ELO (FiveThirtyEight style)\n")
    f.write("  - Margin of Victory ELO adjustment\n")
    f.write("  - WElo momentum (hot-hand effect)\n")
    f.write("  - Inactivity decay\n")
    f.write("  - Fatigue features (days since last match, recent match load)\n")
    f.write("  - Surface-specific win rates\n")
    f.write("  - Tournament history\n")
    f.write("  - 2nd serve stats + double fault %\n")
    f.write("  - Retirement flag\n")
    f.write("  - Optuna Bayesian hyperparameter tuning\n")
    f.write("  - LightGBM + Stacking Ensemble\n")
    f.write("  - Training on modern era only (2005+)\n\n")
    
    for r in results:
        star = " ★ BEST" if r["name"] == best_result["name"] else ""
        f.write(f"--- {r['name']}{star} ---\n")
        f.write(f"Accuracy: {r['accuracy']:.4f}\n")
        f.write(f"ROC-AUC:  {r['roc_auc']:.4f}\n")
        f.write(f"Log Loss: {r['log_loss']:.4f}\n\n")
    
    f.write(f"\nV1 baseline: 65.5% accuracy (flat K=32 ELO, GridSearch XGBoost, 43 features)\n")
    f.write(f"V2 best:     {best_result['accuracy']:.1%} accuracy ({best_result['name']})\n")
    f.write(f"\nXGBoost best params:\n{json.dumps(study_xgb.best_params, indent=2)}\n")
    f.write(f"\nLightGBM best params:\n{json.dumps(study_lgb.best_params, indent=2)}\n")
    f.write(f"\nTop 10 features by importance:\n")
    for _, row in fi.head(10).iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

# Print summary
print(f"\n{'='*60}")
print("V2 FINAL RESULTS:")
print(f"{'='*60}")
for r in results:
    marker = " ★" if r["name"] == best_result["name"] else ""
    print(f"  {r['name']:<25} Acc: {r['accuracy']:.4f}  AUC: {r['roc_auc']:.4f}  LL: {r['log_loss']:.4f}{marker}")
print(f"\n  V1 baseline:             Acc: 0.6554  AUC: 0.7216  LL: 0.6096")
print(f"\n  Improvement:             +{(best_result['accuracy'] - 0.6554)*100:.1f} percentage points")
print(f"{'='*60}")

print(f"\nTop 10 features:")
for _, row in fi.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")
