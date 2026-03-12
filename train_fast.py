#!/usr/bin/env python3
"""Fastest possible V2 training — 25 Optuna trials, 2-fold CV, 10k sample."""
import os, sys, time, json
import numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
from src.features import get_feature_columns

print("Loading features...")
df = pd.read_csv(os.path.join(DATA_DIR, "processed_features_v2.csv"), low_memory=False)
df["date"] = pd.to_datetime(df["date"])
feature_cols = get_feature_columns()
df = df[df[["elo_diff", "surface_elo_diff", "rank_diff"]].notna().all(axis=1)].copy()
df[feature_cols] = df[feature_cols].fillna(0)

train = df[(df["date"] >= "2005-01-01") & (df["date"] < "2024-01-01")]
test = df[df["date"] >= "2024-01-01"]
X_train = train[feature_cols].values.astype(np.float32)
y_train = train["target"].values.astype(int)
X_test = test[feature_cols].values.astype(np.float32)
y_test = test["target"].values.astype(int)
print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")

# CV on small recent slice
X_cv, y_cv = X_train[-10000:], y_train[-10000:]
tscv = TimeSeriesSplit(n_splits=2)
results = []

# ===== XGBoost Optuna =====
print(f"\n=== XGBoost Optuna (25 trials) ===")
def xgb_obj(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 3.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 12),
    }
    sc = []
    for ti, vi in tscv.split(X_cv):
        m = XGBClassifier(**p, random_state=42, eval_metric="logloss", n_jobs=2,
                          early_stopping_rounds=10, tree_method="hist")
        m.fit(X_cv[ti], y_cv[ti], eval_set=[(X_cv[vi], y_cv[vi])], verbose=False)
        sc.append(accuracy_score(y_cv[vi], m.predict(X_cv[vi])))
    return np.mean(sc)

study_x = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
t0 = time.time()
study_x.optimize(xgb_obj, n_trials=25, timeout=120)
print(f"  {len(study_x.trials)} trials in {time.time()-t0:.0f}s | Best CV: {study_x.best_value:.4f}")

xgb_best = XGBClassifier(**study_x.best_params, random_state=42, eval_metric="logloss", n_jobs=-1, tree_method="hist")
xgb_best.fit(X_train, y_train)
yp, yb = xgb_best.predict(X_test), xgb_best.predict_proba(X_test)[:,1]
xa, xu, xl = accuracy_score(y_test,yp), roc_auc_score(y_test,yb), log_loss(y_test,yb)
print(f"  TEST: Acc={xa:.4f}  AUC={xu:.4f}  LL={xl:.4f}")
results.append({"name":"XGBoost (Optuna)","accuracy":xa,"roc_auc":xu,"log_loss":xl})

# ===== LightGBM Optuna =====
print(f"\n=== LightGBM Optuna (25 trials) ===")
def lgb_obj(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 3.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
    }
    sc = []
    for ti, vi in tscv.split(X_cv):
        m = lgb.LGBMClassifier(**p, random_state=42, verbose=-1, n_jobs=2)
        m.fit(X_cv[ti], y_cv[ti], eval_set=[(X_cv[vi], y_cv[vi])],
              callbacks=[lgb.early_stopping(10, verbose=False)])
        sc.append(accuracy_score(y_cv[vi], m.predict(X_cv[vi])))
    return np.mean(sc)

study_l = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
t0 = time.time()
study_l.optimize(lgb_obj, n_trials=25, timeout=120)
print(f"  {len(study_l.trials)} trials in {time.time()-t0:.0f}s | Best CV: {study_l.best_value:.4f}")

lgb_best = lgb.LGBMClassifier(**study_l.best_params, random_state=42, verbose=-1, n_jobs=-1)
lgb_best.fit(X_train, y_train)
yp, yb = lgb_best.predict(X_test), lgb_best.predict_proba(X_test)[:,1]
la, lu, ll = accuracy_score(y_test,yp), roc_auc_score(y_test,yb), log_loss(y_test,yb)
print(f"  TEST: Acc={la:.4f}  AUC={lu:.4f}  LL={ll:.4f}")
results.append({"name":"LightGBM (Optuna)","accuracy":la,"roc_auc":lu,"log_loss":ll})

# ===== Stacking =====
print(f"\n=== Stacking Ensemble ===")
stack = StackingClassifier(
    estimators=[("xgb", xgb_best), ("lgb", lgb_best)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=2, passthrough=True, n_jobs=-1,
)
t0=time.time()
stack.fit(X_train, y_train)
print(f"  Fit: {time.time()-t0:.0f}s")
yp, yb = stack.predict(X_test), stack.predict_proba(X_test)[:,1]
sa, su, sl = accuracy_score(y_test,yp), roc_auc_score(y_test,yb), log_loss(y_test,yb)
print(f"  TEST: Acc={sa:.4f}  AUC={su:.4f}  LL={sl:.4f}")
results.append({"name":"Stacking Ensemble","accuracy":sa,"roc_auc":su,"log_loss":sl})

# ===== Save =====
best_r = max(results, key=lambda x: x["accuracy"])
models_map = {"XGBoost (Optuna)": xgb_best, "LightGBM (Optuna)": lgb_best, "Stacking Ensemble": stack}
joblib.dump(models_map[best_r["name"]], os.path.join(MODELS_DIR, "best_model.joblib"))
joblib.dump(xgb_best, os.path.join(MODELS_DIR, "xgboost_optuna.joblib"))
joblib.dump(lgb_best, os.path.join(MODELS_DIR, "lightgbm_optuna.joblib"))
joblib.dump(stack, os.path.join(MODELS_DIR, "stacking_ensemble.joblib"))
joblib.dump(study_x, os.path.join(MODELS_DIR, "xgb_optuna_study.joblib"))
joblib.dump(study_l, os.path.join(MODELS_DIR, "lgb_optuna_study.joblib"))
with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_cols, f)

fi = pd.DataFrame({"feature":feature_cols,"importance":xgb_best.feature_importances_}).sort_values("importance",ascending=False)
fi.to_csv(os.path.join(OUTPUTS_DIR, "feature_importance_v2.csv"), index=False)
pd.DataFrame(results).to_csv(os.path.join(OUTPUTS_DIR, "model_comparison_v2.csv"), index=False)

with open(os.path.join(OUTPUTS_DIR, "training_report_v2.txt"), "w") as f:
    f.write("="*60+"\nTENNIS V2 TRAINING REPORT\n"+"="*60+"\n\n")
    f.write(f"Train: 2005-2023 ({len(X_train)} matches)\nTest: 2024 ({len(X_test)} matches)\nFeatures: {len(feature_cols)}\n\n")
    f.write("V2 IMPROVEMENTS:\n- Dynamic K-factor ELO\n- Margin of Victory\n- WElo momentum\n- Inactivity decay\n")
    f.write("- Fatigue features\n- Surface win rates\n- Tournament history\n- 2nd serve/DF stats\n")
    f.write("- Optuna Bayesian tuning\n- LightGBM + Stacking\n- Modern era training (2005+)\n\n")
    for r in results:
        star = " ★" if r["name"]==best_r["name"] else ""
        f.write(f"--- {r['name']}{star} ---\nAcc: {r['accuracy']:.4f} | AUC: {r['roc_auc']:.4f} | LL: {r['log_loss']:.4f}\n\n")
    f.write(f"V1 baseline: 0.6554 acc\nV2 best: {best_r['accuracy']:.4f} acc ({best_r['name']})\n")
    f.write(f"\nXGB params: {json.dumps(study_x.best_params, indent=2)}\nLGB params: {json.dumps(study_l.best_params, indent=2)}\n")
    f.write(f"\nTop 10 features:\n")
    for _, r in fi.head(10).iterrows():
        f.write(f"  {r['feature']}: {r['importance']:.4f}\n")

print(f"\n{'='*60}")
print("V2 FINAL RESULTS:")
print(f"{'='*60}")
for r in results:
    m = " ★" if r["name"]==best_r["name"] else ""
    print(f"  {r['name']:<25} Acc={r['accuracy']:.4f}  AUC={r['roc_auc']:.4f}  LL={r['log_loss']:.4f}{m}")
print(f"\n  V1 baseline:             Acc=0.6554  AUC=0.7216  LL=0.6096")
print(f"  Improvement:             {(best_r['accuracy']-0.6554)*100:+.1f} pp")
print(f"{'='*60}")
print(f"\nTop 5 features:")
for _, r in fi.head(5).iterrows():
    print(f"  {r['feature']}: {r['importance']:.4f}")
