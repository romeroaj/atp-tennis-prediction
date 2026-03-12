#!/usr/bin/env python3
"""Train XGBoost with Optuna only."""
import os, sys, time, json
import numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from xgboost import XGBClassifier
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
from src.features import get_feature_columns

print("Loading features...")
df = pd.read_csv(os.path.join(DATA_DIR, "processed_features_v2.csv"), low_memory=False)
df["date"] = pd.to_datetime(df["date"])
feature_cols = get_feature_columns()
key_f = ["elo_diff", "surface_elo_diff", "rank_diff"]
df = df[df[key_f].notna().all(axis=1)].copy()
df[feature_cols] = df[feature_cols].fillna(0)

train = df[(df["date"] >= "2005-01-01") & (df["date"] < "2024-01-01")]
test = df[df["date"] >= "2024-01-01"]
X_train = train[feature_cols].values.astype(np.float32)
y_train = train["target"].values.astype(int)
X_test = test[feature_cols].values.astype(np.float32)
y_test = test["target"].values.astype(int)
print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")

# CV on last 15k samples for speed
X_cv = X_train[-15000:]
y_cv = y_train[-15000:]
tscv = TimeSeriesSplit(n_splits=3)

def xgb_objective(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
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
                          early_stopping_rounds=15, tree_method="hist")
        m.fit(X_cv[ti], y_cv[ti], eval_set=[(X_cv[vi], y_cv[vi])], verbose=False)
        scores.append(accuracy_score(y_cv[vi], m.predict(X_cv[vi])))
    return np.mean(scores)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
t0 = time.time()
study.optimize(xgb_objective, n_trials=50, timeout=360)
print(f"\nCompleted {len(study.trials)} trials in {time.time()-t0:.0f}s")
print(f"Best CV accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Final model on full data
print("\nTraining final XGBoost on full training set...")
xgb_best = XGBClassifier(**study.best_params, random_state=42, eval_metric="logloss",
                          n_jobs=-1, tree_method="hist")
xgb_best.fit(X_train, y_train)

y_pred = xgb_best.predict(X_test)
y_prob = xgb_best.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
ll = log_loss(y_test, y_prob)
print(f"\nXGBoost TEST: Accuracy={acc:.4f}, AUC={auc:.4f}, LogLoss={ll:.4f}")
print(f"V1 baseline:  Accuracy=0.6554, AUC=0.7216, LogLoss=0.6096")
print(f"Improvement:  {(acc-0.6554)*100:+.1f} pp accuracy, {(auc-0.7216)*100:+.1f} pp AUC")

joblib.dump(xgb_best, os.path.join(MODELS_DIR, "xgboost_optuna.joblib"))
joblib.dump(study, os.path.join(MODELS_DIR, "xgb_optuna_study.joblib"))
joblib.dump({"accuracy": acc, "roc_auc": auc, "log_loss": ll, "params": study.best_params},
            os.path.join(MODELS_DIR, "xgb_results.joblib"))

# Feature importance
fi = pd.DataFrame({"feature": feature_cols, "importance": xgb_best.feature_importances_})
fi = fi.sort_values("importance", ascending=False)
fi.to_csv(os.path.join(OUTPUTS_DIR, "feature_importance_v2.csv"), index=False)
print("\nTop 10 features:")
for _, r in fi.head(10).iterrows():
    print(f"  {r['feature']}: {r['importance']:.4f}")
