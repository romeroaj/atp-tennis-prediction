#!/usr/bin/env python3
"""Lean V2 pipeline — runs in stages to avoid timeout."""

import os, sys, time, json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"

def stage_data():
    """Stage 1: Load data, compute ELO, engineer features."""
    from src.data_loader import load_and_clean
    from src.elo import compute_elo_ratings
    from src.features import engineer_features
    
    print("[1/3] Loading data...")
    matches, players = load_and_clean()
    
    print("[2/3] Computing V2 ELO...")
    matches, overall_elo, surface_elo = compute_elo_ratings(matches)
    
    print("[3/3] Engineering V2 features...")
    feature_df = engineer_features(matches)
    
    # Save
    feature_df.to_csv(os.path.join(DATA_DIR, "processed_features_v2.csv"), index=False)
    joblib.dump(overall_elo, os.path.join(MODELS_DIR, "overall_elo.joblib"))
    joblib.dump(surface_elo, os.path.join(MODELS_DIR, "surface_elo.joblib"))
    joblib.dump(matches, os.path.join(MODELS_DIR, "matches_cache.joblib"))
    joblib.dump(players, os.path.join(MODELS_DIR, "players_cache.joblib"))
    print(f"Features shape: {feature_df.shape}")
    print("Stage DATA complete.")


def stage_train():
    """Stage 2: Train with Optuna."""
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
    from xgboost import XGBClassifier
    import lightgbm as lgb
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from src.features import get_feature_columns
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print("Loading features...")
    feature_df = pd.read_csv(os.path.join(DATA_DIR, "processed_features_v2.csv"), low_memory=False)
    feature_df["date"] = pd.to_datetime(feature_df["date"])
    
    feature_cols = get_feature_columns()
    
    # Filter valid rows
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
    
    results = []
    
    # ---- XGBoost Optuna ----
    print("\n=== XGBoost Optuna (50 trials) ===")
    tscv = TimeSeriesSplit(n_splits=3)
    
    def xgb_objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0, 3.0),
        }
        scores = []
        for ti, vi in tscv.split(X_train):
            m = XGBClassifier(**p, random_state=42, eval_metric="logloss", n_jobs=-1,
                              early_stopping_rounds=30, tree_method="hist")
            m.fit(X_train[ti], y_train[ti], eval_set=[(X_train[vi], y_train[vi])], verbose=False)
            scores.append(accuracy_score(y_train[vi], m.predict(X_train[vi])))
        return np.mean(scores)
    
    study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    t0 = time.time()
    study_xgb.optimize(xgb_objective, n_trials=50)
    print(f"  Time: {time.time()-t0:.0f}s | Best CV: {study_xgb.best_value:.4f}")
    print(f"  Best params: {study_xgb.best_params}")
    
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
    print("\n=== LightGBM Optuna (50 trials) ===")
    
    def lgb_objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        }
        scores = []
        for ti, vi in tscv.split(X_train):
            m = lgb.LGBMClassifier(**p, random_state=42, verbose=-1, n_jobs=-1)
            m.fit(X_train[ti], y_train[ti], eval_set=[(X_train[vi], y_train[vi])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            scores.append(accuracy_score(y_train[vi], m.predict(X_train[vi])))
        return np.mean(scores)
    
    study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    t0 = time.time()
    study_lgb.optimize(lgb_objective, n_trials=50)
    print(f"  Time: {time.time()-t0:.0f}s | Best CV: {study_lgb.best_value:.4f}")
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
    print("\n=== Stacking Ensemble ===")
    stack = StackingClassifier(
        estimators=[("xgb", xgb_best), ("lgb", lgb_best)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=TimeSeriesSplit(n_splits=3),
        passthrough=True, n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    
    y_pred = stack.predict(X_test)
    y_prob = stack.predict_proba(X_test)[:, 1]
    st_acc = accuracy_score(y_test, y_pred)
    st_auc = roc_auc_score(y_test, y_prob)
    st_ll = log_loss(y_test, y_prob)
    print(f"  TEST — Accuracy: {st_acc:.4f}, AUC: {st_auc:.4f}, LogLoss: {st_ll:.4f}")
    results.append({"name": "Stacking Ensemble", "accuracy": st_acc, "roc_auc": st_auc, "log_loss": st_ll})
    
    # ---- Save ----
    best_model = max([(xgb_best, xgb_acc, "XGBoost"), (lgb_best, lgb_acc, "LightGBM"), (stack, st_acc, "Stack")], key=lambda x: x[1])
    joblib.dump(best_model[0], os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(xgb_best, os.path.join(MODELS_DIR, "xgboost_optuna.joblib"))
    joblib.dump(lgb_best, os.path.join(MODELS_DIR, "lightgbm_optuna.joblib"))
    joblib.dump(stack, os.path.join(MODELS_DIR, "stacking_ensemble.joblib"))
    joblib.dump(study_xgb, os.path.join(MODELS_DIR, "xgb_optuna_study.joblib"))
    joblib.dump(study_lgb, os.path.join(MODELS_DIR, "lgb_optuna_study.joblib"))
    
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    
    # Save comparison
    pd.DataFrame(results).to_csv(os.path.join(OUTPUTS_DIR, "model_comparison_v2.csv"), index=False)
    
    # Save report
    with open(os.path.join(OUTPUTS_DIR, "training_report_v2.txt"), "w") as f:
        f.write("=" * 60 + "\n")
        f.write("TENNIS MATCH PREDICTION V2 - TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training data: 2005-2023 ({len(X_train)} matches)\n")
        f.write(f"Test data: 2024 ({len(X_test)} matches)\n")
        f.write(f"Features: {len(feature_cols)}\n\n")
        
        for r in results:
            f.write(f"--- {r['name']} ---\n")
            f.write(f"Accuracy: {r['accuracy']:.4f}\n")
            f.write(f"ROC-AUC:  {r['roc_auc']:.4f}\n")
            f.write(f"Log Loss: {r['log_loss']:.4f}\n\n")
        
        best_r = max(results, key=lambda x: x["accuracy"])
        f.write(f"\nBEST: {best_r['name']} (accuracy={best_r['accuracy']:.4f})\n")
        
        f.write(f"\nXGBoost best params:\n{json.dumps(study_xgb.best_params, indent=2)}\n")
        f.write(f"\nLightGBM best params:\n{json.dumps(study_lgb.best_params, indent=2)}\n")
        
        f.write(f"\nV1 baseline (for reference): 65.5% accuracy\n")
    
    # Summary
    print(f"\n{'='*60}")
    print("V2 RESULTS:")
    for r in results:
        marker = " ★" if r["accuracy"] == max(x["accuracy"] for x in results) else ""
        print(f"  {r['name']:<25} Acc: {r['accuracy']:.4f}  AUC: {r['roc_auc']:.4f}  LL: {r['log_loss']:.4f}{marker}")
    print(f"\nV1 baseline: 65.5% accuracy")
    print(f"{'='*60}")


if __name__ == "__main__":
    if STAGE in ("all", "data"):
        stage_data()
    if STAGE in ("all", "train"):
        stage_train()
