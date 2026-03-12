"""V2 Model training: Optuna tuning for XGBoost + LightGBM, stacking ensemble, rolling CV."""

import os
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import lightgbm as lgb
import optuna

from .features import get_feature_columns

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def split_data(feature_df):
    """
    V2 time-based split:
    - Train: matches from 2005 through Dec 2023 (modern era only)
    - Test: all matches in 2024
    ELO is still computed from 1985 for full history.
    """
    feature_cols = get_feature_columns()

    # Drop rows with NaN in critical features
    key_features = ["elo_diff", "surface_elo_diff", "rank_diff"]
    mask = feature_df[key_features].notna().all(axis=1)
    df = feature_df[mask].copy()

    # Fill remaining NaN with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    train = df[(df["date"] >= "2005-01-01") & (df["date"] < "2024-01-01")]
    test = df[df["date"] >= "2024-01-01"]

    X_train = train[feature_cols].values
    y_train = train["target"].values.astype(int)
    X_test = test[feature_cols].values
    y_test = test["target"].values.astype(int)

    print(f"Train: {len(X_train)} samples (2005-2023), Test: {len(X_test)} samples (2024)")
    print(f"Train target distribution: {np.mean(y_train):.3f}")
    print(f"Test target distribution: {np.mean(y_test):.3f}")
    print(f"Feature columns: {len(feature_cols)}")

    return X_train, y_train, X_test, y_test, feature_cols, train, test


def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train Decision Tree with GridSearchCV."""
    print("\n--- Training Decision Tree ---")
    param_grid = {
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [10, 20, 50],
        "min_samples_leaf": [5, 10, 20],
    }
    dt = DecisionTreeClassifier(random_state=SEED)
    grid = GridSearchCV(dt, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    return evaluate_model("Decision Tree", best, X_test, y_test)


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with GridSearchCV."""
    print("\n--- Training Random Forest ---")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, 20],
        "min_samples_split": [10, 20],
        "min_samples_leaf": [5, 10],
    }
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    return evaluate_model("Random Forest", best, X_test, y_test)


def train_xgboost_grid(X_train, y_train, X_test, y_test):
    """Train XGBoost with GridSearchCV (V1 baseline)."""
    print("\n--- Training XGBoost (GridSearch - V1 baseline) ---")
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_alpha": [0.01, 0.1],
        "reg_lambda": [1.0, 5.0],
    }
    xgb = XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1)
    grid = GridSearchCV(xgb, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    return evaluate_model("XGBoost (GridSearch)", best, X_test, y_test)


def train_xgboost_optuna(X_train, y_train, X_test, y_test, n_trials=200):
    """Train XGBoost with Optuna Bayesian optimization — THE STAR."""
    print(f"\n--- Training XGBoost (Optuna, {n_trials} trials) ---")

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 1.2),
        }

        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]

            model = XGBClassifier(
                **params, random_state=SEED, eval_metric="logloss", n_jobs=-1,
                early_stopping_rounds=50,
            )
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            scores.append(accuracy_score(y_v, model.predict(X_v)))

        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    # Print progress every 50 trials
    def progress_callback(study, trial):
        if (trial.number + 1) % 50 == 0:
            print(f"    Trial {trial.number + 1}/{n_trials}, best so far: {study.best_value:.4f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])

    best_params = study.best_params
    print(f"  Best Optuna params: {best_params}")
    print(f"  Best CV accuracy: {study.best_value:.4f}")

    # Train final model with best params on full training set
    final_model = XGBClassifier(
        **best_params, random_state=SEED, eval_metric="logloss", n_jobs=-1,
    )
    final_model.fit(X_train, y_train)

    result = evaluate_model("XGBoost (Optuna)", final_model, X_test, y_test)
    result["optuna_study"] = study
    return result


def train_lightgbm_optuna(X_train, y_train, X_test, y_test, n_trials=200):
    """Train LightGBM with Optuna Bayesian optimization."""
    print(f"\n--- Training LightGBM (Optuna, {n_trials} trials) ---")

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]

            model = lgb.LGBMClassifier(
                **params, random_state=SEED, verbose=-1, n_jobs=-1,
            )
            model.fit(
                X_t, y_t,
                eval_set=[(X_v, y_v)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            scores.append(accuracy_score(y_v, model.predict(X_v)))

        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    def progress_callback(study, trial):
        if (trial.number + 1) % 50 == 0:
            print(f"    Trial {trial.number + 1}/{n_trials}, best so far: {study.best_value:.4f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])

    best_params = study.best_params
    print(f"  Best Optuna params: {best_params}")
    print(f"  Best CV accuracy: {study.best_value:.4f}")

    final_model = lgb.LGBMClassifier(
        **best_params, random_state=SEED, verbose=-1, n_jobs=-1,
    )
    final_model.fit(X_train, y_train)

    result = evaluate_model("LightGBM (Optuna)", final_model, X_test, y_test)
    result["optuna_study"] = study
    return result


def train_neural_network(X_train, y_train, X_test, y_test):
    """Train MLP Neural Network."""
    print("\n--- Training Neural Network ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=500,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate="adaptive",
        verbose=False,
    )
    mlp.fit(X_train_scaled, y_train)

    result = evaluate_model("Neural Network", mlp, X_test_scaled, y_test)
    result["scaler"] = scaler
    return result


def train_stacking_ensemble(X_train, y_train, X_test, y_test, xgb_model, lgb_model):
    """Stack XGBoost + LightGBM with Logistic Regression meta-learner."""
    print("\n--- Training Stacking Ensemble ---")
    estimators = [
        ("xgb", xgb_model),
        ("lgb", lgb_model),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=TimeSeriesSplit(n_splits=5),
        passthrough=True,
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    return evaluate_model("Stacking Ensemble", stack, X_test, y_test)


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    ll = log_loss(y_test, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"  {name} Results:")
    print(f"    Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"    ROC-AUC:  {auc:.4f}")
    if ll is not None:
        print(f"    Log Loss: {ll:.4f}")
    print(f"    Confusion Matrix:\n{cm}")

    return {
        "name": name, "model": model,
        "accuracy": acc, "roc_auc": auc, "log_loss": ll,
        "confusion_matrix": cm, "classification_report": report,
    }


def train_all_models(feature_df):
    """Train all V2 models and compare."""
    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = split_data(feature_df)

    results = []

    # V2 stars — Optuna-tuned
    xgb_optuna = train_xgboost_optuna(X_train, y_train, X_test, y_test, n_trials=200)
    results.append(xgb_optuna)

    lgb_optuna = train_lightgbm_optuna(X_train, y_train, X_test, y_test, n_trials=200)
    results.append(lgb_optuna)

    # Stacking ensemble using the Optuna-tuned models
    stack_result = train_stacking_ensemble(
        X_train, y_train, X_test, y_test,
        xgb_optuna["model"], lgb_optuna["model"],
    )
    results.append(stack_result)

    # Find best
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best['name']} with accuracy {best['accuracy']:.4f}")
    print(f"{'='*60}")

    # Save results
    save_results(results, feature_cols, best)

    return results, feature_cols, best


def save_results(results, feature_cols, best):
    """Save models, comparison CSV, and training report."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Save best model
    joblib.dump(best["model"], os.path.join(MODELS_DIR, "best_model.joblib"))
    print(f"Best model saved")

    # Save individual models
    for r in results:
        name = r["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(r["model"], os.path.join(MODELS_DIR, f"{name}.joblib"))
        if "scaler" in r:
            joblib.dump(r["scaler"], os.path.join(MODELS_DIR, "nn_scaler.joblib"))
        if "optuna_study" in r:
            joblib.dump(r["optuna_study"], os.path.join(MODELS_DIR, f"{name}_study.joblib"))

    # Save feature columns
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)

    # Save model comparison CSV
    comparison = []
    for r in results:
        comparison.append({
            "model": r["name"],
            "accuracy": round(r["accuracy"], 4),
            "roc_auc": round(r["roc_auc"], 4) if r["roc_auc"] else None,
            "log_loss": round(r["log_loss"], 4) if r["log_loss"] else None,
        })
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison_v2.csv"), index=False)

    # Save training report
    report_path = os.path.join(OUTPUTS_DIR, "training_report_v2.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("TENNIS MATCH PREDICTION V2 - TRAINING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for r in results:
            f.write(f"--- {r['name']} ---\n")
            f.write(f"Accuracy: {r['accuracy']:.4f}\n")
            if r["roc_auc"]:
                f.write(f"ROC-AUC:  {r['roc_auc']:.4f}\n")
            if r["log_loss"]:
                f.write(f"Log Loss: {r['log_loss']:.4f}\n")
            f.write(f"Confusion Matrix:\n{r['confusion_matrix']}\n")
            report = r["classification_report"]
            for key in ["0", "1"]:
                if key in report:
                    f.write(f"  Class {key}: precision={report[key]['precision']:.3f}, "
                            f"recall={report[key]['recall']:.3f}, "
                            f"f1={report[key]['f1-score']:.3f}\n")
            f.write("\n")

        f.write(f"\nBEST MODEL: {best['name']} (accuracy={best['accuracy']:.4f})\n")
        f.write(f"\nFeature columns ({len(feature_cols)}):\n")
        for col in feature_cols:
            f.write(f"  - {col}\n")

    print(f"Training report saved to {report_path}")
