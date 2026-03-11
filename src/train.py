"""Model training, evaluation, and comparison."""

import os
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import get_feature_columns

SEED = 42
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def split_data(feature_df):
    """
    Time-based split:
    - Train: all matches through Dec 2023
    - Test: all matches in 2024
    """
    feature_cols = get_feature_columns()

    # Drop rows with too many NaNs in key features
    key_features = ["elo_diff", "surface_elo_diff", "rank_diff"]
    mask = feature_df[key_features].notna().all(axis=1)
    df = feature_df[mask].copy()

    # Fill remaining NaN with 0 (for rolling stats that haven't accumulated enough data)
    df[feature_cols] = df[feature_cols].fillna(0)

    train = df[df["date"] < "2024-01-01"]
    test = df[df["date"] >= "2024-01-01"]

    X_train = train[feature_cols].values
    y_train = train["target"].values.astype(int)
    X_test = test[feature_cols].values
    y_test = test["target"].values.astype(int)

    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Train target distribution: {np.mean(y_train):.3f}")
    print(f"Test target distribution: {np.mean(y_test):.3f}")

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
        "n_estimators": [100],
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


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with GridSearchCV."""
    print("\n--- Training XGBoost ---")
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_alpha": [0.01, 0.1],
        "reg_lambda": [1.0, 5.0],
    }
    xgb = XGBClassifier(
        random_state=SEED,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
    )
    grid = GridSearchCV(xgb, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    return evaluate_model("XGBoost", best, X_test, y_test)


def train_neural_network(X_train, y_train, X_test, y_test):
    """Train MLP Neural Network."""
    print("\n--- Training Neural Network ---")
    # Scale features for NN
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
        "name": name,
        "model": model,
        "accuracy": acc,
        "roc_auc": auc,
        "log_loss": ll,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def train_all_models(feature_df):
    """Train all models and compare."""
    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = split_data(feature_df)

    results = []
    results.append(train_decision_tree(X_train, y_train, X_test, y_test))
    results.append(train_random_forest(X_train, y_train, X_test, y_test))
    results.append(train_xgboost(X_train, y_train, X_test, y_test))
    results.append(train_neural_network(X_train, y_train, X_test, y_test))

    # Find best model
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
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best["model"], model_path)
    print(f"Best model saved to {model_path}")

    # Save all models
    for r in results:
        model_file = os.path.join(MODELS_DIR, f"{r['name'].lower().replace(' ', '_')}.joblib")
        joblib.dump(r["model"], model_file)
        if "scaler" in r:
            scaler_file = os.path.join(MODELS_DIR, "nn_scaler.joblib")
            joblib.dump(r["scaler"], scaler_file)

    # Save feature columns
    cols_path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(cols_path, "w") as f:
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
    comp_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"Model comparison saved to {comp_path}")

    # Save training report
    report_path = os.path.join(OUTPUTS_DIR, "training_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("TENNIS MATCH PREDICTION - TRAINING REPORT\n")
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
            f.write(f"Classification Report:\n")
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
