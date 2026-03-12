"""V2 Visualizations: feature importance, ELO history, Optuna plots, V1 vs V2 comparison."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .elo import get_player_elo_history

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

# Big Three player IDs
BIG_THREE = {
    "Roger Federer": 103819,
    "Rafael Nadal": 104745,
    "Novak Djokovic": 104925,
}


def plot_feature_importance(model, feature_cols, top_n=20, suffix="v2"):
    """Plot feature importance for the best model."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("  Model has no feature_importances_, skipping plot.")
        return

    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_cols[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("viridis", n_colors=top_n)
    ax.barh(range(len(top_features)), top_importances, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances (V2 Best Model)", fontsize=14)
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, f"feature_importance_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance chart saved to {path}")


def plot_elo_history(matches):
    """Plot ELO history for the Big Three."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"Roger Federer": "#e74c3c", "Rafael Nadal": "#f39c12", "Novak Djokovic": "#2ecc71"}

    for name, pid in BIG_THREE.items():
        history = get_player_elo_history(matches, pid)
        if len(history) > 0:
            history = history.set_index("date").resample("M").last().dropna().reset_index()
            ax.plot(history["date"], history["elo"], label=name, color=colors[name], linewidth=1.5)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("ELO Rating", fontsize=12)
    ax.set_title("V2 ELO Rating History - Big Three", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "elo_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ELO history chart saved to {path}")


def plot_optuna_history(study, model_name, filename):
    """Plot Optuna optimization history (accuracy vs trial number)."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    trials = study.trials
    trial_nums = [t.number for t in trials]
    trial_values = [t.value for t in trials if t.value is not None]
    trial_nums_valid = [t.number for t in trials if t.value is not None]

    # Running best
    running_best = []
    best_so_far = 0
    for v in trial_values:
        best_so_far = max(best_so_far, v)
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(trial_nums_valid, trial_values, alpha=0.3, s=15, color="steelblue", label="Trial accuracy")
    ax.plot(trial_nums_valid, running_best, color="red", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("CV Accuracy", fontsize=12)
    ax.set_title(f"Optuna Optimization History — {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Optuna history plot saved to {path}")


def plot_optuna_param_importance(study, model_name, filename):
    """Plot parameter importance from Optuna study."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    try:
        from optuna.importance import get_param_importances
        importances = get_param_importances(study)

        params = list(importances.keys())
        values = list(importances.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(params)), values, color=sns.color_palette("coolwarm", len(params)))
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params, fontsize=10)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"Optuna Parameter Importance — {model_name}", fontsize=14)
        plt.tight_layout()

        path = os.path.join(OUTPUTS_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Optuna param importance plot saved to {path}")
    except Exception as e:
        print(f"  Could not plot param importance for {model_name}: {e}")


def plot_v1_v2_comparison(v1_results_path=None):
    """
    Plot V1 vs V2 model comparison bar chart.
    Reads V1 results from model_comparison.csv and V2 from model_comparison_v2.csv.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    v1_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    v2_path = os.path.join(OUTPUTS_DIR, "model_comparison_v2.csv")

    if not os.path.exists(v2_path):
        print("  V2 comparison not found, skipping V1 vs V2 plot.")
        return

    v2_df = pd.read_csv(v2_path)

    fig, ax = plt.subplots(figsize=(12, 7))

    # If V1 results exist, show side by side
    if os.path.exists(v1_path):
        v1_df = pd.read_csv(v1_path)

        # Merge V1 models with V2 on matching names
        v1_models = set(v1_df["model"].values)
        v2_models = set(v2_df["model"].values)
        all_models = sorted(v2_models)

        x = np.arange(len(all_models))
        width = 0.35

        v1_accs = []
        v2_accs = []
        for m in all_models:
            v1_row = v1_df[v1_df["model"] == m]
            v2_row = v2_df[v2_df["model"] == m]
            v1_accs.append(v1_row["accuracy"].values[0] if len(v1_row) > 0 else 0)
            v2_accs.append(v2_row["accuracy"].values[0] if len(v2_row) > 0 else 0)

        bars1 = ax.bar(x - width/2, v1_accs, width, label="V1", color="#e74c3c", alpha=0.8)
        bars2 = ax.bar(x + width/2, v2_accs, width, label="V2", color="#2ecc71", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(all_models, rotation=30, ha="right", fontsize=10)
        ax.legend(fontsize=12)

        # Add value labels
        for bar in bars1:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)
    else:
        # Just plot V2 results
        models = v2_df["model"].values
        accs = v2_df["accuracy"].values
        colors = sns.color_palette("viridis", len(models))
        bars = ax.bar(range(len(models)), accs, color=colors)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("V1 vs V2 Model Comparison", fontsize=14)
    ax.set_ylim(0.5, max(v2_df["accuracy"].max() + 0.05, 0.75))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "v1_v2_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  V1 vs V2 comparison chart saved to {path}")
