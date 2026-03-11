"""Generate visualizations: feature importance and ELO history."""

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


def plot_feature_importance(model, feature_cols, top_n=20):
    """Plot XGBoost feature importance."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("  Model has no feature_importances_, skipping plot.")
        return

    # Sort by importance
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_cols[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_importances, color="steelblue")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Top Feature Importances (Best Model)", fontsize=14)
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "feature_importance.png")
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
            # Smooth with rolling average for readability
            history = history.set_index("date").resample("M").last().dropna().reset_index()
            ax.plot(history["date"], history["elo"], label=name, color=colors[name], linewidth=1.5)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("ELO Rating", fontsize=12)
    ax.set_title("ELO Rating History - Big Three", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "elo_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ELO history chart saved to {path}")
