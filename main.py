#!/usr/bin/env python3
"""V2 Tennis Match Prediction Pipeline — full end-to-end."""

import os
import sys
import time
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_and_clean
from src.elo import compute_elo_ratings
from src.features import engineer_features
from src.train import train_all_models
from src.visualize import (
    plot_feature_importance, plot_elo_history,
    plot_optuna_history, plot_optuna_param_importance,
    plot_v1_v2_comparison,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 60)
    print("TENNIS MATCH PREDICTION V2 PIPELINE")
    print("=" * 60)
    print("Steps: Load → ELO → Features → Train (200 Optuna trials each) → Save → Visualize")

    # Step 1: Load data (all years for ELO history)
    t0 = time.time()
    print("\n[Step 1/6] Loading and cleaning data...")
    matches, players = load_and_clean()
    print(f"  Time: {time.time() - t0:.1f}s")

    # Step 2: Compute V2 ELO ratings
    t1 = time.time()
    print("\n[Step 2/6] Computing V2 ELO ratings...")
    matches, overall_elo, surface_elo = compute_elo_ratings(matches)
    print(f"  Time: {time.time() - t1:.1f}s")

    # Step 3: Engineer V2 features
    t2 = time.time()
    print("\n[Step 3/6] Engineering V2 features...")
    feature_df = engineer_features(matches)
    print(f"  Time: {time.time() - t2:.1f}s")

    # Save processed features
    feature_path = os.path.join(DATA_DIR, "processed_features_v2.csv")
    feature_df.to_csv(feature_path, index=False)
    print(f"  Features saved to {feature_path}")

    # Step 4: Train all models
    t3 = time.time()
    print("\n[Step 4/6] Training models...")
    results, feature_cols, best = train_all_models(feature_df)
    print(f"  Training time: {time.time() - t3:.1f}s")

    # Step 5: Save ELO state and caches for prediction
    t4 = time.time()
    print("\n[Step 5/6] Saving artifacts...")
    joblib.dump(overall_elo, os.path.join(MODELS_DIR, "overall_elo.joblib"))
    joblib.dump(surface_elo, os.path.join(MODELS_DIR, "surface_elo.joblib"))
    joblib.dump(matches, os.path.join(MODELS_DIR, "matches_cache.joblib"))
    joblib.dump(players, os.path.join(MODELS_DIR, "players_cache.joblib"))
    print(f"  Time: {time.time() - t4:.1f}s")

    # Step 6: Generate visualizations
    t5 = time.time()
    print("\n[Step 6/6] Generating visualizations...")
    plot_feature_importance(best["model"], feature_cols)
    plot_elo_history(matches)
    for r in results:
        if "optuna_study" in r:
            sname = r["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
            plot_optuna_history(r["optuna_study"], r["name"], f"optuna_history_{sname}.png")
            plot_optuna_param_importance(r["optuna_study"], r["name"], f"optuna_params_{sname}.png")
    plot_v1_v2_comparison()
    print(f"  Time: {time.time() - t5:.1f}s")

    # Summary
    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — Total time: {total_time:.1f}s")
    print(f"Best model: {best['name']} (accuracy={best['accuracy']:.4f}, ROC-AUC={best['roc_auc']:.4f})")
    print(f"{'=' * 60}")

    # Print full comparison table
    print("\nMODEL COMPARISON:")
    print(f"{'Model':<30} {'Accuracy':>10} {'ROC-AUC':>10} {'Log Loss':>10}")
    print("-" * 62)
    for r in results:
        acc = f"{r['accuracy']:.4f}"
        auc = f"{r['roc_auc']:.4f}" if r['roc_auc'] else "N/A"
        ll = f"{r['log_loss']:.4f}" if r['log_loss'] else "N/A"
        marker = " ★" if r['name'] == best['name'] else ""
        print(f"{r['name']:<30} {acc:>10} {auc:>10} {ll:>10}{marker}")


if __name__ == "__main__":
    main()
