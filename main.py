"""Main orchestration script - runs the full tennis prediction pipeline."""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_and_clean
from src.elo import compute_elo_ratings
from src.features import engineer_features, get_feature_columns
from src.train import train_all_models
from src.visualize import plot_feature_importance, plot_elo_history
from src.predict import predict_match

import joblib
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    start = time.time()

    print("=" * 60)
    print("TENNIS MATCH PREDICTION PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/6] Loading and cleaning data...")
    matches, players = load_and_clean(DATA_DIR)

    # Step 2: Compute ELO ratings
    print("\n[2/6] Computing ELO ratings...")
    matches, overall_elo, surface_elo = compute_elo_ratings(matches)

    # Step 3: Engineer features
    print("\n[3/6] Engineering features...")
    feature_df = engineer_features(matches, seed=42)

    # Save processed features
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR), exist_ok=True)
    feature_path = os.path.join(DATA_DIR, "processed_features.csv")
    feature_df.to_csv(feature_path, index=False)
    print(f"  Saved processed features to {feature_path}")

    # Step 4: Train models
    print("\n[4/6] Training and evaluating models...")
    results, feature_cols, best = train_all_models(feature_df)

    # Save ELO dicts for prediction
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(dict(overall_elo), os.path.join(MODELS_DIR, "overall_elo.joblib"))
    joblib.dump(dict(surface_elo), os.path.join(MODELS_DIR, "surface_elo.joblib"))
    joblib.dump(matches, os.path.join(MODELS_DIR, "matches_cache.joblib"))
    joblib.dump(players, os.path.join(MODELS_DIR, "players_cache.joblib"))

    # Step 5: Generate visualizations
    print("\n[5/6] Generating visualizations...")
    plot_feature_importance(best["model"], feature_cols)
    plot_elo_history(matches)

    # Step 6: Demo predictions
    print("\n[6/6] Demo predictions...")
    demo_matchups = [
        ("Jannik Sinner", "Carlos Alcaraz", "Hard"),
        ("Novak Djokovic", "Carlos Alcaraz", "Clay"),
        ("Jannik Sinner", "Novak Djokovic", "Hard"),
    ]

    for p1_name, p2_name, surface in demo_matchups:
        pred = predict_match(
            p1_name, p2_name, surface,
            matches, players, overall_elo, surface_elo,
            model=best["model"], feature_cols=feature_cols,
        )
        if "error" in pred:
            print(f"  Error: {pred['error']}")
        else:
            print(f"\n  {pred['p1_name']} vs {pred['p2_name']} on {surface}:")
            print(f"    Predicted winner: {pred['predicted_winner']}")
            print(f"    Win probabilities: {pred['p1_name']} {pred['p1_win_probability']:.1%} - "
                  f"{pred['p2_name']} {pred['p2_win_probability']:.1%}")
            print(f"    ELO: {pred['p1_elo']:.0f} vs {pred['p2_elo']:.0f}")
            print(f"    H2H: {pred['h2h']}")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return results, best


if __name__ == "__main__":
    main()
