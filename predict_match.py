#!/usr/bin/env python3
"""CLI tool for predicting tennis match outcomes.

Usage:
    python predict_match.py "Jannik Sinner" "Carlos Alcaraz" "Hard"
    python predict_match.py "Novak Djokovic" "Rafael Nadal" "Clay" --best_of 5 --round F --level G
"""

import argparse
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(__file__))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def main():
    parser = argparse.ArgumentParser(description="Predict a tennis match outcome")
    parser.add_argument("player1", type=str, help="Name of player 1")
    parser.add_argument("player2", type=str, help="Name of player 2")
    parser.add_argument("surface", type=str, choices=["Hard", "Clay", "Grass"],
                        help="Court surface")
    parser.add_argument("--best_of", type=int, default=3, choices=[3, 5],
                        help="Best of 3 or 5 sets (default: 3)")
    parser.add_argument("--round", type=str, default="QF",
                        help="Round (R128, R64, R32, R16, QF, SF, F)")
    parser.add_argument("--level", type=str, default="M",
                        help="Tournament level: G=Grand Slam, M=Masters, A=ATP")
    args = parser.parse_args()

    # Load artifacts
    print("Loading model and data...")
    from src.predict import predict_match, load_prediction_artifacts

    model, feature_cols = load_prediction_artifacts()
    matches = joblib.load(os.path.join(MODELS_DIR, "matches_cache.joblib"))
    players = joblib.load(os.path.join(MODELS_DIR, "players_cache.joblib"))
    overall_elo = joblib.load(os.path.join(MODELS_DIR, "overall_elo.joblib"))
    surface_elo = joblib.load(os.path.join(MODELS_DIR, "surface_elo.joblib"))

    # Make prediction
    result = predict_match(
        args.player1, args.player2, args.surface,
        matches, players, overall_elo, surface_elo,
        model=model, feature_cols=feature_cols,
        best_of=args.best_of, round_name=args.round, tourney_level=args.level,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    # Display results
    print(f"\n{'='*50}")
    print(f"  MATCH PREDICTION")
    print(f"{'='*50}")
    print(f"  {result['p1_name']} vs {result['p2_name']}")
    print(f"  Surface: {result['surface']}")
    print(f"{'='*50}")
    print(f"\n  Predicted Winner: {result['predicted_winner']}")
    print(f"\n  Win Probabilities:")
    print(f"    {result['p1_name']}: {result['p1_win_probability']:.1%}")
    print(f"    {result['p2_name']}: {result['p2_win_probability']:.1%}")
    print(f"\n  ELO Ratings:")
    print(f"    {result['p1_name']}: {result['p1_elo']:.0f} (overall), {result['p1_surface_elo']:.0f} ({result['surface']})")
    print(f"    {result['p2_name']}: {result['p2_elo']:.0f} (overall), {result['p2_surface_elo']:.0f} ({result['surface']})")
    print(f"\n  Rankings:")
    print(f"    {result['p1_name']}: #{int(result['p1_rank'])}")
    print(f"    {result['p2_name']}: #{int(result['p2_rank'])}")
    print(f"\n  Head-to-Head: {result['h2h']}")

    if result["top_features"]:
        print(f"\n  Top Predictive Features:")
        for feat, imp in result["top_features"][:5]:
            print(f"    {feat}: {imp:.4f}")

    print(f"\n{'='*50}")


if __name__ == "__main__":
    main()
