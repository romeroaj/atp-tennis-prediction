"""Prediction pipeline for upcoming matches."""

import os
import json
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict

from .features import get_feature_columns, SERVE_WINDOW

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def load_prediction_artifacts():
    """Load saved model and metadata."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "r") as f:
        feature_cols = json.load(f)
    return model, feature_cols


def find_player(name, players_df):
    """Find a player by name (fuzzy match)."""
    name_lower = name.lower().strip()
    # Try exact match first
    mask = players_df["full_name"].str.lower() == name_lower
    if mask.any():
        return players_df[mask].iloc[0]

    # Try partial match
    mask = players_df["full_name"].str.lower().str.contains(name_lower, na=False)
    if mask.any():
        return players_df[mask].iloc[0]

    # Try last name only
    mask = players_df["last_name"].str.lower().str.contains(name_lower, na=False)
    if mask.any():
        return players_df[mask].iloc[0]

    return None


def get_player_stats(player_id, matches_df, overall_elo, surface_elo, surface):
    """Get current stats for a player from historical data."""
    # Current ELO
    elo = overall_elo.get(player_id, 1500)
    surf_elo = surface_elo.get((player_id, surface), 1500)

    # Recent win rate
    as_winner = matches_df[matches_df["winner_id"] == player_id]
    as_loser = matches_df[matches_df["loser_id"] == player_id]

    all_matches = pd.concat([
        as_winner[["date"]].assign(won=1),
        as_loser[["date"]].assign(won=0),
    ]).sort_values("date")

    total = len(all_matches)
    recent_10 = all_matches.tail(10)["won"].mean() if total >= 10 else (all_matches["won"].mean() if total > 0 else 0.5)
    recent_25 = all_matches.tail(25)["won"].mean() if total >= 25 else (all_matches["won"].mean() if total > 0 else 0.5)
    recent_50 = all_matches.tail(50)["won"].mean() if total >= 50 else (all_matches["won"].mean() if total > 0 else 0.5)
    recent_100 = all_matches.tail(100)["won"].mean() if total >= 100 else (all_matches["won"].mean() if total > 0 else 0.5)

    # Serve stats (from last SERVE_WINDOW matches as winner + loser)
    recent_w = as_winner.tail(SERVE_WINDOW)
    recent_l = as_loser.tail(SERVE_WINDOW)

    serve_stats_1st, serve_stats_bp, serve_stats_ace = [], [], []

    for _, row in recent_w.iterrows():
        if pd.notna(row.get("w_1stIn")) and row["w_1stIn"] > 0:
            serve_stats_1st.append(row["w_1stWon"] / row["w_1stIn"])
        if pd.notna(row.get("w_bpFaced")) and row["w_bpFaced"] > 0:
            serve_stats_bp.append(row["w_bpSaved"] / row["w_bpFaced"])
        if pd.notna(row.get("w_svpt")) and row["w_svpt"] > 0:
            serve_stats_ace.append(row["w_ace"] / row["w_svpt"])

    for _, row in recent_l.iterrows():
        if pd.notna(row.get("l_1stIn")) and row["l_1stIn"] > 0:
            serve_stats_1st.append(row["l_1stWon"] / row["l_1stIn"])
        if pd.notna(row.get("l_bpFaced")) and row["l_bpFaced"] > 0:
            serve_stats_bp.append(row["l_bpSaved"] / row["l_bpFaced"])
        if pd.notna(row.get("l_svpt")) and row["l_svpt"] > 0:
            serve_stats_ace.append(row["l_ace"] / row["l_svpt"])

    avg_1st = np.mean(serve_stats_1st) if serve_stats_1st else 0.0
    avg_bp = np.mean(serve_stats_bp) if serve_stats_bp else 0.0
    avg_ace = np.mean(serve_stats_ace) if serve_stats_ace else 0.0

    # Rank info (from last match)
    last_as_w = as_winner.tail(1)
    last_as_l = as_loser.tail(1)
    if not last_as_w.empty and (last_as_l.empty or last_as_w["date"].values[0] >= last_as_l["date"].values[0]):
        rank = last_as_w["winner_rank"].values[0]
        rank_pts = last_as_w["winner_rank_points"].values[0]
        age = last_as_w["winner_age"].values[0]
        height = last_as_w["winner_ht"].values[0]
    elif not last_as_l.empty:
        rank = last_as_l["loser_rank"].values[0]
        rank_pts = last_as_l["loser_rank_points"].values[0]
        age = last_as_l["loser_age"].values[0]
        height = last_as_l["loser_ht"].values[0]
    else:
        rank, rank_pts, age, height = 500, 0, 25, 185

    return {
        "elo": elo,
        "surface_elo": surf_elo,
        "rank": rank if pd.notna(rank) else 500,
        "rank_points": rank_pts if pd.notna(rank_pts) else 0,
        "age": age if pd.notna(age) else 25,
        "height": height if pd.notna(height) else 185,
        "winrate_10": recent_10,
        "winrate_25": recent_25,
        "winrate_50": recent_50,
        "winrate_100": recent_100,
        "avg_1stServeWinPct": avg_1st,
        "avg_bpSavedPct": avg_bp,
        "avg_acePct": avg_ace,
    }


def get_h2h(p1_id, p2_id, matches_df):
    """Get head-to-head record."""
    p1_wins = len(matches_df[
        (matches_df["winner_id"] == p1_id) & (matches_df["loser_id"] == p2_id)
    ])
    p2_wins = len(matches_df[
        (matches_df["winner_id"] == p2_id) & (matches_df["loser_id"] == p1_id)
    ])
    return p1_wins, p2_wins


def predict_match(p1_name, p2_name, surface, matches_df, players_df, overall_elo, surface_elo,
                   model=None, feature_cols=None, best_of=3, round_name="QF", tourney_level="M"):
    """
    Predict match outcome between two players.

    Returns prediction dict with probabilities and driving features.
    """
    if model is None or feature_cols is None:
        model, feature_cols = load_prediction_artifacts()

    # Find players
    p1 = find_player(p1_name, players_df)
    p2 = find_player(p2_name, players_df)

    if p1 is None:
        return {"error": f"Player not found: {p1_name}"}
    if p2 is None:
        return {"error": f"Player not found: {p2_name}"}

    p1_id = int(p1["player_id"])
    p2_id = int(p2["player_id"])
    p1_full = p1["full_name"]
    p2_full = p2["full_name"]

    # Get current stats
    s1 = get_player_stats(p1_id, matches_df, overall_elo, surface_elo, surface)
    s2 = get_player_stats(p2_id, matches_df, overall_elo, surface_elo, surface)
    h2h_p1, h2h_p2 = get_h2h(p1_id, p2_id, matches_df)

    from .features import ROUND_MAP
    round_num = ROUND_MAP.get(round_name, 3)

    # Build feature vector (p1 perspective)
    feature_dict = {
        "p1_elo": s1["elo"],
        "p2_elo": s2["elo"],
        "p1_surface_elo": s1["surface_elo"],
        "p2_surface_elo": s2["surface_elo"],
        "elo_diff": s1["elo"] - s2["elo"],
        "surface_elo_diff": s1["surface_elo"] - s2["surface_elo"],
        "total_elo": s1["elo"] + s2["elo"],
        "h2h_p1_wins": h2h_p1,
        "h2h_p2_wins": h2h_p2,
        "h2h_diff": h2h_p1 - h2h_p2,
        "age_diff": s1["age"] - s2["age"],
        "height_diff": s1["height"] - s2["height"],
        "rank_diff": s1["rank"] - s2["rank"],
        "rank_points_diff": s1["rank_points"] - s2["rank_points"],
        "best_of": best_of,
        "round_num": round_num,
        "surface_Hard": 1 if surface == "Hard" else 0,
        "surface_Clay": 1 if surface == "Clay" else 0,
        "surface_Grass": 1 if surface == "Grass" else 0,
        "tourney_G": 1 if tourney_level == "G" else 0,
        "tourney_M": 1 if tourney_level == "M" else 0,
        "tourney_A": 1 if tourney_level in ("A", "B") else 0,
        "p1_avg_1stServeWinPct": s1["avg_1stServeWinPct"],
        "p2_avg_1stServeWinPct": s2["avg_1stServeWinPct"],
        "p1_avg_bpSavedPct": s1["avg_bpSavedPct"],
        "p2_avg_bpSavedPct": s2["avg_bpSavedPct"],
        "p1_avg_acePct": s1["avg_acePct"],
        "p2_avg_acePct": s2["avg_acePct"],
        "serve_1stWinPct_diff": s1["avg_1stServeWinPct"] - s2["avg_1stServeWinPct"],
        "serve_bpSavedPct_diff": s1["avg_bpSavedPct"] - s2["avg_bpSavedPct"],
        "serve_acePct_diff": s1["avg_acePct"] - s2["avg_acePct"],
        "p1_winrate_last_10": s1["winrate_10"],
        "p2_winrate_last_10": s2["winrate_10"],
        "winrate_diff_last_10": s1["winrate_10"] - s2["winrate_10"],
        "p1_winrate_last_25": s1["winrate_25"],
        "p2_winrate_last_25": s2["winrate_25"],
        "winrate_diff_last_25": s1["winrate_25"] - s2["winrate_25"],
        "p1_winrate_last_50": s1["winrate_50"],
        "p2_winrate_last_50": s2["winrate_50"],
        "winrate_diff_last_50": s1["winrate_50"] - s2["winrate_50"],
        "p1_winrate_last_100": s1["winrate_100"],
        "p2_winrate_last_100": s2["winrate_100"],
        "winrate_diff_last_100": s1["winrate_100"] - s2["winrate_100"],
    }

    X = np.array([[feature_dict.get(c, 0) for c in feature_cols]])

    # Predict
    proba = model.predict_proba(X)[0]
    p1_win_prob = proba[1]
    p2_win_prob = proba[0]

    # Feature importance for this prediction
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_features = sorted(
            zip(feature_cols, importances),
            key=lambda x: abs(x[1]), reverse=True
        )[:10]
    else:
        top_features = []

    return {
        "p1_name": p1_full,
        "p2_name": p2_full,
        "surface": surface,
        "p1_win_probability": float(p1_win_prob),
        "p2_win_probability": float(p2_win_prob),
        "predicted_winner": p1_full if p1_win_prob > 0.5 else p2_full,
        "p1_elo": s1["elo"],
        "p2_elo": s2["elo"],
        "p1_surface_elo": s1["surface_elo"],
        "p2_surface_elo": s2["surface_elo"],
        "h2h": f"{p1_full} {h2h_p1} - {h2h_p2} {p2_full}",
        "p1_rank": s1["rank"],
        "p2_rank": s2["rank"],
        "top_features": top_features,
    }
