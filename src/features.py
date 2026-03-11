"""Feature engineering with strict no-data-leakage discipline."""

import numpy as np
import pandas as pd
from collections import defaultdict


ROUND_MAP = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7,
    "RR": 3, "BR": 4, "ER": 1,
}

ROLLING_WINDOWS = [10, 25, 50, 100]
SERVE_WINDOW = 20


def engineer_features(matches, seed=42):
    """
    Engineer features from match data. ALL features use only pre-match data.

    CRITICAL: Randomly assign winner/loser as p1/p2 to avoid leakage.
    Target: did p1 win? (1 or 0)
    """
    np.random.seed(seed)
    matches = matches.sort_values(["date", "tourney_id", "match_num"]).reset_index(drop=True)
    n = len(matches)

    print(f"Engineering features for {n} matches...")

    # Random coin flip: if True, p1=winner; if False, p1=loser
    coin = np.random.randint(0, 2, size=n).astype(bool)

    # --- Build player history structures for rolling stats ---
    # We'll iterate chronologically and maintain running histories
    player_results = defaultdict(list)  # player_id -> list of (date_idx, won_bool)
    player_serve_stats = defaultdict(list)  # player_id -> list of dicts with serve stats
    h2h_record = defaultdict(lambda: [0, 0])  # (p1_id, p2_id) sorted -> [smaller_id_wins, larger_id_wins]

    # Pre-extract columns as arrays for speed
    winner_ids = matches["winner_id"].values
    loser_ids = matches["loser_id"].values
    dates = matches["date"].values
    surfaces = matches["surface"].values
    winner_elo = matches["winner_elo"].values
    loser_elo = matches["loser_elo"].values
    winner_surface_elo = matches["winner_surface_elo"].values
    loser_surface_elo = matches["loser_surface_elo"].values
    winner_age = matches["winner_age"].values
    loser_age = matches["loser_age"].values
    winner_ht = matches["winner_ht"].values
    loser_ht = matches["loser_ht"].values
    winner_rank = matches["winner_rank"].values
    loser_rank = matches["loser_rank"].values
    winner_rank_pts = matches["winner_rank_points"].values
    loser_rank_pts = matches["loser_rank_points"].values
    best_of = matches["best_of"].values
    rounds = matches["round"].values
    tourney_levels = matches["tourney_level"].values

    # Serve stat columns
    w_ace = matches["w_ace"].values if "w_ace" in matches.columns else np.full(n, np.nan)
    w_svpt = matches["w_svpt"].values if "w_svpt" in matches.columns else np.full(n, np.nan)
    w_1stIn = matches["w_1stIn"].values if "w_1stIn" in matches.columns else np.full(n, np.nan)
    w_1stWon = matches["w_1stWon"].values if "w_1stWon" in matches.columns else np.full(n, np.nan)
    w_bpSaved = matches["w_bpSaved"].values if "w_bpSaved" in matches.columns else np.full(n, np.nan)
    w_bpFaced = matches["w_bpFaced"].values if "w_bpFaced" in matches.columns else np.full(n, np.nan)
    l_ace = matches["l_ace"].values if "l_ace" in matches.columns else np.full(n, np.nan)
    l_svpt = matches["l_svpt"].values if "l_svpt" in matches.columns else np.full(n, np.nan)
    l_1stIn = matches["l_1stIn"].values if "l_1stIn" in matches.columns else np.full(n, np.nan)
    l_1stWon = matches["l_1stWon"].values if "l_1stWon" in matches.columns else np.full(n, np.nan)
    l_bpSaved = matches["l_bpSaved"].values if "l_bpSaved" in matches.columns else np.full(n, np.nan)
    l_bpFaced = matches["l_bpFaced"].values if "l_bpFaced" in matches.columns else np.full(n, np.nan)

    # Output feature arrays
    features = {
        "p1_elo": np.zeros(n),
        "p2_elo": np.zeros(n),
        "p1_surface_elo": np.zeros(n),
        "p2_surface_elo": np.zeros(n),
        "elo_diff": np.zeros(n),
        "surface_elo_diff": np.zeros(n),
        "total_elo": np.zeros(n),
        "h2h_p1_wins": np.zeros(n),
        "h2h_p2_wins": np.zeros(n),
        "h2h_diff": np.zeros(n),
        "age_diff": np.full(n, np.nan),
        "height_diff": np.full(n, np.nan),
        "rank_diff": np.full(n, np.nan),
        "rank_points_diff": np.full(n, np.nan),
        "best_of": np.zeros(n),
        "round_num": np.zeros(n),
        "surface_Hard": np.zeros(n),
        "surface_Clay": np.zeros(n),
        "surface_Grass": np.zeros(n),
        "tourney_G": np.zeros(n),
        "tourney_M": np.zeros(n),
        "tourney_A": np.zeros(n),
        "p1_avg_1stServeWinPct": np.full(n, np.nan),
        "p2_avg_1stServeWinPct": np.full(n, np.nan),
        "p1_avg_bpSavedPct": np.full(n, np.nan),
        "p2_avg_bpSavedPct": np.full(n, np.nan),
        "p1_avg_acePct": np.full(n, np.nan),
        "p2_avg_acePct": np.full(n, np.nan),
        "serve_1stWinPct_diff": np.full(n, np.nan),
        "serve_bpSavedPct_diff": np.full(n, np.nan),
        "serve_acePct_diff": np.full(n, np.nan),
        "target": np.zeros(n),
        "p1_id": np.zeros(n, dtype=int),
        "p2_id": np.zeros(n, dtype=int),
    }

    # Add rolling win rate features
    for w in ROLLING_WINDOWS:
        features[f"p1_winrate_last_{w}"] = np.full(n, np.nan)
        features[f"p2_winrate_last_{w}"] = np.full(n, np.nan)
        features[f"winrate_diff_last_{w}"] = np.full(n, np.nan)

    print("  Computing features chronologically (this may take a few minutes)...")
    for i in range(n):
        if i % 20000 == 0 and i > 0:
            print(f"    Processed {i}/{n} matches...")

        w_id = int(winner_ids[i])
        l_id = int(loser_ids[i])

        # Randomly assign p1/p2
        if coin[i]:
            p1_id, p2_id = w_id, l_id
            p1_won = 1
        else:
            p1_id, p2_id = l_id, w_id
            p1_won = 0

        features["p1_id"][i] = p1_id
        features["p2_id"][i] = p2_id
        features["target"][i] = p1_won

        # ELO features (pre-match)
        if coin[i]:
            p1_elo_val = winner_elo[i]
            p2_elo_val = loser_elo[i]
            p1_surf_elo = winner_surface_elo[i]
            p2_surf_elo = loser_surface_elo[i]
        else:
            p1_elo_val = loser_elo[i]
            p2_elo_val = winner_elo[i]
            p1_surf_elo = loser_surface_elo[i]
            p2_surf_elo = winner_surface_elo[i]

        features["p1_elo"][i] = p1_elo_val
        features["p2_elo"][i] = p2_elo_val
        features["p1_surface_elo"][i] = p1_surf_elo
        features["p2_surface_elo"][i] = p2_surf_elo
        features["elo_diff"][i] = p1_elo_val - p2_elo_val
        features["surface_elo_diff"][i] = p1_surf_elo - p2_surf_elo
        features["total_elo"][i] = p1_elo_val + p2_elo_val

        # H2H (pre-match)
        key = (min(p1_id, p2_id), max(p1_id, p2_id))
        if p1_id < p2_id:
            p1_h2h = h2h_record[key][0]
            p2_h2h = h2h_record[key][1]
        else:
            p1_h2h = h2h_record[key][1]
            p2_h2h = h2h_record[key][0]

        features["h2h_p1_wins"][i] = p1_h2h
        features["h2h_p2_wins"][i] = p2_h2h
        features["h2h_diff"][i] = p1_h2h - p2_h2h

        # Rolling win rates (pre-match)
        for w in ROLLING_WINDOWS:
            p1_results = player_results[p1_id]
            p2_results = player_results[p2_id]

            if len(p1_results) >= 5:
                recent_p1 = p1_results[-w:]
                features[f"p1_winrate_last_{w}"][i] = np.mean([r[1] for r in recent_p1])
            if len(p2_results) >= 5:
                recent_p2 = p2_results[-w:]
                features[f"p2_winrate_last_{w}"][i] = np.mean([r[1] for r in recent_p2])
            if len(p1_results) >= 5 and len(p2_results) >= 5:
                features[f"winrate_diff_last_{w}"][i] = (
                    features[f"p1_winrate_last_{w}"][i] - features[f"p2_winrate_last_{w}"][i]
                )

        # Serve stats (pre-match, rolling average of last SERVE_WINDOW matches)
        p1_serve = player_serve_stats[p1_id]
        p2_serve = player_serve_stats[p2_id]

        if len(p1_serve) >= 3:
            recent = p1_serve[-SERVE_WINDOW:]
            features["p1_avg_1stServeWinPct"][i] = np.nanmean([s["first_serve_win"] for s in recent])
            features["p1_avg_bpSavedPct"][i] = np.nanmean([s["bp_saved"] for s in recent])
            features["p1_avg_acePct"][i] = np.nanmean([s["ace_pct"] for s in recent])

        if len(p2_serve) >= 3:
            recent = p2_serve[-SERVE_WINDOW:]
            features["p2_avg_1stServeWinPct"][i] = np.nanmean([s["first_serve_win"] for s in recent])
            features["p2_avg_bpSavedPct"][i] = np.nanmean([s["bp_saved"] for s in recent])
            features["p2_avg_acePct"][i] = np.nanmean([s["ace_pct"] for s in recent])

        if not np.isnan(features["p1_avg_1stServeWinPct"][i]) and not np.isnan(features["p2_avg_1stServeWinPct"][i]):
            features["serve_1stWinPct_diff"][i] = features["p1_avg_1stServeWinPct"][i] - features["p2_avg_1stServeWinPct"][i]
            features["serve_bpSavedPct_diff"][i] = features["p1_avg_bpSavedPct"][i] - features["p2_avg_bpSavedPct"][i]
            features["serve_acePct_diff"][i] = features["p1_avg_acePct"][i] - features["p2_avg_acePct"][i]

        # Player attributes
        if coin[i]:
            p1_age, p2_age = winner_age[i], loser_age[i]
            p1_ht, p2_ht = winner_ht[i], loser_ht[i]
            p1_rank, p2_rank = winner_rank[i], loser_rank[i]
            p1_rpts, p2_rpts = winner_rank_pts[i], loser_rank_pts[i]
        else:
            p1_age, p2_age = loser_age[i], winner_age[i]
            p1_ht, p2_ht = loser_ht[i], winner_ht[i]
            p1_rank, p2_rank = loser_rank[i], winner_rank[i]
            p1_rpts, p2_rpts = loser_rank_pts[i], winner_rank_pts[i]

        features["age_diff"][i] = p1_age - p2_age if not (np.isnan(p1_age) or np.isnan(p2_age)) else np.nan
        features["height_diff"][i] = p1_ht - p2_ht if not (np.isnan(p1_ht) or np.isnan(p2_ht)) else np.nan
        features["rank_diff"][i] = p1_rank - p2_rank if not (np.isnan(p1_rank) or np.isnan(p2_rank)) else np.nan
        features["rank_points_diff"][i] = p1_rpts - p2_rpts if not (np.isnan(p1_rpts) or np.isnan(p2_rpts)) else np.nan

        # Match context
        features["best_of"][i] = best_of[i] if not np.isnan(best_of[i]) else 3
        features["round_num"][i] = ROUND_MAP.get(rounds[i], 3)

        surf = surfaces[i]
        if surf == "Hard":
            features["surface_Hard"][i] = 1
        elif surf == "Clay":
            features["surface_Clay"][i] = 1
        elif surf == "Grass":
            features["surface_Grass"][i] = 1

        tl = tourney_levels[i]
        if tl == "G":
            features["tourney_G"][i] = 1
        elif tl == "M":
            features["tourney_M"][i] = 1
        elif tl in ("A", "B"):
            features["tourney_A"][i] = 1

        # --- NOW update histories (AFTER recording pre-match features) ---
        # Update win/loss records
        player_results[w_id].append((i, True))
        player_results[l_id].append((i, False))

        # Update H2H
        h2h_key = (min(w_id, l_id), max(w_id, l_id))
        if w_id < l_id:
            h2h_record[h2h_key][0] += 1
        else:
            h2h_record[h2h_key][1] += 1

        # Update serve stats for winner
        if not np.isnan(w_svpt[i]) and w_svpt[i] > 0:
            first_serve_win = w_1stWon[i] / w_1stIn[i] if not np.isnan(w_1stIn[i]) and w_1stIn[i] > 0 else np.nan
            bp_saved = w_bpSaved[i] / w_bpFaced[i] if not np.isnan(w_bpFaced[i]) and w_bpFaced[i] > 0 else np.nan
            ace_pct = w_ace[i] / w_svpt[i] if not np.isnan(w_ace[i]) else np.nan
            player_serve_stats[w_id].append({
                "first_serve_win": first_serve_win,
                "bp_saved": bp_saved,
                "ace_pct": ace_pct,
            })

        # Update serve stats for loser
        if not np.isnan(l_svpt[i]) and l_svpt[i] > 0:
            first_serve_win = l_1stWon[i] / l_1stIn[i] if not np.isnan(l_1stIn[i]) and l_1stIn[i] > 0 else np.nan
            bp_saved = l_bpSaved[i] / l_bpFaced[i] if not np.isnan(l_bpFaced[i]) and l_bpFaced[i] > 0 else np.nan
            ace_pct = l_ace[i] / l_svpt[i] if not np.isnan(l_ace[i]) else np.nan
            player_serve_stats[l_id].append({
                "first_serve_win": first_serve_win,
                "bp_saved": bp_saved,
                "ace_pct": ace_pct,
            })

    # Build DataFrame
    feature_df = pd.DataFrame(features)
    # Add date for splitting
    feature_df["date"] = matches["date"].values
    feature_df["surface"] = matches["surface"].values

    # Add player names for reference
    feature_df["winner_name"] = matches["winner_name"].values
    feature_df["loser_name"] = matches["loser_name"].values

    print(f"  Feature engineering complete. Shape: {feature_df.shape}")
    return feature_df


def get_feature_columns():
    """Return list of feature column names used for training."""
    cols = [
        "p1_elo", "p2_elo", "p1_surface_elo", "p2_surface_elo",
        "elo_diff", "surface_elo_diff", "total_elo",
        "h2h_p1_wins", "h2h_p2_wins", "h2h_diff",
        "age_diff", "height_diff", "rank_diff", "rank_points_diff",
        "best_of", "round_num",
        "surface_Hard", "surface_Clay", "surface_Grass",
        "tourney_G", "tourney_M", "tourney_A",
        "p1_avg_1stServeWinPct", "p2_avg_1stServeWinPct",
        "p1_avg_bpSavedPct", "p2_avg_bpSavedPct",
        "p1_avg_acePct", "p2_avg_acePct",
        "serve_1stWinPct_diff", "serve_bpSavedPct_diff", "serve_acePct_diff",
    ]
    for w in ROLLING_WINDOWS:
        cols.extend([
            f"p1_winrate_last_{w}", f"p2_winrate_last_{w}", f"winrate_diff_last_{w}",
        ])
    return cols
