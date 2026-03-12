"""V2 Feature engineering: all V1 features + fatigue, surface win rates, tourney history, 2nd serve, retirement flag."""

import re
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
SURFACE_WIN_WINDOW = 30


def is_retirement(score_str):
    """Check if a match ended in retirement/walkover."""
    if not isinstance(score_str, str):
        return False
    s = score_str.upper()
    return any(x in s for x in ['RET', 'W/O', 'DEF', 'ABN', 'UNFINISHED'])


def engineer_features(matches, seed=42):
    """
    V2 feature engineering. ALL features use only pre-match data.
    Randomly assign winner/loser as p1/p2. Target: did p1 win? (1 or 0)
    """
    np.random.seed(seed)
    matches = matches.sort_values(["date", "tourney_id", "match_num"]).reset_index(drop=True)
    n = len(matches)

    print(f"Engineering V2 features for {n} matches...")

    coin = np.random.randint(0, 2, size=n).astype(bool)

    # --- Player history structures ---
    player_results = defaultdict(list)  # player_id -> list of (match_idx, won_bool)
    player_serve_stats = defaultdict(list)
    h2h_record = defaultdict(lambda: [0, 0])
    player_match_dates = defaultdict(list)  # for fatigue
    player_surface_results = defaultdict(list)  # (player_id, surface) -> list of won_bool
    player_tourney_results = defaultdict(list)  # (player_id, tourney_name) -> list of won_bool

    # Pre-extract columns
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
    best_of_arr = matches["best_of"].values
    rounds = matches["round"].values
    tourney_levels = matches["tourney_level"].values
    tourney_names = matches["tourney_name"].values
    scores = matches["score"].values

    # Serve stat columns
    def safe_col(col_name):
        return matches[col_name].values if col_name in matches.columns else np.full(n, np.nan)

    w_ace = safe_col("w_ace")
    w_df = safe_col("w_df")
    w_svpt = safe_col("w_svpt")
    w_1stIn = safe_col("w_1stIn")
    w_1stWon = safe_col("w_1stWon")
    w_2ndWon = safe_col("w_2ndWon")
    w_bpSaved = safe_col("w_bpSaved")
    w_bpFaced = safe_col("w_bpFaced")
    l_ace = safe_col("l_ace")
    l_df = safe_col("l_df")
    l_svpt = safe_col("l_svpt")
    l_1stIn = safe_col("l_1stIn")
    l_1stWon = safe_col("l_1stWon")
    l_2ndWon = safe_col("l_2ndWon")
    l_bpSaved = safe_col("l_bpSaved")
    l_bpFaced = safe_col("l_bpFaced")

    # --- Output feature arrays ---
    features = {
        # V1 ELO features
        "p1_elo": np.zeros(n), "p2_elo": np.zeros(n),
        "p1_surface_elo": np.zeros(n), "p2_surface_elo": np.zeros(n),
        "elo_diff": np.zeros(n), "surface_elo_diff": np.zeros(n), "total_elo": np.zeros(n),
        # V1 H2H
        "h2h_p1_wins": np.zeros(n), "h2h_p2_wins": np.zeros(n), "h2h_diff": np.zeros(n),
        # V1 Player attributes
        "age_diff": np.full(n, np.nan), "height_diff": np.full(n, np.nan),
        "rank_diff": np.full(n, np.nan), "rank_points_diff": np.full(n, np.nan),
        # V1 Match context
        "best_of": np.zeros(n), "round_num": np.zeros(n),
        "surface_Hard": np.zeros(n), "surface_Clay": np.zeros(n), "surface_Grass": np.zeros(n),
        "tourney_G": np.zeros(n), "tourney_M": np.zeros(n), "tourney_A": np.zeros(n),
        # V1 Serve stats
        "p1_avg_1stServeWinPct": np.full(n, np.nan), "p2_avg_1stServeWinPct": np.full(n, np.nan),
        "p1_avg_bpSavedPct": np.full(n, np.nan), "p2_avg_bpSavedPct": np.full(n, np.nan),
        "p1_avg_acePct": np.full(n, np.nan), "p2_avg_acePct": np.full(n, np.nan),
        "serve_1stWinPct_diff": np.full(n, np.nan),
        "serve_bpSavedPct_diff": np.full(n, np.nan),
        "serve_acePct_diff": np.full(n, np.nan),
        # V2 NEW: 2nd serve stats
        "p1_avg_2ndServeWinPct": np.full(n, np.nan), "p2_avg_2ndServeWinPct": np.full(n, np.nan),
        "serve_2ndWinPct_diff": np.full(n, np.nan),
        "p1_avg_dfPct": np.full(n, np.nan), "p2_avg_dfPct": np.full(n, np.nan),
        "serve_dfPct_diff": np.full(n, np.nan),
        # V2 NEW: Fatigue
        "p1_days_since_last": np.full(n, np.nan), "p2_days_since_last": np.full(n, np.nan),
        "p1_matches_last_7d": np.zeros(n), "p2_matches_last_7d": np.zeros(n),
        "p1_matches_last_30d": np.zeros(n), "p2_matches_last_30d": np.zeros(n),
        "fatigue_diff_7d": np.zeros(n), "fatigue_diff_30d": np.zeros(n),
        # V2 NEW: Surface win rates
        "p1_surface_winrate": np.full(n, np.nan), "p2_surface_winrate": np.full(n, np.nan),
        "surface_winrate_diff": np.full(n, np.nan),
        # V2 NEW: Tournament history
        "p1_tourney_winrate": np.full(n, np.nan), "p2_tourney_winrate": np.full(n, np.nan),
        "tourney_winrate_diff": np.full(n, np.nan),
        # V2 NEW: Retirement flag
        "is_retirement": np.zeros(n),
        # Target and IDs
        "target": np.zeros(n),
        "p1_id": np.zeros(n, dtype=int), "p2_id": np.zeros(n, dtype=int),
    }

    # Rolling win rate features
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
        match_date = dates[i]
        surf = surfaces[i]
        tname = tourney_names[i]

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

        # --- ELO features (pre-match) ---
        if coin[i]:
            p1_elo_val, p2_elo_val = winner_elo[i], loser_elo[i]
            p1_surf_elo, p2_surf_elo = winner_surface_elo[i], loser_surface_elo[i]
        else:
            p1_elo_val, p2_elo_val = loser_elo[i], winner_elo[i]
            p1_surf_elo, p2_surf_elo = loser_surface_elo[i], winner_surface_elo[i]

        features["p1_elo"][i] = p1_elo_val
        features["p2_elo"][i] = p2_elo_val
        features["p1_surface_elo"][i] = p1_surf_elo
        features["p2_surface_elo"][i] = p2_surf_elo
        features["elo_diff"][i] = p1_elo_val - p2_elo_val
        features["surface_elo_diff"][i] = p1_surf_elo - p2_surf_elo
        features["total_elo"][i] = p1_elo_val + p2_elo_val

        # --- H2H (pre-match) ---
        key = (min(p1_id, p2_id), max(p1_id, p2_id))
        if p1_id < p2_id:
            p1_h2h, p2_h2h = h2h_record[key][0], h2h_record[key][1]
        else:
            p1_h2h, p2_h2h = h2h_record[key][1], h2h_record[key][0]
        features["h2h_p1_wins"][i] = p1_h2h
        features["h2h_p2_wins"][i] = p2_h2h
        features["h2h_diff"][i] = p1_h2h - p2_h2h

        # --- Rolling win rates (pre-match) ---
        for w in ROLLING_WINDOWS:
            p1_res = player_results[p1_id]
            p2_res = player_results[p2_id]
            if len(p1_res) >= 5:
                recent = p1_res[-w:]
                features[f"p1_winrate_last_{w}"][i] = np.mean([r[1] for r in recent])
            if len(p2_res) >= 5:
                recent = p2_res[-w:]
                features[f"p2_winrate_last_{w}"][i] = np.mean([r[1] for r in recent])
            if len(p1_res) >= 5 and len(p2_res) >= 5:
                features[f"winrate_diff_last_{w}"][i] = (
                    features[f"p1_winrate_last_{w}"][i] - features[f"p2_winrate_last_{w}"][i]
                )

        # --- Serve stats (pre-match) ---
        for pid_label, pid_val, serve_list in [("p1", p1_id, player_serve_stats[p1_id]),
                                                 ("p2", p2_id, player_serve_stats[p2_id])]:
            if len(serve_list) >= 3:
                recent = serve_list[-SERVE_WINDOW:]
                features[f"{pid_label}_avg_1stServeWinPct"][i] = np.nanmean([s["first_serve_win"] for s in recent])
                features[f"{pid_label}_avg_bpSavedPct"][i] = np.nanmean([s["bp_saved"] for s in recent])
                features[f"{pid_label}_avg_acePct"][i] = np.nanmean([s["ace_pct"] for s in recent])
                features[f"{pid_label}_avg_2ndServeWinPct"][i] = np.nanmean([s["second_serve_win"] for s in recent])
                features[f"{pid_label}_avg_dfPct"][i] = np.nanmean([s["df_pct"] for s in recent])

        # Serve diffs
        for stat in ["1stServeWinPct", "bpSavedPct", "acePct", "2ndServeWinPct", "dfPct"]:
            p1_val = features[f"p1_avg_{stat}"][i]
            p2_val = features[f"p2_avg_{stat}"][i]
            diff_key = f"serve_{stat}_diff"
            if diff_key in features and not np.isnan(p1_val) and not np.isnan(p2_val):
                features[diff_key][i] = p1_val - p2_val

        # --- Player attributes ---
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

        if not (np.isnan(p1_age) or np.isnan(p2_age)):
            features["age_diff"][i] = p1_age - p2_age
        if not (np.isnan(p1_ht) or np.isnan(p2_ht)):
            features["height_diff"][i] = p1_ht - p2_ht
        if not (np.isnan(p1_rank) or np.isnan(p2_rank)):
            features["rank_diff"][i] = p1_rank - p2_rank
        if not (np.isnan(p1_rpts) or np.isnan(p2_rpts)):
            features["rank_points_diff"][i] = p1_rpts - p2_rpts

        # --- Match context ---
        features["best_of"][i] = best_of_arr[i] if not np.isnan(best_of_arr[i]) else 3
        features["round_num"][i] = ROUND_MAP.get(rounds[i], 3)
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

        # --- V2: Retirement flag ---
        features["is_retirement"][i] = 1 if is_retirement(scores[i]) else 0

        # --- V2: Fatigue features (pre-match) ---
        for pid_label, pid_val in [("p1", p1_id), ("p2", p2_id)]:
            pdates = player_match_dates[pid_val]
            if len(pdates) > 0:
                try:
                    last_d = pd.Timestamp(pdates[-1])
                    curr_d = pd.Timestamp(match_date)
                    days_since = (curr_d - last_d).days
                    features[f"{pid_label}_days_since_last"][i] = days_since
                except:
                    pass

                # Matches in last 7 and 30 days
                count_7 = 0
                count_30 = 0
                for d in reversed(pdates):
                    try:
                        diff = (pd.Timestamp(match_date) - pd.Timestamp(d)).days
                    except:
                        continue
                    if diff <= 7:
                        count_7 += 1
                    if diff <= 30:
                        count_30 += 1
                    else:
                        break
                features[f"{pid_label}_matches_last_7d"][i] = count_7
                features[f"{pid_label}_matches_last_30d"][i] = count_30

        features["fatigue_diff_7d"][i] = features["p1_matches_last_7d"][i] - features["p2_matches_last_7d"][i]
        features["fatigue_diff_30d"][i] = features["p1_matches_last_30d"][i] - features["p2_matches_last_30d"][i]

        # --- V2: Surface-specific win rate (pre-match) ---
        for pid_label, pid_val in [("p1", p1_id), ("p2", p2_id)]:
            sresults = player_surface_results[(pid_val, surf)]
            if len(sresults) >= 5:
                recent = sresults[-SURFACE_WIN_WINDOW:]
                features[f"{pid_label}_surface_winrate"][i] = np.mean(recent)
        if not np.isnan(features["p1_surface_winrate"][i]) and not np.isnan(features["p2_surface_winrate"][i]):
            features["surface_winrate_diff"][i] = features["p1_surface_winrate"][i] - features["p2_surface_winrate"][i]

        # --- V2: Tournament history (pre-match) ---
        for pid_label, pid_val in [("p1", p1_id), ("p2", p2_id)]:
            tresults = player_tourney_results[(pid_val, tname)]
            if len(tresults) >= 3:
                features[f"{pid_label}_tourney_winrate"][i] = np.mean(tresults)
        if not np.isnan(features["p1_tourney_winrate"][i]) and not np.isnan(features["p2_tourney_winrate"][i]):
            features["tourney_winrate_diff"][i] = features["p1_tourney_winrate"][i] - features["p2_tourney_winrate"][i]

        # ===== NOW update histories (AFTER recording pre-match features) =====
        player_results[w_id].append((i, True))
        player_results[l_id].append((i, False))

        # H2H
        h2h_key = (min(w_id, l_id), max(w_id, l_id))
        if w_id < l_id:
            h2h_record[h2h_key][0] += 1
        else:
            h2h_record[h2h_key][1] += 1

        # Match dates for fatigue
        player_match_dates[w_id].append(match_date)
        player_match_dates[l_id].append(match_date)

        # Surface results
        player_surface_results[(w_id, surf)].append(True)
        player_surface_results[(l_id, surf)].append(False)

        # Tournament results
        player_tourney_results[(w_id, tname)].append(True)
        player_tourney_results[(l_id, tname)].append(False)

        # Serve stats (winner)
        if not np.isnan(w_svpt[i]) and w_svpt[i] > 0:
            fsw = w_1stWon[i] / w_1stIn[i] if not np.isnan(w_1stIn[i]) and w_1stIn[i] > 0 else np.nan
            bp = w_bpSaved[i] / w_bpFaced[i] if not np.isnan(w_bpFaced[i]) and w_bpFaced[i] > 0 else np.nan
            ace = w_ace[i] / w_svpt[i] if not np.isnan(w_ace[i]) else np.nan
            second_pts = w_svpt[i] - (w_1stIn[i] if not np.isnan(w_1stIn[i]) else 0)
            ssw = w_2ndWon[i] / second_pts if not np.isnan(w_2ndWon[i]) and second_pts > 0 else np.nan
            dfp = w_df[i] / w_svpt[i] if not np.isnan(w_df[i]) else np.nan
            player_serve_stats[w_id].append({
                "first_serve_win": fsw, "bp_saved": bp, "ace_pct": ace,
                "second_serve_win": ssw, "df_pct": dfp,
            })

        # Serve stats (loser)
        if not np.isnan(l_svpt[i]) and l_svpt[i] > 0:
            fsw = l_1stWon[i] / l_1stIn[i] if not np.isnan(l_1stIn[i]) and l_1stIn[i] > 0 else np.nan
            bp = l_bpSaved[i] / l_bpFaced[i] if not np.isnan(l_bpFaced[i]) and l_bpFaced[i] > 0 else np.nan
            ace = l_ace[i] / l_svpt[i] if not np.isnan(l_ace[i]) else np.nan
            second_pts = l_svpt[i] - (l_1stIn[i] if not np.isnan(l_1stIn[i]) else 0)
            ssw = l_2ndWon[i] / second_pts if not np.isnan(l_2ndWon[i]) and second_pts > 0 else np.nan
            dfp = l_df[i] / l_svpt[i] if not np.isnan(l_df[i]) else np.nan
            player_serve_stats[l_id].append({
                "first_serve_win": fsw, "bp_saved": bp, "ace_pct": ace,
                "second_serve_win": ssw, "df_pct": dfp,
            })

    # Build DataFrame
    feature_df = pd.DataFrame(features)
    feature_df["date"] = matches["date"].values
    feature_df["surface_name"] = matches["surface"].values
    feature_df["winner_name"] = matches["winner_name"].values
    feature_df["loser_name"] = matches["loser_name"].values

    print(f"  V2 feature engineering complete. Shape: {feature_df.shape}")
    return feature_df


def get_feature_columns():
    """Return list of ALL V2 feature column names used for training."""
    cols = [
        # V1 ELO
        "p1_elo", "p2_elo", "p1_surface_elo", "p2_surface_elo",
        "elo_diff", "surface_elo_diff", "total_elo",
        # V1 H2H
        "h2h_p1_wins", "h2h_p2_wins", "h2h_diff",
        # V1 Player attributes
        "age_diff", "height_diff", "rank_diff", "rank_points_diff",
        # V1 Match context
        "best_of", "round_num",
        "surface_Hard", "surface_Clay", "surface_Grass",
        "tourney_G", "tourney_M", "tourney_A",
        # V1 Serve
        "p1_avg_1stServeWinPct", "p2_avg_1stServeWinPct",
        "p1_avg_bpSavedPct", "p2_avg_bpSavedPct",
        "p1_avg_acePct", "p2_avg_acePct",
        "serve_1stWinPct_diff", "serve_bpSavedPct_diff", "serve_acePct_diff",
        # V2 2nd serve
        "p1_avg_2ndServeWinPct", "p2_avg_2ndServeWinPct", "serve_2ndWinPct_diff",
        "p1_avg_dfPct", "p2_avg_dfPct", "serve_dfPct_diff",
        # V2 Fatigue
        "p1_days_since_last", "p2_days_since_last",
        "p1_matches_last_7d", "p2_matches_last_7d",
        "p1_matches_last_30d", "p2_matches_last_30d",
        "fatigue_diff_7d", "fatigue_diff_30d",
        # V2 Surface win rate
        "p1_surface_winrate", "p2_surface_winrate", "surface_winrate_diff",
        # V2 Tournament history
        "p1_tourney_winrate", "p2_tourney_winrate", "tourney_winrate_diff",
        # V2 Retirement flag
        "is_retirement",
    ]
    # Rolling win rates
    for w in ROLLING_WINDOWS:
        cols.extend([
            f"p1_winrate_last_{w}", f"p2_winrate_last_{w}", f"winrate_diff_last_{w}",
        ])
    return cols
