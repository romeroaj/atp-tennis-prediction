"""V2 ELO rating system: dynamic K-factor, margin of victory, WElo momentum, inactivity decay."""

import re
import numpy as np
import pandas as pd
from collections import defaultdict


INITIAL_ELO = 1500
MIN_ELO = 1400


def expected_score(rating_a, rating_b):
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def dynamic_k_factor(num_matches):
    """FiveThirtyEight-style dynamic K: high for new players, low for veterans."""
    return 250.0 / ((num_matches + 5) ** 0.4)


def parse_score_margin(score_str):
    """Parse a tennis score string and return (winner_games, loser_games).
    
    Handles: '6-4 7-6(5) 6-3', '6-4 3-6 7-6(3)', retirements, walkovers.
    Returns (None, None) if unparseable.
    """
    if not isinstance(score_str, str):
        return None, None
    
    # Check for retirement/walkover — return None to signal no MOV adjustment
    score_upper = score_str.upper()
    if any(x in score_upper for x in ['RET', 'W/O', 'DEF', 'ABN', 'UNP', 'UNFINISHED']):
        return None, None
    
    w_games = 0
    l_games = 0
    # Split into sets
    sets = score_str.strip().split()
    for s in sets:
        # Remove tiebreak info: "7-6(5)" -> "7-6"
        s_clean = re.sub(r'\(.*?\)', '', s).strip()
        if '-' not in s_clean:
            continue
        parts = s_clean.split('-')
        if len(parts) != 2:
            continue
        try:
            g1 = int(parts[0])
            g2 = int(parts[1])
            w_games += g1
            l_games += g2
        except ValueError:
            continue
    
    if w_games == 0 and l_games == 0:
        return None, None
    return w_games, l_games


def margin_of_victory_multiplier(score_str):
    """Compute MOV multiplier from score. Returns 1.0 if unparseable."""
    w_games, l_games = parse_score_margin(score_str)
    if w_games is None:
        return 1.0
    
    margin = w_games - l_games
    total = w_games + l_games
    if total <= 0 or margin <= 0:
        return 1.0
    
    mov = np.log(1 + margin) / np.log(1 + total) + 1.0
    return np.clip(mov, 0.5, 2.0)


def compute_elo_ratings(matches):
    """
    V2 ELO: dynamic K, margin of victory, WElo momentum, inactivity decay.
    
    Processes matches chronologically. Records PRE-MATCH ELO ratings.
    """
    matches = matches.sort_values(["date", "tourney_id", "match_num"]).reset_index(drop=True)
    n = len(matches)

    # Overall ELO ratings
    overall_elo = defaultdict(lambda: INITIAL_ELO)
    # Surface-specific ELO
    surface_elo = defaultdict(lambda: INITIAL_ELO)
    # Match count per player (for dynamic K)
    match_count = defaultdict(int)
    match_count_surface = defaultdict(int)
    # Last match result per player: True=won, False=lost (for WElo momentum)
    last_result = {}
    # Last expected score per player (for momentum scaling)
    last_expected = {}
    # Last match date per player (for inactivity decay)
    last_match_date = {}

    # Pre-allocate output arrays
    winner_elo_arr = np.zeros(n)
    loser_elo_arr = np.zeros(n)
    winner_surface_elo_arr = np.zeros(n)
    loser_surface_elo_arr = np.zeros(n)

    # Extract columns as arrays for speed
    winner_ids = matches["winner_id"].values
    loser_ids = matches["loser_id"].values
    surfaces = matches["surface"].values
    scores = matches["score"].values
    dates = matches["date"].values

    print(f"Computing V2 ELO ratings for {n} matches...")
    for i in range(n):
        if i % 20000 == 0 and i > 0:
            print(f"  Processed {i}/{n} matches...")

        w_id = int(winner_ids[i])
        l_id = int(loser_ids[i])
        surf = surfaces[i]
        match_date = dates[i]

        # --- Inactivity decay (before recording pre-match ELO) ---
        for pid in [w_id, l_id]:
            if pid in last_match_date:
                try:
                    days_inactive = (pd.Timestamp(match_date) - pd.Timestamp(last_match_date[pid])).days
                except:
                    days_inactive = 0
                if days_inactive > 90:
                    decay = (days_inactive - 90) * 0.5
                    overall_elo[pid] = max(MIN_ELO, overall_elo[pid] - decay)
                    surface_elo[(pid, surf)] = max(MIN_ELO, surface_elo[(pid, surf)] - decay * 0.5)

        # --- Record PRE-MATCH ELO ratings ---
        w_overall = overall_elo[w_id]
        l_overall = overall_elo[l_id]
        w_surf = surface_elo[(w_id, surf)]
        l_surf = surface_elo[(l_id, surf)]

        winner_elo_arr[i] = w_overall
        loser_elo_arr[i] = l_overall
        winner_surface_elo_arr[i] = w_surf
        loser_surface_elo_arr[i] = l_surf

        # --- Compute K-factors (dynamic) ---
        k_w = dynamic_k_factor(match_count[w_id])
        k_l = dynamic_k_factor(match_count[l_id])
        k_w_surf = dynamic_k_factor(match_count_surface[(w_id, surf)])
        k_l_surf = dynamic_k_factor(match_count_surface[(l_id, surf)])

        # --- Margin of victory multiplier ---
        mov = margin_of_victory_multiplier(scores[i])

        # --- Update overall ELO ---
        exp_w = expected_score(w_overall, l_overall)
        overall_elo[w_id] = w_overall + k_w * mov * (1.0 - exp_w)
        overall_elo[l_id] = l_overall + k_l * mov * (0.0 - (1.0 - exp_w))

        # --- Update surface-specific ELO ---
        exp_w_surf = expected_score(w_surf, l_surf)
        surface_elo[(w_id, surf)] = w_surf + k_w_surf * mov * (1.0 - exp_w_surf)
        surface_elo[(l_id, surf)] = l_surf + k_l_surf * mov * (0.0 - (1.0 - exp_w_surf))

        # --- WElo Momentum adjustment ---
        # Winner momentum
        if w_id in last_result:
            if last_result[w_id]:  # won last match too
                bonus = 5 * (1.0 - exp_w)
                overall_elo[w_id] += bonus
            # no penalty for winner
        # Loser momentum
        if l_id in last_result:
            if not last_result[l_id]:  # lost last match too
                penalty = 3 * (1.0 - exp_w)
                overall_elo[l_id] = max(MIN_ELO, overall_elo[l_id] - penalty)

        # Ensure min ELO
        overall_elo[w_id] = max(MIN_ELO, overall_elo[w_id])
        overall_elo[l_id] = max(MIN_ELO, overall_elo[l_id])
        surface_elo[(w_id, surf)] = max(MIN_ELO, surface_elo[(w_id, surf)])
        surface_elo[(l_id, surf)] = max(MIN_ELO, surface_elo[(l_id, surf)])

        # --- Update tracking state ---
        match_count[w_id] += 1
        match_count[l_id] += 1
        match_count_surface[(w_id, surf)] += 1
        match_count_surface[(l_id, surf)] += 1
        last_result[w_id] = True
        last_result[l_id] = False
        last_expected[w_id] = exp_w
        last_expected[l_id] = 1.0 - exp_w
        last_match_date[w_id] = match_date
        last_match_date[l_id] = match_date

    matches = matches.copy()
    matches["winner_elo"] = winner_elo_arr
    matches["loser_elo"] = loser_elo_arr
    matches["winner_surface_elo"] = winner_surface_elo_arr
    matches["loser_surface_elo"] = loser_surface_elo_arr

    print(f"  V2 ELO computation complete.")
    print(f"  Unique players with ELO: {len(overall_elo)}")
    return matches, dict(overall_elo), {k: v for k, v in surface_elo.items()}


def get_player_elo_history(matches, player_id):
    """Get the ELO history for a specific player over time."""
    as_winner = matches[matches["winner_id"] == player_id][["date", "winner_elo"]].copy()
    as_winner = as_winner.rename(columns={"winner_elo": "elo"})
    as_loser = matches[matches["loser_id"] == player_id][["date", "loser_elo"]].copy()
    as_loser = as_loser.rename(columns={"loser_elo": "elo"})
    history = pd.concat([as_winner, as_loser]).sort_values("date").reset_index(drop=True)
    return history
