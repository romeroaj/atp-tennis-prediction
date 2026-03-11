"""Custom ELO rating system: overall + surface-specific."""

import numpy as np
import pandas as pd
from collections import defaultdict


K_FACTOR = 32
INITIAL_ELO = 1500


def expected_score(rating_a, rating_b):
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating, actual_score, expected):
    """Compute new ELO rating."""
    return rating + K_FACTOR * (actual_score - expected)


def compute_elo_ratings(matches):
    """
    Process matches chronologically and compute pre-match ELO ratings.

    Adds columns: winner_elo, loser_elo, winner_surface_elo, loser_surface_elo

    IMPORTANT: ELO values recorded are PRE-MATCH (before the result is known).
    """
    # Ensure sorted chronologically
    matches = matches.sort_values(["date", "tourney_id", "match_num"]).reset_index(drop=True)

    # Overall ELO ratings
    overall_elo = defaultdict(lambda: INITIAL_ELO)
    # Surface-specific ELO ratings: {(player_id, surface): elo}
    surface_elo = defaultdict(lambda: INITIAL_ELO)

    # Pre-allocate output arrays
    n = len(matches)
    winner_elo_arr = np.zeros(n)
    loser_elo_arr = np.zeros(n)
    winner_surface_elo_arr = np.zeros(n)
    loser_surface_elo_arr = np.zeros(n)

    # Extract columns as arrays for speed
    winner_ids = matches["winner_id"].values
    loser_ids = matches["loser_id"].values
    surfaces = matches["surface"].values

    print(f"Computing ELO ratings for {n} matches...")
    for i in range(n):
        if i % 20000 == 0 and i > 0:
            print(f"  Processed {i}/{n} matches...")

        w_id = int(winner_ids[i])
        l_id = int(loser_ids[i])
        surf = surfaces[i]

        # Record PRE-MATCH ELO ratings
        w_overall = overall_elo[w_id]
        l_overall = overall_elo[l_id]
        w_surf = surface_elo[(w_id, surf)]
        l_surf = surface_elo[(l_id, surf)]

        winner_elo_arr[i] = w_overall
        loser_elo_arr[i] = l_overall
        winner_surface_elo_arr[i] = w_surf
        loser_surface_elo_arr[i] = l_surf

        # Update overall ELO
        exp_w = expected_score(w_overall, l_overall)
        overall_elo[w_id] = update_elo(w_overall, 1.0, exp_w)
        overall_elo[l_id] = update_elo(l_overall, 0.0, 1.0 - exp_w)

        # Update surface-specific ELO
        exp_w_surf = expected_score(w_surf, l_surf)
        surface_elo[(w_id, surf)] = update_elo(w_surf, 1.0, exp_w_surf)
        surface_elo[(l_id, surf)] = update_elo(l_surf, 0.0, 1.0 - exp_w_surf)

    matches["winner_elo"] = winner_elo_arr
    matches["loser_elo"] = loser_elo_arr
    matches["winner_surface_elo"] = winner_surface_elo_arr
    matches["loser_surface_elo"] = loser_surface_elo_arr

    print(f"  ELO computation complete.")
    print(f"  Unique players with ELO: {len(overall_elo)}")
    return matches, overall_elo, surface_elo


def get_player_elo_history(matches, player_id):
    """Get the ELO history for a specific player over time."""
    # Matches where player was winner
    as_winner = matches[matches["winner_id"] == player_id][["date", "winner_elo"]].copy()
    as_winner = as_winner.rename(columns={"winner_elo": "elo"})

    # Matches where player was loser
    as_loser = matches[matches["loser_id"] == player_id][["date", "loser_elo"]].copy()
    as_loser = as_loser.rename(columns={"loser_elo": "elo"})

    history = pd.concat([as_winner, as_loser]).sort_values("date").reset_index(drop=True)
    return history
