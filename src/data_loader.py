"""Load and clean ATP match data (1985-2024)."""

import glob
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_matches(data_dir=DATA_DIR, year_start=1985, year_end=2024):
    """Load and concatenate all ATP match CSVs, sorted chronologically."""
    files = sorted(glob.glob(os.path.join(data_dir, "atp_matches_*.csv")))
    files = [
        f for f in files
        if year_start <= int(os.path.basename(f).split("_")[-1].split(".")[0]) <= year_end
    ]
    print(f"Loading {len(files)} match files ({year_start}-{year_end})...")

    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

    matches = pd.concat(dfs, ignore_index=True)
    print(f"  Raw rows: {len(matches)}")
    return matches


def load_players(data_dir=DATA_DIR):
    """Load player info."""
    path = os.path.join(data_dir, "atp_players.csv")
    players = pd.read_csv(path, low_memory=False)
    players = players.rename(columns={
        "name_first": "first_name",
        "name_last": "last_name",
        "dob": "birth_date",
        "ioc": "country_code",
    })
    players["full_name"] = players["first_name"].fillna("") + " " + players["last_name"].fillna("")
    players["full_name"] = players["full_name"].str.strip()
    return players


def clean_matches(matches):
    """Clean and prepare match data."""
    # Parse tourney_date as proper date
    matches["tourney_date"] = pd.to_numeric(matches["tourney_date"], errors="coerce")
    matches = matches.dropna(subset=["tourney_date"])
    matches["tourney_date"] = matches["tourney_date"].astype(int).astype(str)
    matches["date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d", errors="coerce")
    matches = matches.dropna(subset=["date"])

    # Ensure player IDs are valid
    matches = matches.dropna(subset=["winner_id", "loser_id"])
    matches["winner_id"] = matches["winner_id"].astype(int)
    matches["loser_id"] = matches["loser_id"].astype(int)

    # Normalize surface
    matches["surface"] = matches["surface"].fillna("Hard")
    matches["surface"] = matches["surface"].replace({"Carpet": "Hard"})

    # Sort chronologically
    matches = matches.sort_values(["date", "tourney_id", "match_num"]).reset_index(drop=True)

    # Remove walkovers/retirements with no score
    matches = matches[matches["score"].notna()].reset_index(drop=True)
    # Remove Davis Cup and other non-standard events for cleaner data
    matches = matches[matches["tourney_level"].isin(["G", "M", "A", "B", "F", "D"])].reset_index(drop=True)

    print(f"  Cleaned rows: {len(matches)}")
    return matches


def load_and_clean(data_dir=DATA_DIR):
    """Full pipeline: load, clean, return matches and players."""
    matches = load_matches(data_dir)
    matches = clean_matches(matches)
    players = load_players(data_dir)
    print(f"  Players loaded: {len(players)}")
    return matches, players
