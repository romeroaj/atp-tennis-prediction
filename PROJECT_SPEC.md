# Tennis Match Prediction Model - Full Spec

## Goal
Replicate and extend the tennis prediction model described in @theGreenCoding's viral project. Build a complete, production-ready pipeline that:
1. Processes 40 years of ATP match data
2. Engineers features including a custom ELO rating system
3. Trains XGBoost to predict match winners at ~85% accuracy
4. Can be used to predict upcoming matches for betting purposes

## Data Location
- Match CSVs: `/home/user/workspace/tennis_prediction/data/atp_matches_YYYY.csv` (1985-2024)
- Player info: `/home/user/workspace/tennis_prediction/data/atp_players.csv`
- Rankings: `/home/user/workspace/tennis_prediction/data/atp_rankings_XXs.csv` (80s, 90s, 00s, 10s, 20s)

### Match CSV Columns
`tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date, match_num, winner_id, winner_seed, winner_entry, winner_name, winner_hand, winner_ht, winner_ioc, winner_age, loser_id, loser_seed, loser_entry, loser_name, loser_hand, loser_ht, loser_ioc, loser_age, score, best_of, round, minutes, w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced, l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced, winner_rank, winner_rank_points, loser_rank, loser_rank_points`

### Player CSV Columns
`player_id, first_name, last_name, hand, birth_date, country_code, height`

## Architecture - Create these files:

### 1. `src/data_loader.py`
- Load and concatenate all match CSVs (1985-2024)
- Merge with player info (height, hand, birth_date, country)
- Parse dates, handle missing values
- Sort chronologically by tourney_date + match_num
- Output: cleaned DataFrame with ~95k+ rows

### 2. `src/elo.py` - Custom ELO Rating System
This is the MOST IMPORTANT feature. Implement exactly as described:

**Overall ELO:**
- Every player starts at 1500
- K-factor = 32 (standard)
- Expected score: E_a = 1 / (1 + 10^((R_b - R_a) / 400))
- New rating: R_new = R_old + K * (S - E)
  - S = 1 for win, 0 for loss
- Process matches chronologically
- For each match, record the ELO of BOTH players BEFORE the match is played (pre-match ELO)
- Then update ELOs after the match

**Surface-Specific ELO:**
- Maintain separate ELO ratings for each surface: Hard, Clay, Grass
- Same formula but only update the surface-specific ELO that matches the match surface
- Each player starts at 1500 on each surface

**Output columns to add to each match:**
- `winner_elo`, `loser_elo` (pre-match overall ELO)
- `winner_surface_elo`, `loser_surface_elo` (pre-match surface-specific ELO)
- `elo_diff` (winner_elo - loser_elo, BEFORE we know who wins)
- `surface_elo_diff`

### 3. `src/features.py` - Feature Engineering
For each match, compute these features using ONLY data available BEFORE the match (no data leakage!):

**ELO Features:**
- `p1_elo`, `p2_elo` - overall ELO
- `p1_surface_elo`, `p2_surface_elo` - surface-specific ELO
- `elo_diff` = p1_elo - p2_elo
- `surface_elo_diff` = p1_surface_elo - p2_surface_elo
- `total_elo` = p1_elo + p2_elo

**Head-to-Head:**
- `h2h_p1_wins`, `h2h_p2_wins` - career H2H record up to this match
- `h2h_diff` = h2h_p1_wins - h2h_p2_wins

**Rolling Performance (last N matches: 10, 25, 50, 100):**
- `p1_winrate_last_N`, `p2_winrate_last_N`
- `winrate_diff_last_N`

**Serve Stats (rolling averages from last 20 matches with stats):**
- `p1_avg_1stServeWinPct`, `p2_avg_1stServeWinPct` (w_1stWon / w_1stIn)
- `p1_avg_bpSavedPct`, `p2_avg_bpSavedPct` (w_bpSaved / w_bpFaced)
- `p1_avg_acePct`, `p2_avg_acePct` (w_ace / w_svpt)
- Differentials for each

**Player Attributes:**
- `age_diff` = p1_age - p2_age
- `height_diff` = p1_height - p2_height
- `rank_diff` = p1_rank - p2_rank
- `rank_points_diff`

**Match Context:**
- `surface` (one-hot encoded: Hard, Clay, Grass)
- `tourney_level` (one-hot: G=Grand Slam, M=Masters, A=ATP 500/250, etc.)
- `best_of` (3 or 5)
- `round` (encoded ordinally: R128=1, R64=2, ..., F=7)

**CRITICAL: Avoiding Data Leakage**
- The raw data has winner/loser columns. We must NOT use post-match stats as features
- Randomly assign one player as p1 and the other as p2 for each match
- Target variable: did p1 win? (1 or 0)
- All features must be pre-match only (ELO, H2H, rolling stats from PREVIOUS matches)
- Never use in-match statistics (aces, serve points, etc. FROM THIS MATCH) as features

### 4. `src/train.py` - Model Training
Train and compare these models:

1. **Decision Tree** (sklearn DecisionTreeClassifier)
   - Use GridSearchCV for max_depth, min_samples_split
   
2. **Random Forest** (sklearn RandomForestClassifier)
   - 100 trees, GridSearchCV for hyperparameters
   
3. **XGBoost** (xgboost.XGBClassifier)
   - Use GridSearchCV or Bayesian optimization
   - Key params: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda
   
4. **Neural Network** (sklearn MLPClassifier) - for comparison

**Train/Test Split:**
- Training: all matches through December 2023
- Validation: all matches in 2024 (excluding Australian Open 2025 if data exists)
- Test: Australian Open 2025 or most recent Grand Slam available

**Metrics:** Accuracy, ROC-AUC, log loss, confusion matrix, classification report

**Save:** Best model as pickle/joblib, feature importances, training history

### 5. `src/predict.py` - Prediction Pipeline
Function that takes two player names and a surface, and returns:
- Win probability for each player
- Which features are driving the prediction
- Current ELO ratings for both players
- Head-to-head record

### 6. `main.py` - Orchestration Script
Runs the full pipeline end-to-end:
1. Load data
2. Compute ELOs
3. Engineer features
4. Train models
5. Evaluate
6. Save artifacts

### 7. `predict_match.py` - CLI for predictions
Simple CLI: `python predict_match.py "Jannik Sinner" "Carlos Alcaraz" "Hard"`

## Output Files to Generate
- `/home/user/workspace/tennis_prediction/models/` - saved models
- `/home/user/workspace/tennis_prediction/outputs/model_comparison.csv` - accuracy/AUC for each model
- `/home/user/workspace/tennis_prediction/outputs/feature_importance.png` - XGBoost feature importance chart
- `/home/user/workspace/tennis_prediction/outputs/elo_history.png` - Big Three ELO over time
- `/home/user/workspace/tennis_prediction/outputs/training_report.txt` - full training results
- `/home/user/workspace/tennis_prediction/data/processed_features.csv` - the engineered feature matrix

## Dependencies
Install: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib

## Important Notes
- Process matches in STRICT chronological order for ELO computation
- All features must be computed using only PAST data (no future leakage)
- The p1/p2 randomization is critical to avoid winner/loser bias
- Set random seeds for reproducibility (seed=42)
- Print progress during long operations
- The final model should be ready for predicting real upcoming matches
