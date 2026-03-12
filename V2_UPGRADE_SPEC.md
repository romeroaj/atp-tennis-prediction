# V2 Upgrade Spec — Tennis Prediction Model

## Overview
Upgrade the existing pipeline to push accuracy from ~65.5% toward 68-70%. All changes must maintain strict no-data-leakage discipline.

The existing code is in `/home/user/workspace/tennis_prediction/src/`. You should modify the existing files in-place and add new ones as needed. DO NOT rename or move files — keep the same structure.

## 1. ELO System Upgrades (`src/elo.py`)

Replace the current flat K=32 ELO with an advanced system that incorporates all of the following:

### 1a. Dynamic K-Factor (FiveThirtyEight style)
Instead of flat K=32, use:
```
K = 250 / ((num_matches_played + 5) ^ 0.4)
```
This means new players (few matches) have K~50 (ratings move fast), while veterans with 500+ matches have K~15 (ratings are stable). Track `num_matches_played` per player as you iterate.

### 1b. Margin of Victory Adjustment
Parse the `score` column to extract games won/lost by each player. Compute a margin multiplier:
```
total_games_winner = sum of games won by winner across all sets
total_games_loser = sum of games won by loser across all sets
margin = total_games_winner - total_games_loser
total_games = total_games_winner + total_games_loser
mov_multiplier = log(1 + margin) / log(1 + total_games) + 1.0
```
Clamp mov_multiplier between 0.5 and 2.0. Then:
```
K_effective = K * mov_multiplier
```
A 6-0 6-0 win moves ratings much more than a 7-6 7-6 win.

Score parsing: scores look like "6-4 7-6(5) 6-3" or "6-4 3-6 7-6(3)". Split by space, then by "-". Handle tiebreaks (ignore the number in parentheses). Handle retirements (scores ending with "RET" or "W/O") — for those, use mov_multiplier=1.0 (no adjustment).

### 1c. Weighted ELO (WElo) — Momentum
After computing the standard ELO update, apply an additional momentum adjustment based on the player's LAST match result:
```
If player won their last match:
    momentum_bonus = +5 * (1 - expected_score)  # bigger bonus for beating tough opponents
If player lost their last match:
    momentum_penalty = -3 * expected_score  # bigger penalty for losing to weaker opponents
```
Track each player's last match result as you iterate. Apply momentum as a post-update adjustment to the rating.

### 1d. Inactivity Decay
Track the date of each player's last match. If a player hasn't played in more than 90 days:
```
days_inactive = (current_match_date - last_match_date).days
decay = max(0, (days_inactive - 90)) * 0.5  # lose 0.5 ELO per day after 90 days of inactivity
new_elo = current_elo - decay  # but never below 1400
```

### Output
Same as before: `winner_elo`, `loser_elo`, `winner_surface_elo`, `loser_surface_elo` columns added to matches DataFrame. Also return the `overall_elo` and `surface_elo` dictionaries.

## 2. New Features (`src/features.py`)

Add these features ON TOP of the existing 43 features (don't remove any existing features):

### 2a. Fatigue Features
- `p1_days_since_last_match`, `p2_days_since_last_match` — days since each player's previous match
- `p1_matches_last_7d`, `p2_matches_last_7d` — matches played in last 7 days
- `p1_matches_last_30d`, `p2_matches_last_30d` — matches played in last 30 days
- `fatigue_diff_7d`, `fatigue_diff_30d` — differences

Track each player's match dates chronologically to compute these.

### 2b. Surface-Specific Win Rates
- `p1_surface_winrate`, `p2_surface_winrate` — rolling win rate on the current match surface (last 30 surface-specific matches)
- `surface_winrate_diff`

Track wins/losses per player per surface.

### 2c. Tournament History
- `p1_tourney_winrate`, `p2_tourney_winrate` — historical win rate at this specific tournament
- `tourney_winrate_diff`

Use the `tourney_name` column. Track per-player per-tournament results.

### 2d. Additional Serve Stats
- `p1_avg_2ndServeWinPct`, `p2_avg_2ndServeWinPct` — from w_2ndWon / (w_svpt - w_1stIn)
- `p1_avg_dfPct`, `p2_avg_dfPct` — double fault percentage: w_df / w_svpt
- `serve_2ndWinPct_diff`, `serve_dfPct_diff`

### 2e. Retirement Filtering
Add a boolean column `is_retirement` by checking if score contains "RET", "W/O", "DEF", or "Def.". Don't remove these matches from training but add the flag as a feature so the model can learn from it.

### 2f. Update `get_feature_columns()`
Add ALL new feature column names to the list returned by this function.

## 3. Training Improvements (`src/train.py`)

### 3a. Train on 2005+ Only
Change the training data filter: instead of using all data from 1985, only use matches from 2005 onward for training. But STILL compute ELO ratings from 1985 (so ELO has the full history).

In `split_data()`:
```python
train = df[(df["date"] >= "2005-01-01") & (df["date"] < "2024-01-01")]
test = df[df["date"] >= "2024-01-01"]
```

### 3b. Optuna Hyperparameter Tuning for XGBoost
THIS IS THE MOST IMPORTANT IMPROVEMENT. Replace the GridSearchCV for XGBoost with Optuna.

```python
import optuna
from sklearn.model_selection import TimeSeriesSplit

def train_xgboost_optuna(X_train, y_train, X_test, y_test, n_trials=200):
    """Train XGBoost with Optuna Bayesian optimization."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0, 5.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.2),
        }
        
        model = XGBClassifier(
            **params,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        
        # Use TimeSeriesSplit for cross-validation (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            scores.append(accuracy_score(y_v, model.predict(X_v)))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Train final model with best params
    best_params = study.best_params
    print(f"  Best Optuna params: {best_params}")
    print(f"  Best CV accuracy: {study.best_value:.4f}")
    
    final_model = XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)
    return evaluate_model("XGBoost (Optuna)", final_model, X_test, y_test), study
```

Run 200 trials (takes a few minutes but finds much better hyperparameters than grid search).

### 3c. Add LightGBM
```python
import lightgbm as lgb

def train_lightgbm_optuna(X_train, y_train, X_test, y_test, n_trials=200):
    """Train LightGBM with Optuna."""
    # Similar Optuna structure as XGBoost but with LightGBM params:
    # num_leaves, max_depth, learning_rate, n_estimators, subsample,
    # colsample_bytree, reg_alpha, reg_lambda, min_child_samples
    ...
```

### 3d. Stacking Ensemble
After training XGBoost and LightGBM individually, build a stacking ensemble:
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def train_stacking_ensemble(X_train, y_train, X_test, y_test, xgb_model, lgb_model):
    """Stack XGBoost + LightGBM with Logistic Regression meta-learner."""
    estimators = [
        ('xgb', xgb_model),
        ('lgb', lgb_model),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=TimeSeriesSplit(n_splits=5),
        passthrough=True,  # pass original features to meta-learner too
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    return evaluate_model("Stacking Ensemble", stack, X_test, y_test)
```

### 3e. Rolling Window Cross-Validation
In addition to the single train/test split, also report rolling window CV scores:
```python
tscv = TimeSeriesSplit(n_splits=5)
# Report mean + std of CV accuracy for the best model
```

### 3f. Model Comparison
Train ALL of these and report results:
1. Decision Tree (keep from V1 for comparison baseline)
2. Random Forest (keep from V1)
3. XGBoost with GridSearch (V1 approach — keep for comparison)
4. **XGBoost with Optuna** (V2 — THIS IS THE STAR)
5. **LightGBM with Optuna** (V2)
6. Neural Network (keep from V1)
7. **Stacking Ensemble** (V2)

Save comparison in `outputs/model_comparison_v2.csv`.

## 4. Updated Visualization (`src/visualize.py`)

Add:
- Optuna optimization history plot (accuracy vs trial number) → `outputs/optuna_history.png`
- Optuna parameter importance plot → `outputs/optuna_param_importance.png`
- Updated feature importance for the best model → `outputs/feature_importance_v2.png`
- V1 vs V2 comparison bar chart → `outputs/v1_v2_comparison.png`

## 5. Dependencies

Install: `pip install optuna lightgbm`
(pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib already installed)

## 6. Execution

Update `main.py` to run the V2 pipeline. It should:
1. Load data (all years 1985-2024 for ELO computation)
2. Compute V2 ELO ratings (dynamic K, MOV, WElo, decay)
3. Engineer V2 features (all original + new features)
4. Filter training data to 2005+ only
5. Train all 7 models
6. Print comprehensive comparison
7. Save everything

Run it end to end and print the full results.

## 7. Update `predict_match.py` and `src/predict.py`
Make sure the prediction CLI works with the new V2 model and new features. The predict function needs to compute all the new V2 features for a given matchup.

## CRITICAL NOTES
- Do NOT break the existing data pipeline — extend it
- All ELO computation must start from 1985 (full history) even though training starts from 2005
- All features must be PRE-MATCH only (no leakage)
- P1/P2 randomization must be maintained
- Set random seed=42 everywhere for reproducibility
- Print progress for long-running operations
- Optuna should suppress verbose output (use optuna.logging.set_verbosity(optuna.logging.WARNING))
- The `n_trials=200` for Optuna is important — don't reduce it. It should take 5-10 minutes but finds much better params.
