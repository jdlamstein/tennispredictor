"""End-to-end backtesting script.

Trains a model on pre-holdout ATP data, then simulates fractional-Kelly
betting against Pinnacle odds from tennis-data.co.uk over the holdout period.

Key design decisions
--------------------
1. Year-based split (not %-based): all matches before HOLDOUT_YEAR train,
   all matches from HOLDOUT_YEAR onwards test. This gives exact chronological
   separation and matches the odds data date coverage.
2. Normalization uses ONLY training statistics (no leakage).
3. Odds matching is on (year, p1_surname, p2_surname) rather than exact date
   because atp_database uses tournament START date while tennis-data.co.uk
   uses the actual match date — these rarely coincide.

Usage
-----
    poetry run python scripts/backtest.py

Environment variables
---------------------
    ATP_ROOTDIR   Parent dir (default: ~/Data/tennis)
    MODEL_TYPE    naive_bayes | adaboost | xgboost (default: naive_bayes)
    HOLDOUT_YEAR  First holdout year (default: 2022)
    KELLY         Fractional Kelly multiplier (default: 0.25)
    MIN_EV        Minimum EV threshold (default: 0.02)
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ATP_ROOTDIR = os.path.expanduser(os.environ.get("ATP_ROOTDIR", "~/Data/tennis"))
ATP_DB = os.path.join(ATP_ROOTDIR, "tennis_data", "atp_database.csv")
ODDS_DIR = os.path.join(ATP_ROOTDIR, "tennis_data", "odds")

MODEL_TYPE = os.environ.get("MODEL_TYPE", "naive_bayes")
HOLDOUT_YEAR = int(os.environ.get("HOLDOUT_YEAR", "2022"))
KELLY = float(os.environ.get("KELLY", "0.25"))
MIN_EV = float(os.environ.get("MIN_EV", "0.02"))

# Columns dropped from features (mirrors Dataspring.process_df)
_DROP_PATTERNS = [
    "player1_score", "player2_score", "player1_rank", "player2_rank",
    "_ace", "_df", "_svpt", "_1stIn", "_1stWon", "_2ndWon",
    "_SvGms", "_bpSaved", "_bpFaced",
]
_DROP_EXACT = ["tourney_id", "month", "day", "minutes", "tourney_date",
               "game_winner", "player1_name", "player2_name",
               "player1_id", "player2_id", "tourney_name",
               "Unnamed: 0", "Unnamed: 0.1", "yday"]


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-feature columns and return (X, y)."""
    df = df.copy().sort_values("tourney_date").reset_index(drop=True)
    df = df.fillna(-10)

    # Extract labels before dropping (game_winner also appears in _DROP_EXACT)
    y = df["game_winner"].values - 1   # 0 or 1

    # Drop by exact name
    drop_cols = [c for c in _DROP_EXACT if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Drop by pattern
    for pat in _DROP_PATTERNS:
        df = df.drop(columns=[c for c in df.columns if pat in c], errors="ignore")

    # game_winner already extracted above; remove if somehow still present
    df = df.drop(columns=["game_winner"], errors="ignore")

    return df.values.astype(float), y, df.columns.tolist()


def match_odds(test_df: pd.DataFrame, odds_df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Join predictions with odds on (year, p1_surname, p2_surname).

    tennis-data.co.uk names: "Djokovic N." → surname = first token.
    atp_database names:      "Novak Djokovic" → surname = last token.

    Tries both player orderings (winner/loser may be swapped vs. odds file).
    """
    test = test_df.copy().reset_index(drop=True)
    # probs[:, 0] = P(class 0) = P(game_winner=1) = P(player1 wins)
    test["p1_win_prob"] = np.clip(probs[:, 0], 1e-6, 1.0 - 1e-6)
    test["match_year"] = test["tourney_date"] // 10000
    test["p1_sn"] = test["player1_name"].fillna("").str.split().str[-1].str.lower()
    test["p2_sn"] = test["player2_name"].fillna("").str.split().str[-1].str.lower()

    odds = odds_df.copy()
    odds["match_year"] = odds["date_int"] // 10000
    odds["w_sn"] = odds["p1_name"].str.split().str[0].str.lower()
    odds["l_sn"] = odds["p2_name"].str.split().str[0].str.lower()

    slim = odds[["match_year", "w_sn", "l_sn", "p1_odds", "p2_odds"]]

    # Case A: our p1 = odds winner
    ma = test.merge(
        slim.rename(columns={"w_sn": "p1_sn", "l_sn": "p2_sn"}),
        on=["match_year", "p1_sn", "p2_sn"], how="inner",
    )
    ma["actual_winner"] = 1

    # Case B: our p1 = odds loser (names swapped).
    # Odds are swapped so p1_odds = PSL (loser's odds), p2_odds = PSW (winner's odds).
    # p1_win_prob stays as-is (model's P(player1 wins), which is low since p1 is the loser).
    # actual_winner = 2 because our p1 lost → player2 won.
    # Backtester will correctly compute high EV for betting on p2 (the winner).
    mb = test.merge(
        slim.rename(columns={"w_sn": "p2_sn", "l_sn": "p1_sn",
                              "p1_odds": "p2_odds_x", "p2_odds": "p1_odds_x"}),
        on=["match_year", "p1_sn", "p2_sn"], how="inner",
    )
    if not mb.empty and "p1_odds_x" in mb.columns:
        mb = mb.rename(columns={"p1_odds_x": "p1_odds", "p2_odds_x": "p2_odds"})
        mb["actual_winner"] = 2

    merged = pd.concat([ma, mb], ignore_index=True)
    merged = merged.drop_duplicates(subset=["match_year", "p1_sn", "p2_sn"])

    # Add date column for backtester (uses match_year as proxy)
    merged["date"] = pd.to_datetime(
        merged["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    return merged


def main() -> None:
    from src.data.odds_loader import load_odds_dir
    from src.evaluation.backtester import BacktestConfig, run, summary
    from src.models.classifiers import make_classifier, CLASSIFIER_REGISTRY
    from src.models.xgboost_model import XGBoostPredictor

    print("=== Tennis Betting Backtest ===")
    print(f"Model      : {MODEL_TYPE}")
    print(f"Holdout    : {HOLDOUT_YEAR}+")
    print(f"Kelly frac : {KELLY}  |  Min EV: {MIN_EV}")
    print()

    # 1. Load raw data
    print("Loading ATP database...")
    raw = pd.read_csv(ATP_DB, low_memory=False)
    raw["year_col"] = raw["tourney_date"] // 10000
    train_raw = raw[raw["year_col"] < HOLDOUT_YEAR].copy()
    test_raw  = raw[raw["year_col"] >= HOLDOUT_YEAR].copy()
    print(f"  Train: {len(train_raw):,} rows  |  Test: {len(test_raw):,} rows")

    print("Loading odds data...")
    odds_df = load_odds_dir(ODDS_DIR)
    holdout_odds = odds_df[odds_df["date_int"] // 10000 >= HOLDOUT_YEAR]
    print(f"  Odds total: {len(odds_df):,}  |  Holdout odds: {len(holdout_odds):,}")

    # 2. Prepare features
    print("Building feature matrices...")
    X_train, y_train, feat_cols = _prepare_features(train_raw)
    X_test,  y_test,  _          = _prepare_features(test_raw)
    print(f"  Features: {X_train.shape[1]}  |  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Normalize using ONLY training statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 3. Train model
    print(f"Training {MODEL_TYPE}...")
    if MODEL_TYPE == "xgboost":
        model = XGBoostPredictor()
        split = int(len(X_train) * 0.8)
        model.fit(X_train[:split], y_train[:split],
                  X_val=X_train[split:], y_val=y_train[split:])
    elif MODEL_TYPE in CLASSIFIER_REGISTRY:
        model = make_classifier(MODEL_TYPE)
        model.fit(X_train, y_train)
    else:
        print(f"Unknown model. Available: {list(CLASSIFIER_REGISTRY)} + xgboost")
        return

    # Quick accuracy check
    test_preds = np.argmax(model.predict_proba(X_test), axis=1)
    accuracy = float(np.mean(test_preds == y_test)) * 100
    print(f"  Holdout accuracy: {accuracy:.1f}%")

    # 4. Match predictions with odds
    test_probs = model.predict_proba(X_test)   # (n, 2)
    print("Matching predictions to bookmaker odds...")
    matched = match_odds(test_raw, holdout_odds, test_probs)
    if matched.empty:
        print("No matches found. Check name format in both datasets.")
        return
    print(f"  Matched {len(matched):,} rows")

    # 5. Run backtest
    print(f"\nRunning backtest on {len(matched):,} matched matches...\n")
    cfg = BacktestConfig(
        initial_bankroll=1000.0,
        kelly_fraction=KELLY,
        min_ev=MIN_EV,
    )
    result = run(matched, cfg)
    print(summary(result))


if __name__ == "__main__":
    main()
