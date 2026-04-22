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
    ATP_DB        Full path to database CSV (default: ATP_ROOTDIR/tennis_data/atp_database.csv)
                  Recommended: use atp_database_enriched.csv for pre-match surface ELO +
                  Glicko-2 features. The base player1_elo/player2_elo columns are dropped
                  automatically (post-match storage — see _DROP_EXACT).
    MODEL_TYPE    naive_bayes | adaboost | xgboost (default: naive_bayes)
    HOLDOUT_YEAR  First holdout year (default: 2022)
    KELLY           Fractional Kelly multiplier (default: 0.25)
    MIN_EV          Minimum EV threshold (default: 0.02)
    MARKET_ALPHA    Model weight in blend [0, 1] (default: 1.0 = pure model).
                    0.0 = pure de-vigged market odds. 0.3–0.7 = typical ensemble range.
    CALIBRATE       Apply temperature scaling before ensemble blend (default: 0).
                    Carves off CALIBRATE_SPLIT fraction of training data as cal set.
    CALIBRATE_SPLIT Fraction of training data reserved for calibration (default: 0.2).
    MAX_KELLY       Hard cap on stake per bet as fraction of bankroll (default: 0.05).
                    Set to "none" for uncapped theoretical maximum.
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
_default_db = os.path.join(ATP_ROOTDIR, "tennis_data", "atp_database.csv")
ATP_DB = os.path.expanduser(os.environ.get("ATP_DB", _default_db))
ODDS_DIR = os.path.join(ATP_ROOTDIR, "tennis_data", "odds")

MODEL_TYPE = os.environ.get("MODEL_TYPE", "naive_bayes")
HOLDOUT_YEAR = int(os.environ.get("HOLDOUT_YEAR", "2022"))
KELLY = float(os.environ.get("KELLY", "0.25"))
MIN_EV = float(os.environ.get("MIN_EV", "0.02"))
MIN_EDGE = float(os.environ.get("MIN_EDGE", "0.05"))
MAX_ODDS = float(os.environ.get("MAX_ODDS", "3.0"))
MARKET_ALPHA = float(os.environ.get("MARKET_ALPHA", "1.0"))
CALIBRATE = os.environ.get("CALIBRATE", "0").lower() in ("1", "true", "yes")
# Fraction of training data held out for temperature scaling (not used for model fitting)
CALIBRATE_SPLIT = float(os.environ.get("CALIBRATE_SPLIT", "0.2"))
# Hard cap on stake per bet as fraction of bankroll. None = uncapped (theoretical).
_max_kelly_env = os.environ.get("MAX_KELLY", "0.05")
MAX_KELLY: float | None = None if _max_kelly_env.lower() == "none" else float(_max_kelly_env)

# Columns dropped from features (mirrors Dataspring.process_df)
_DROP_PATTERNS = [
    "player1_score", "player2_score", "player1_rank", "player2_rank",
    "_ace", "_df", "_svpt", "_1stIn", "_1stWon", "_2ndWon",
    "_SvGms", "_bpSaved", "_bpFaced",
]
_DROP_EXACT = ["tourney_id", "month", "day", "minutes", "tourney_date",
               "game_winner", "player1_name", "player2_name",
               "player1_id", "player2_id", "tourney_name",
               "Unnamed: 0", "Unnamed: 0.1", "yday",
               # Base DB ELO is stored post-match (updated after result) — leakage.
               # Use surface-specific ELOs from add_surface_elo() which are pre-match.
               "player1_elo", "player2_elo"]


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Drop non-feature columns and return (X, y, feature_cols).

    ROW-ORDERING CONTRACT
    ---------------------
    Rows are sorted by ``tourney_date`` internally before building X.
    Row i of the returned X corresponds to the i-th row of
    ``df.sort_values("tourney_date").reset_index(drop=True)``.

    Callers that pair the returned array with other per-match data (e.g.,
    player names for odds matching) MUST use the same sorted DataFrame, not
    the original unsorted input. Failure causes a systematic row-order
    mismatch where every prediction is assigned to the wrong match.

    Correct call pattern in main()::

        test_sorted = test_raw.sort_values("tourney_date").reset_index(drop=True)
        X_test, y_test, _ = _prepare_features(test_raw)   # sorts internally
        probs = model.predict_proba(X_test)
        matched = match_odds(test_sorted, odds_df, probs)  # must use test_sorted

    LABEL ENCODING
    --------------
    ``y = game_winner - 1``: player1 wins → 0, player2 wins → 1.
    ``game_winner`` must not appear in any feature-drop list (it is extracted
    first and then dropped explicitly so the two operations cannot conflict).
    """
    df = df.copy().sort_values("tourney_date").reset_index(drop=True)
    df = df.fillna(-10)

    # Extract labels BEFORE dropping — game_winner also appears in _DROP_EXACT.
    assert "game_winner" in df.columns, "Input DataFrame must contain 'game_winner'"
    assert set(np.unique(df["game_winner"].dropna().astype(int))).issubset({1, 2}), (
        f"game_winner must contain only 1 and 2, got: {df['game_winner'].unique()}"
    )
    y = df["game_winner"].values - 1   # 0 = player1 wins, 1 = player2 wins

    # Drop by exact name
    drop_cols = [c for c in _DROP_EXACT if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Drop by pattern — exclude EMA columns (_ema_ prefix = pre-match rolling avg, not raw stats)
    for pat in _DROP_PATTERNS:
        df = df.drop(columns=[c for c in df.columns if pat in c and "_ema_" not in c], errors="ignore")

    # game_winner was already extracted; ensure it's not in the feature matrix
    df = df.drop(columns=["game_winner"], errors="ignore")

    feat_cols = df.columns.tolist()
    leakage = {"game_winner", "player1_name", "player2_name", "tourney_date"} & set(feat_cols)
    assert not leakage, f"Leakage columns found in feature matrix: {leakage}"

    X = df.values.astype(float)
    assert len(X) == len(y), f"Feature matrix ({len(X)}) and labels ({len(y)}) have different lengths"
    return X, y, feat_cols


def match_odds(test_df: pd.DataFrame, odds_df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Join predictions with odds on (year, p1_surname, p2_surname).

    ALIGNMENT CONTRACT
    ------------------
    ``test_df`` must be sorted by ``tourney_date`` with a reset integer index so
    that ``test_df.iloc[i]`` corresponds to ``probs[i]``. Pass the same frozen
    (sorted + reset_index) DataFrame that was used to compute X_test — never the
    original unsorted DataFrame. Violating this silently assigns every probability
    to the wrong match, producing random-walk predictions (~49% win rate).

    PROBABILITY CONVENTION
    ----------------------
    ``probs[:, 0]`` = P(player1 wins) following sklearn's convention (column k =
    P(class k), class 0 = game_winner 1 = player1 wins). Do NOT use ``probs[:, 1]``.

    CASE A vs CASE B
    ----------------
    Case A: ATP player1 surname = odds Winner surname. ``actual_winner = 1``.
        Filtered to ``game_winner == 1`` to discard name-collision false matches.
    Case B: ATP player1 surname = odds Loser surname (names swapped in odds file).
        ``p1_odds = PSL`` (loser's odds), ``p2_odds = PSW`` (winner's odds).
        ``actual_winner = 2``. Filtered to ``game_winner == 2``.
        p1_win_prob is NOT flipped — it stays as P(player1 wins). The backtester's
        EV formula compares p1_win_prob against p1_odds directly, which is correct.

    Name formats
    ------------
    tennis-data.co.uk: "Djokovic N." → surname = first token (str.split().str[0]).
    atp_database:      "Novak Djokovic" → surname = last token (str.split().str[-1]).
    """
    # --- Alignment guards ---
    assert probs.ndim == 2 and probs.shape[1] == 2, (
        f"probs must have shape (n, 2), got {probs.shape}"
    )
    assert probs.shape[0] == len(test_df), (
        f"Row count mismatch: probs has {probs.shape[0]} rows, test_df has {len(test_df)}. "
        "Pass the sorted test DataFrame that was used to build X_test."
    )
    if "tourney_date" in test_df.columns:
        dates = test_df["tourney_date"].values
        assert (dates[:-1] <= dates[1:]).all(), (
            "test_df must be sorted by tourney_date (ascending). "
            "Pass test_sorted = test_raw.sort_values('tourney_date').reset_index(drop=True)."
        )

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

    has_closing = {"max_p1_odds", "max_p2_odds"}.issubset(odds.columns)
    closing_cols = ["max_p1_odds", "max_p2_odds"] if has_closing else []
    slim = odds[["match_year", "w_sn", "l_sn", "p1_odds", "p2_odds"] + closing_cols]

    # Case A: our p1 = odds winner.
    # Valid only when ATP game_winner == 1 (confirms p1 actually won, not a name collision).
    ma = test.merge(
        slim.rename(columns={"w_sn": "p1_sn", "l_sn": "p2_sn"}),
        on=["match_year", "p1_sn", "p2_sn"], how="inner",
    )
    ma["actual_winner"] = 1
    ma = ma[ma["game_winner"] == 1]   # drop name-collision mismatches
    if has_closing and not ma.empty:
        ma = ma.rename(columns={"max_p1_odds": "p1_closing_odds",
                                 "max_p2_odds": "p2_closing_odds"})

    # Case B: our p1 = odds loser (names swapped).
    # p1_odds = PSL (loser's odds), p2_odds = PSW (winner's odds).
    # actual_winner = 2; valid only when ATP game_winner == 2.
    mb_rename: dict = {"w_sn": "p2_sn", "l_sn": "p1_sn",
                       "p1_odds": "p2_odds_x", "p2_odds": "p1_odds_x"}
    if has_closing:
        mb_rename["max_p1_odds"] = "p2_closing_odds"  # winner's max = p2 closing
        mb_rename["max_p2_odds"] = "p1_closing_odds"  # loser's max  = p1 closing
    mb = test.merge(
        slim.rename(columns=mb_rename),
        on=["match_year", "p1_sn", "p2_sn"], how="inner",
    )
    if not mb.empty and "p1_odds_x" in mb.columns:
        mb = mb.rename(columns={"p1_odds_x": "p1_odds", "p2_odds_x": "p2_odds"})
        mb["actual_winner"] = 2
        mb = mb[mb["game_winner"] == 2]   # drop name-collision mismatches

    merged = pd.concat([ma, mb], ignore_index=True)
    # Dedup on sorted surnames to avoid betting both orderings of the same physical match.
    if not merged.empty:
        merged["_key"] = merged.apply(
            lambda r: tuple(sorted([r["p1_sn"], r["p2_sn"]])) + (int(r["match_year"]),), axis=1
        )
        merged = merged.drop_duplicates(subset=["_key"]).drop(columns=["_key"])

    # Add date column for backtester (uses match_year as proxy)
    merged["date"] = pd.to_datetime(
        merged["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )

    match_rate = 100.0 * len(merged) / max(len(test_df), 1)
    logging.info(
        "match_odds: Case A=%d, Case B=%d, deduped=%d (%.1f%% of test set matched)",
        len(ma), len(mb), len(merged), match_rate,
    )
    if match_rate < 5.0:
        logging.warning(
            "Only %.1f%% of test rows matched odds. Check name formats and year coverage.",
            match_rate,
        )
    return merged


def main() -> None:
    from src.data.odds_loader import load_odds_dir
    from src.evaluation.backtester import BacktestConfig, run, summary
    from src.evaluation.calibration import blend_with_market, CalibratedPredictor
    from src.models.classifiers import make_classifier, CLASSIFIER_REGISTRY
    from src.models.xgboost_model import XGBoostPredictor

    print("=== Tennis Betting Backtest ===")
    print(f"Model      : {MODEL_TYPE}")
    print(f"Holdout    : {HOLDOUT_YEAR}+")
    print(f"Kelly frac : {KELLY}  |  Min EV: {MIN_EV}  |  Max Kelly: {MAX_KELLY}  |  Market alpha: {MARKET_ALPHA}")
    print(f"Calibrate  : {CALIBRATE}  |  Cal split: {CALIBRATE_SPLIT:.0%}")
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

    # 2. Prepare features — _prepare_features sorts by tourney_date internally,
    # so we must use the same sorted DataFrame when passing probs to match_odds.
    print("Building feature matrices...")
    test_sorted = test_raw.sort_values("tourney_date").reset_index(drop=True)
    X_train, y_train, feat_cols = _prepare_features(train_raw)
    X_test,  y_test,  _          = _prepare_features(test_raw)   # sorts internally
    # Alignment guard: test_sorted and X_test must have the same row count so
    # probs[i] correctly corresponds to test_sorted.iloc[i] in match_odds().
    assert len(test_sorted) == len(X_test), (
        f"Alignment failure: test_sorted has {len(test_sorted)} rows "
        f"but X_test has {len(X_test)}. Both must derive from the same test_raw."
    )
    assert set(np.unique(y_train)).issubset({0, 1}), (
        f"Unexpected label values in y_train: {np.unique(y_train)}"
    )
    print(f"  Features: {X_train.shape[1]}  |  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Normalize using ONLY training statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Carve calibration split from training data (chronologically last CALIBRATE_SPLIT fraction).
    # This must happen AFTER scaling so X_cal uses training-derived statistics.
    cal_split_idx = int(len(X_train) * (1.0 - CALIBRATE_SPLIT))
    X_fit, y_fit   = X_train[:cal_split_idx], y_train[:cal_split_idx]
    X_cal, y_cal   = X_train[cal_split_idx:], y_train[cal_split_idx:]

    # 3. Train model
    print(f"Training {MODEL_TYPE}...")
    if MODEL_TYPE == "xgboost":
        model = XGBoostPredictor()
        xgb_split = int(len(X_fit) * 0.8)
        model.fit(X_fit[:xgb_split], y_fit[:xgb_split],
                  X_val=X_fit[xgb_split:], y_val=y_fit[xgb_split:])
    elif MODEL_TYPE in CLASSIFIER_REGISTRY:
        model = make_classifier(MODEL_TYPE)
        model.fit(X_fit, y_fit)
    else:
        print(f"Unknown model. Available: {list(CLASSIFIER_REGISTRY)} + xgboost")
        return

    # 3b. Optional temperature scaling calibration — fit on held-out cal split.
    if CALIBRATE:
        cal_model = CalibratedPredictor(model)
        cal_model.calibrate(X_cal, y_cal)
        raw_probs_cal = model.predict_proba(X_cal)
        cal_probs_cal = cal_model.predict_proba(X_cal)
        T = cal_model._calibrator.temperature_
        print(f"  Temperature T = {T:.4f}  "
              f"(cal p1_win mean: {raw_probs_cal[:,0].mean():.3f} → "
              f"{cal_probs_cal[:,0].mean():.3f})")
        model = cal_model

    # Quick accuracy check
    test_preds = np.argmax(model.predict_proba(X_test), axis=1)
    accuracy = float(np.mean(test_preds == y_test)) * 100
    print(f"  Holdout accuracy: {accuracy:.1f}%")

    # 4. Match predictions with odds
    test_probs = model.predict_proba(X_test)   # (n, 2)
    print("Matching predictions to bookmaker odds...")
    matched = match_odds(test_sorted, holdout_odds, test_probs)
    if matched.empty:
        print("No matches found. Check name format in both datasets.")
        return
    print(f"  Matched {len(matched):,} rows")

    # 4b. Optional market ensemble — blend p_model with de-vigged implied probability
    if MARKET_ALPHA < 1.0:
        matched = blend_with_market(matched, alpha=MARKET_ALPHA)
        print(f"  Applied market blend: alpha={MARKET_ALPHA} "
              f"(model {MARKET_ALPHA:.0%} / market {1.0 - MARKET_ALPHA:.0%})")

    # 5. Run backtest
    print(f"\nRunning backtest on {len(matched):,} matched matches...\n")
    cfg = BacktestConfig(
        initial_bankroll=1000.0,
        kelly_fraction=KELLY,
        min_ev=MIN_EV,
        min_edge=MIN_EDGE,
        max_odds=MAX_ODDS,
        max_kelly=MAX_KELLY,
    )
    result = run(matched, cfg)
    print(summary(result))


if __name__ == "__main__":
    main()
