"""Parameter sensitivity analysis for the betting backtest.

Grid-searches Kelly fraction × MIN_EV × MARKET_ALPHA on the holdout period.
Prints a markdown table sorted by Sharpe ratio.

Usage
-----
    poetry run python scripts/sensitivity_analysis.py

Environment variables
---------------------
    ATP_DB        Path to enriched ATP database CSV
    ODDS_DIR      Directory with tennis-data.co.uk XLSX files
    HOLDOUT_YEAR  First holdout year (default: 2022)
    MODEL_TYPE    naive_bayes | adaboost | xgboost (default: naive_bayes)
    MAX_KELLY     Hard cap on stake fraction (default: 0.05)
"""

import itertools
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

ATP_ROOTDIR  = os.path.expanduser(os.environ.get("ATP_ROOTDIR", "~/Data/tennis"))
_default_db  = os.path.join(ATP_ROOTDIR, "tennis_data", "atp_database.csv")
ATP_DB       = os.path.expanduser(os.environ.get("ATP_DB", _default_db))
ODDS_DIR     = os.environ.get("ODDS_DIR", os.path.join(ATP_ROOTDIR, "tennis_data", "odds"))

MODEL_TYPE   = os.environ.get("MODEL_TYPE", "naive_bayes")
HOLDOUT_YEAR = int(os.environ.get("HOLDOUT_YEAR", "2022"))
_mk_env      = os.environ.get("MAX_KELLY", "0.05")
MAX_KELLY    = None if _mk_env.lower() == "none" else float(_mk_env)

GRID = {
    "kelly":        [0.10, 0.25, 0.50],
    "min_ev":       [0.01, 0.02, 0.05],
    "market_alpha": [0.0, 0.3, 0.5, 1.0],
}


def main() -> None:
    from scripts.backtest import _prepare_features, match_odds
    from src.data.odds_loader import load_odds_dir
    from src.evaluation.backtester import BacktestConfig, run as bt_run
    from src.evaluation.calibration import blend_with_market
    from src.models.classifiers import make_classifier, CLASSIFIER_REGISTRY
    from src.models.xgboost_model import XGBoostPredictor

    print("=== Sensitivity Analysis ===")
    print(f"Model: {MODEL_TYPE}  |  Holdout: {HOLDOUT_YEAR}+  |  MaxKelly: {MAX_KELLY}")
    print(f"Grid: kelly={GRID['kelly']}  min_ev={GRID['min_ev']}  alpha={GRID['market_alpha']}")
    n_combos = len(GRID["kelly"]) * len(GRID["min_ev"]) * len(GRID["market_alpha"])
    print(f"Total combinations: {n_combos}")
    print()

    # Load data once
    print("Loading data...")
    raw = pd.read_csv(ATP_DB, low_memory=False)
    raw["year_col"] = raw["tourney_date"] // 10000
    train_raw = raw[raw["year_col"] < HOLDOUT_YEAR].copy()
    test_raw  = raw[raw["year_col"] >= HOLDOUT_YEAR].copy()

    odds_df = load_odds_dir(ODDS_DIR)
    holdout_odds = odds_df[odds_df["date_int"] // 10000 >= HOLDOUT_YEAR]

    # Train model + scale once (outer loop)
    test_sorted = test_raw.sort_values("tourney_date").reset_index(drop=True)
    X_train, y_train, _ = _prepare_features(train_raw)
    X_test,  _,        _ = _prepare_features(test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    split = int(len(X_train) * 0.8)
    if MODEL_TYPE == "xgboost":
        model = XGBoostPredictor()
        model.fit(X_train[:split], y_train[:split],
                  X_val=X_train[split:], y_val=y_train[split:])
    elif MODEL_TYPE in CLASSIFIER_REGISTRY:
        model = make_classifier(MODEL_TYPE)
        model.fit(X_train, y_train)
    else:
        print(f"Unknown model type: {MODEL_TYPE}")
        return

    probs = model.predict_proba(X_test)
    base_matched = match_odds(test_sorted, holdout_odds, probs)
    if base_matched.empty:
        print("No matched rows — cannot run sensitivity analysis.")
        return
    print(f"Matched {len(base_matched)} rows. Running grid...\n")

    # Grid search — blend varies per combo, Kelly/EV vary per BacktestConfig
    results = []
    for kelly, min_ev, alpha in itertools.product(
        GRID["kelly"], GRID["min_ev"], GRID["market_alpha"]
    ):
        if alpha < 1.0:
            matched = blend_with_market(base_matched.copy(), alpha=alpha)
        else:
            matched = base_matched.copy()

        cfg = BacktestConfig(
            initial_bankroll=1000.0,
            kelly_fraction=kelly,
            min_ev=min_ev,
            max_kelly=MAX_KELLY,
        )
        result = bt_run(matched, cfg)
        m = result.metrics
        results.append({
            "kelly":   kelly,
            "min_ev":  min_ev,
            "alpha":   alpha,
            "n_bets":  int(m.get("n_bets", 0)),
            "roi":     m.get("roi", float("nan")),
            "sharpe":  m.get("sharpe", float("nan")),
            "max_dd":  m.get("max_drawdown", float("nan")),
            "clv":     m.get("clv", float("nan")),
        })

    # Sort by Sharpe descending
    results.sort(key=lambda r: r["sharpe"] if np.isfinite(r["sharpe"]) else -999, reverse=True)

    # Print table
    hdr = f"{'Kelly':>6} {'MinEV':>6} {'Alpha':>6} {'Bets':>6} {'ROI%':>7} {'Sharpe':>7} {'MaxDD%':>7} {'CLV':>7}"
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for r in results:
        clv_s = f"{r['clv']:+.2f}" if np.isfinite(r["clv"]) else "  n/a"
        roi_s = f"{r['roi']:+.1f}" if np.isfinite(r["roi"]) else "  n/a"
        sh_s  = f"{r['sharpe']:>.2f}" if np.isfinite(r["sharpe"]) else "  n/a"
        print(
            f"{r['kelly']:>6.2f} {r['min_ev']:>6.3f} {r['alpha']:>6.1f} "
            f"{r['n_bets']:>6} {roi_s:>7} {sh_s:>7} {r['max_dd']:>+6.1f}% {clv_s:>7}"
        )
    print(sep)

    best = results[0]
    print(f"\nBest config (by Sharpe):")
    print(f"  Kelly={best['kelly']}  MinEV={best['min_ev']}  Alpha={best['alpha']}")
    print(f"  Bets={best['n_bets']}  ROI={best['roi']:+.1f}%  Sharpe={best['sharpe']:.2f}")


if __name__ == "__main__":
    main()
