"""Walk-forward rolling backtest.

Trains on all years < Y, tests on year Y, for Y in START_YEAR..END_YEAR.
Prints a per-year metrics table and aggregate statistics.

Usage
-----
    poetry run python scripts/rolling_backtest.py

Environment variables
---------------------
    ATP_DB          Path to enriched ATP database CSV
    ODDS_DIR        Directory with tennis-data.co.uk XLSX files
    MODEL_TYPE      naive_bayes | adaboost | xgboost (default: naive_bayes)
    START_YEAR      First holdout year (default: 2019)
    END_YEAR        Last holdout year inclusive (default: 2024)
    KELLY           Fractional Kelly multiplier (default: 0.25)
    MIN_EV          Minimum EV threshold (default: 0.02)
    MAX_KELLY       Hard cap on stake fraction (default: 0.05)
    MARKET_ALPHA    Model weight in blend (default: 1.0)
"""

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
START_YEAR   = int(os.environ.get("START_YEAR", "2019"))
END_YEAR     = int(os.environ.get("END_YEAR", "2024"))
KELLY        = float(os.environ.get("KELLY", "0.25"))
MIN_EV       = float(os.environ.get("MIN_EV", "0.02"))
_mk_env      = os.environ.get("MAX_KELLY", "0.05")
MAX_KELLY    = None if _mk_env.lower() == "none" else float(_mk_env)
MARKET_ALPHA = float(os.environ.get("MARKET_ALPHA", "1.0"))


def _run_year(
    raw: pd.DataFrame,
    odds_df: pd.DataFrame,
    holdout_year: int,
) -> dict:
    """Train on pre-holdout data, test on holdout_year. Return metrics dict."""
    from scripts.backtest import _prepare_features, match_odds
    from src.evaluation.backtester import BacktestConfig, run as bt_run
    from src.evaluation.calibration import blend_with_market
    from src.models.classifiers import make_classifier, CLASSIFIER_REGISTRY
    from src.models.xgboost_model import XGBoostPredictor

    if raw.empty or "year_col" not in raw.columns:
        return {"year": holdout_year, "n_bets": 0, "error": "no data"}

    train_raw = raw[raw["year_col"] < holdout_year].copy()
    test_raw  = raw[raw["year_col"] == holdout_year].copy()

    if len(train_raw) == 0 or len(test_raw) == 0:
        return {"year": holdout_year, "n_bets": 0, "error": "no data"}

    if odds_df.empty or "date_int" not in odds_df.columns:
        return {"year": holdout_year, "n_bets": 0, "error": "no odds"}

    year_odds = odds_df[odds_df["date_int"] // 10000 == holdout_year]
    if year_odds.empty:
        return {"year": holdout_year, "n_bets": 0, "error": "no odds"}

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
        return {"year": holdout_year, "n_bets": 0, "error": f"unknown model {MODEL_TYPE}"}

    probs = model.predict_proba(X_test)
    matched = match_odds(test_sorted, year_odds, probs)
    if matched.empty:
        return {"year": holdout_year, "n_bets": 0, "error": "no matches"}

    if MARKET_ALPHA < 1.0:
        matched = blend_with_market(matched, alpha=MARKET_ALPHA)

    cfg = BacktestConfig(
        initial_bankroll=1000.0,
        kelly_fraction=KELLY,
        min_ev=MIN_EV,
        max_kelly=MAX_KELLY,
    )
    result = bt_run(matched, cfg)
    m = result.metrics
    return {
        "year":        holdout_year,
        "n_bets":      int(m.get("n_bets", 0)),
        "win_pct":     m.get("win_rate", float("nan")),
        "roi":         m.get("roi", float("nan")),
        "sharpe":      m.get("sharpe", float("nan")),
        "max_dd":      m.get("max_drawdown", float("nan")),
        "clv":         m.get("clv", float("nan")),
        "total_profit": m.get("total_profit", float("nan")),
    }


def _print_table(rows: list[dict]) -> None:
    """Print results as a markdown-style table."""
    header = f"{'Year':>6} {'Bets':>6} {'Win%':>7} {'ROI%':>7} {'Sharpe':>7} {'MaxDD%':>7} {'CLV':>8} {'Profit':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    roi_vals, sharpe_vals, clv_vals = [], [], []
    profitable_years = 0

    for r in rows:
        if r.get("error"):
            print(f"  {r['year']}  — skipped ({r['error']})")
            continue
        n = r["n_bets"]
        if n == 0:
            print(f"  {r['year']}  — no bets placed")
            continue

        roi   = r["roi"]
        sh    = r["sharpe"]
        clv   = r["clv"]
        clv_s = f"{clv:+.2f}" if np.isfinite(clv) else "  n/a"

        profit = r["total_profit"]
        profit_s = f"{profit:>+.2e}" if abs(profit) > 1e6 else f"{profit:>+8.1f}"
        print(
            f"{r['year']:>6} {n:>6} {r['win_pct']:>6.1f}% {roi:>+6.1f}% "
            f"{sh:>7.2f} {-r['max_dd']:>+6.1f}% {clv_s:>8} {profit_s}"
        )
        roi_vals.append(roi)
        sharpe_vals.append(sh)
        if np.isfinite(clv):
            clv_vals.append(clv)
        if roi > 0:
            profitable_years += 1

    print(sep)

    if roi_vals:
        mean_roi    = float(np.mean(roi_vals))
        std_roi     = float(np.std(roi_vals))
        mean_sharpe = float(np.mean(sharpe_vals))
        mean_clv    = float(np.mean(clv_vals)) if clv_vals else float("nan")
        clv_s       = f"{mean_clv:+.2f}" if np.isfinite(mean_clv) else "  n/a"
        n_years     = len(roi_vals)
        print(
            f"{'Mean':>6} {'':>6} {'':>7} {mean_roi:>+6.1f}% "
            f"{mean_sharpe:>7.2f} {'':>7} {clv_s:>8}"
        )
        print(f"  Std ROI: {std_roi:+.1f}%  |  Profitable years: {profitable_years}/{n_years}"
              f"  |  Avg CLV: {clv_s}")


def main() -> None:
    from src.data.odds_loader import load_odds_dir

    print("=== Rolling Walk-Forward Backtest ===")
    print(f"Model: {MODEL_TYPE}  |  Years: {START_YEAR}–{END_YEAR}")
    print(f"Kelly: {KELLY}  |  MinEV: {MIN_EV}  |  MaxKelly: {MAX_KELLY}  |  Alpha: {MARKET_ALPHA}")
    print()

    print("Loading data...")
    raw = pd.read_csv(ATP_DB, low_memory=False)
    raw["year_col"] = raw["tourney_date"] // 10000
    odds_df = load_odds_dir(ODDS_DIR)
    print(f"  ATP rows: {len(raw):,}  |  Odds rows: {len(odds_df):,}")
    print()

    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"  Running {year}...", end="", flush=True)
        r = _run_year(raw, odds_df, year)
        rows.append(r)
        if r.get("error"):
            print(f" skipped ({r['error']})")
        else:
            print(f" {r['n_bets']} bets, ROI {r['roi']:+.1f}%")

    print()
    _print_table(rows)


if __name__ == "__main__":
    main()
