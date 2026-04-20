"""Equity-curve and drawdown visualisation for backtest results.

Runs the backtest (or loads cached results) and saves two figures:
  figures/equity_curve_{MODEL_TYPE}_{HOLDOUT_YEAR}.png  — two-panel equity+drawdown
  figures/rolling_roi.png                               — per-year ROI bar chart
                                                          (requires rolling_backtest results)

Usage
-----
    # Single-year equity curve
    poetry run python scripts/plot_backtest.py

    # Rolling bar chart (runs rolling backtest first)
    poetry run python scripts/plot_backtest.py --rolling

Environment variables
---------------------
    Same as backtest.py (ATP_DB, ODDS_DIR, MODEL_TYPE, HOLDOUT_YEAR, KELLY, MIN_EV,
    MAX_KELLY, MARKET_ALPHA)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", "figures"))


def _run_single_backtest() -> tuple[pd.DataFrame, dict]:
    """Run the standard backtest and return (bets_df, metrics)."""
    from scripts.backtest import (
        ATP_DB, ODDS_DIR, MODEL_TYPE, HOLDOUT_YEAR, KELLY, MIN_EV,
        MAX_KELLY, MARKET_ALPHA, _prepare_features, match_odds,
    )
    from src.data.odds_loader import load_odds_dir
    from src.evaluation.backtester import BacktestConfig, run as bt_run
    from src.evaluation.calibration import blend_with_market
    from src.models.classifiers import make_classifier, CLASSIFIER_REGISTRY
    from src.models.xgboost_model import XGBoostPredictor
    from sklearn.preprocessing import StandardScaler

    raw = pd.read_csv(ATP_DB, low_memory=False)
    raw["year_col"] = raw["tourney_date"] // 10000
    train_raw = raw[raw["year_col"] < HOLDOUT_YEAR].copy()
    test_raw  = raw[raw["year_col"] >= HOLDOUT_YEAR].copy()

    odds_df = load_odds_dir(ODDS_DIR)
    holdout_odds = odds_df[odds_df["date_int"] // 10000 >= HOLDOUT_YEAR]

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
        raise ValueError(f"Unknown model: {MODEL_TYPE}")

    probs = model.predict_proba(X_test)
    matched = match_odds(test_sorted, holdout_odds, probs)
    if matched.empty:
        raise RuntimeError("No matched rows. Check ATP_DB and ODDS_DIR.")

    if MARKET_ALPHA < 1.0:
        matched = blend_with_market(matched, alpha=MARKET_ALPHA)

    cfg = BacktestConfig(
        initial_bankroll=1000.0, kelly_fraction=KELLY,
        min_ev=MIN_EV, max_kelly=MAX_KELLY,
    )
    result = bt_run(matched, cfg)
    return result.bets, result.metrics


def plot_equity_curve(bets: pd.DataFrame, metrics: dict, out_path: Path) -> None:
    """Save two-panel equity curve + drawdown plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    bankroll = bets["bankroll_after"].values
    dates    = pd.to_datetime(bets["date"], errors="coerce")
    initial  = 1000.0

    # Drawdown from rolling peak
    peak = np.maximum.accumulate(bankroll)
    drawdown_pct = 100.0 * (bankroll - peak) / peak

    roi   = metrics.get("roi", float("nan"))
    sharpe = metrics.get("sharpe", float("nan"))
    max_dd = metrics.get("max_drawdown", float("nan"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"Backtest — ROI {roi:+.1f}%  Sharpe {sharpe:.2f}  MaxDD {max_dd:.1f}%",
        fontsize=13, fontweight="bold",
    )

    # Top: equity curve
    ax1.plot(dates, bankroll, linewidth=1.5, color="steelblue", label="Bankroll")
    ax1.axhline(initial, linestyle="--", color="grey", linewidth=0.8, label="Start")
    ax1.set_ylabel("Bankroll (units)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.grid(alpha=0.3)

    # Bottom: drawdown
    ax2.fill_between(dates, drawdown_pct, 0, color="crimson", alpha=0.4, label="Drawdown")
    ax2.axhline(-30, linestyle="--", color="darkred", linewidth=0.8, label="−30% danger")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_rolling_roi(rows: list[dict], out_path: Path) -> None:
    """Save per-year ROI bar chart from rolling backtest rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = [r["year"] for r in rows if not r.get("error") and r.get("n_bets", 0) > 0]
    rois  = [r["roi"] for r in rows if not r.get("error") and r.get("n_bets", 0) > 0]
    if not years:
        print("No data to plot for rolling ROI.")
        return

    colors = ["steelblue" if r >= 0 else "crimson" for r in rois]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(years, rois, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Holdout Year")
    ax.set_ylabel("ROI %")
    ax.set_title("Walk-Forward Backtest — Annual ROI")
    ax.set_xticks(years)

    for bar, roi in zip(bars, rois):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{roi:+.1f}%", ha="center", va="bottom", fontsize=9)

    mean_roi = float(np.mean(rois))
    ax.axhline(mean_roi, linestyle="--", color="grey", linewidth=0.8,
               label=f"Mean {mean_roi:+.1f}%")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling", action="store_true",
                        help="Also run rolling backtest and save bar chart")
    args = parser.parse_args()

    from scripts.backtest import MODEL_TYPE, HOLDOUT_YEAR

    # Single-year equity curve
    print("Running backtest...")
    bets, metrics = _run_single_backtest()
    out = FIGURES_DIR / f"equity_curve_{MODEL_TYPE}_{HOLDOUT_YEAR}.png"
    plot_equity_curve(bets, metrics, out)

    if args.rolling:
        from scripts.rolling_backtest import (
            raw, odds_df, START_YEAR, END_YEAR,
        )
        from scripts.rolling_backtest import _run_year
        print("Running rolling backtest...")
        rows = []
        for year in range(START_YEAR, END_YEAR + 1):
            r = _run_year(raw, odds_df, year)
            rows.append(r)
        plot_rolling_roi(rows, FIGURES_DIR / "rolling_roi.png")


if __name__ == "__main__":
    main()
