"""Paper-trading CLI.

Runs the PaperTrader daemon using a trained XGBoost model, OddsPortal odds,
and Sackmann result fetching. All state persists in a local SQLite database.

Usage
-----
    # Start the scheduler (runs until Ctrl-C)
    poetry run python scripts/paper_trade.py start

    # Run one prediction cycle now (fetch odds + log predictions)
    poetry run python scripts/paper_trade.py predict

    # Settle yesterday's bets using Sackmann results
    poetry run python scripts/paper_trade.py settle --days 1

    # Print running P&L summary
    poetry run python scripts/paper_trade.py summary

Environment variables
---------------------
    ATP_DB          Path to enriched database CSV (for feature building)
    PAPER_DB        Path to SQLite paper-trade database (default: paper_trades.db)
    PAPER_BANKROLL  Starting bankroll (default: 1000.0)
    KELLY           Fractional Kelly multiplier (default: 0.25)
    MIN_EV          Minimum EV threshold (default: 0.02)
    MAX_KELLY       Hard stake cap as fraction of bankroll (default: 0.05)
    MODEL_PATH      Path to saved XGBoost model JSON (optional — trains if absent)
    HOLDOUT_YEAR    First holdout year used to train saved model (default: 2022)
    ODDS_API_KEY    API key for the-odds-api.com (free tier, 500 credits/month)
    USE_BETFAIR     Set to 1 to use Betfair API instead of TheOddsAPI
"""

import argparse
import logging
import os
import sys

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

ATP_ROOTDIR = os.path.expanduser(os.environ.get("ATP_ROOTDIR", "~/Data/tennis"))
_default_db = os.path.join(ATP_ROOTDIR, "tennis_data", "atp_database_enriched.csv")
ATP_DB       = os.path.expanduser(os.environ.get("ATP_DB", _default_db))

PAPER_DB       = os.path.abspath(os.environ.get("PAPER_DB", "paper_trades.db"))
PAPER_BANKROLL = float(os.environ.get("PAPER_BANKROLL", "1000.0"))
KELLY          = float(os.environ.get("KELLY", "0.25"))
MIN_EV         = float(os.environ.get("MIN_EV", "0.02"))
MIN_EDGE       = float(os.environ.get("MIN_EDGE", "0.05"))
MAX_ODDS       = float(os.environ.get("MAX_ODDS", "3.0"))
MAX_KELLY      = float(os.environ.get("MAX_KELLY", "0.05"))
MODEL_PATH       = os.environ.get("MODEL_PATH", "")
MODEL_CACHE_PATH = os.environ.get("MODEL_CACHE_PATH", "paper_model.joblib")
HOLDOUT_YEAR     = int(os.environ.get("HOLDOUT_YEAR", "2022"))
USE_BETFAIR      = os.environ.get("USE_BETFAIR", "0").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Model loading / training
# ---------------------------------------------------------------------------

def _load_or_train_model(scaler_ref: list):
    """Return a fitted model + scaler. Loads joblib cache if available."""
    from src.models.xgboost_model import XGBoostPredictor
    from src.features.feature_store import FeatureStore

    # Fast path: load from joblib cache (model + scaler bundled together)
    if MODEL_CACHE_PATH and os.path.exists(MODEL_CACHE_PATH):
        try:
            bundle = joblib.load(MODEL_CACHE_PATH)
            if bundle.get("holdout_year") != HOLDOUT_YEAR:
                logger.warning(
                    "Cache holdout_year=%s ≠ current HOLDOUT_YEAR=%d — retraining.",
                    bundle.get("holdout_year"), HOLDOUT_YEAR,
                )
            else:
                scaler_ref.append(bundle["scaler"])
                logger.info("Loaded model cache from %s", MODEL_CACHE_PATH)
                return bundle["model"]
        except Exception as exc:
            logger.warning("Cache load failed (%s) — retraining.", exc)

    # Train from scratch using FeatureStore (54 features — matches inference path)
    logger.info("Training XGBoost on %s (holdout=%d)...", ATP_DB, HOLDOUT_YEAR)
    _store, X_train, y_train = FeatureStore.build_training_matrix(ATP_DB, holdout_year=HOLDOUT_YEAR)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    scaler_ref.append(scaler)

    split = int(len(X_train) * 0.8)
    model = XGBoostPredictor()
    model.fit(
        X_train[:split], y_train[:split],
        X_val=X_train[split:], y_val=y_train[split:],
    )
    logger.info("Model trained.")

    # Save cache so future calls skip re-training
    if MODEL_CACHE_PATH:
        try:
            joblib.dump({"model": model, "scaler": scaler,
                         "holdout_year": HOLDOUT_YEAR}, MODEL_CACHE_PATH)
            logger.info("Model cache saved to %s", MODEL_CACHE_PATH)
        except Exception as exc:
            logger.warning("Could not save model cache: %s", exc)

    return model


def _build_feature_builder(scaler):
    """Return a feature_builder callable for PaperTrader.

    Builds a FeatureStore from the enriched ATP database and returns a closure
    that produces model-ready feature vectors given player names + match context.
    Falls back to None (odds-only mode) if the database is unavailable.
    """
    if not os.path.exists(ATP_DB):
        logger.warning(
            "ATP_DB not found (%s) — using odds-only mode. "
            "Set ATP_DB env var to path of enriched ATP CSV.",
            ATP_DB,
        )
        return None

    try:
        from src.features.feature_store import FeatureStore
        store = FeatureStore.build(ATP_DB, holdout_year=HOLDOUT_YEAR)
    except Exception as exc:
        logger.warning("FeatureStore build failed (%s) — falling back to odds-only mode.", exc)
        return None

    def feature_builder(match) -> "np.ndarray | None":
        """Accept a MatchOdds and return scaled feature array for model inference."""
        X = store.make_features(
            p1_name=match.player1,
            p2_name=match.player2,
        )
        if X is None:
            return None
        return scaler.transform(X)

    logger.info("FeatureStore ready — model-based predictions enabled.")
    return feature_builder


# ---------------------------------------------------------------------------
# Build trader
# ---------------------------------------------------------------------------

def _make_trader(model, scaler):
    from src.betting.paper_trader import PaperTrader, PaperTraderConfig
    from src.betting.odds_fetcher import BetfairFetcher, OddsPortalFetcher, TheOddsAPIFetcher

    cfg = PaperTraderConfig(
        db_path=PAPER_DB,
        initial_bankroll=PAPER_BANKROLL,
        kelly_fraction=KELLY,
        min_ev=MIN_EV,
        min_edge=MIN_EDGE,
        max_odds=MAX_ODDS,
        max_kelly=MAX_KELLY,
        use_betfair=USE_BETFAIR,
    )

    feature_builder = _build_feature_builder(scaler)

    if USE_BETFAIR:
        try:
            fetcher = BetfairFetcher.from_env()
        except ValueError as exc:
            logger.warning("Betfair not configured (%s) — trying TheOddsAPI.", exc)
            fetcher = TheOddsAPIFetcher.from_env() or OddsPortalFetcher()
    else:
        fetcher = TheOddsAPIFetcher.from_env() or OddsPortalFetcher()

    return PaperTrader(
        cfg,
        model=model,
        feature_builder=feature_builder,
        odds_fetcher=fetcher,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_start(args) -> None:
    scaler_ref: list = []
    model = _load_or_train_model(scaler_ref)
    scaler = scaler_ref[0] if scaler_ref else None
    trader = _make_trader(model, scaler)
    print(f"Starting paper trader. DB: {PAPER_DB}  Bankroll: {PAPER_BANKROLL}")
    trader.start()  # blocks


def cmd_predict(args) -> None:
    scaler_ref: list = []
    model = _load_or_train_model(scaler_ref)
    scaler = scaler_ref[0] if scaler_ref else None
    trader = _make_trader(model, scaler)
    ids = trader.run_predict_cycle()
    print(f"Prediction cycle complete: {len(ids)} predictions logged.")


def cmd_settle(args) -> None:
    from src.data.results_fetcher import fetch_recent_results

    days = getattr(args, "days", 1)
    results = fetch_recent_results(days=days)
    if not results:
        print(f"No results found in last {days} day(s).")
        return

    scaler_ref: list = []
    model = _load_or_train_model(scaler_ref)
    scaler = scaler_ref[0] if scaler_ref else None
    trader = _make_trader(model, scaler)
    summary = trader.run_result_cycle(results)
    print(
        f"Settled: {summary['settled']}  "
        f"Wins: {summary['wins']}  "
        f"P&L: {summary['pnl']:+.2f}"
    )


def cmd_summary(args) -> None:
    scaler_ref: list = []
    model = _load_or_train_model(scaler_ref)
    scaler = scaler_ref[0] if scaler_ref else None
    trader = _make_trader(model, scaler)
    s = trader.summary()
    print("=== Paper Trading Summary ===")
    print(f"Bets placed   : {s['n_bets']}")
    print(f"Win rate      : {s['win_rate']:.1%}")
    print(f"Total staked  : {s['total_staked']:.2f}")
    print(f"Total P&L     : {s['total_pnl']:+.2f}")
    print(f"ROI           : {s['roi_pct']:.2f}%")
    print(f"Bankroll      : {s['bankroll']:.2f}")
    print(f"First bet     : {s['first_bet']}")
    print(f"Last bet      : {s['last_bet']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-trading daemon for ATP tennis predictions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("start",   help="Start APScheduler daemon (blocks until Ctrl-C)")
    sub.add_parser("predict", help="Run one prediction cycle now")

    settle_p = sub.add_parser("settle", help="Settle pending bets using recent results")
    settle_p.add_argument("--days", type=int, default=1,
                          help="Look-back window in days (default: 1)")

    sub.add_parser("summary", help="Print running P&L summary")

    args = parser.parse_args()
    dispatch = {
        "start":   cmd_start,
        "predict": cmd_predict,
        "settle":  cmd_settle,
        "summary": cmd_summary,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
