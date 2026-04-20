"""Paper-trading daemon for ATP tennis predictions.

Runs on a schedule (APScheduler), logs predictions + EV before match starts,
then records P&L after result is known. All state persists in a SQLite database.

Workflow (per scheduled cycle)
-------------------------------
1. Fetch upcoming ATP matches from odds source (Betfair or OddsPortal).
2. Build feature vectors for each match (requires a feature-lookup table).
3. Run model inference → p1_win_prob per match.
4. Compute EV and fractional-Kelly stake (paper money only).
5. Persist predictions to SQLite ``predictions`` table BEFORE match.
6. On result cycle: fetch completed matches, update ``predictions.result``,
   compute P&L, append to ``results`` table.

Usage
-----
    from src.betting.paper_trader import PaperTrader, PaperTraderConfig

    cfg = PaperTraderConfig(db_path="paper_trades.db", initial_bankroll=1000.0)
    trader = PaperTrader(cfg, model=my_model, odds_fetcher=my_fetcher)
    trader.start()   # starts APScheduler; blocks until Ctrl-C
    trader.stop()

Tables
------
predictions
    id, fetched_at, tournament, player1, player2, p1_win_prob,
    p1_odds, p2_odds, ev, stake, bet_side, result (nullable), pnl (nullable),
    bankroll_before, bankroll_after (nullable)

daily_summary
    date, n_bets, n_wins, total_staked, total_pnl, bankroll_eod, roi_pct
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Iterator, Optional

import numpy as np

from src.betting.kelly import compute_kelly_stake, expected_value
from src.betting.odds_fetcher import BetfairFetcher, MatchOdds, OddsPortalFetcher
from src.models.base_predictor import BasePredictor

logger = logging.getLogger(__name__)

_CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at      TEXT    NOT NULL,
    tournament      TEXT    NOT NULL DEFAULT '',
    player1         TEXT    NOT NULL,
    player2         TEXT    NOT NULL,
    p1_win_prob     REAL    NOT NULL,
    p1_odds         REAL    NOT NULL,
    p2_odds         REAL    NOT NULL,
    ev              REAL    NOT NULL,
    bet_side        INTEGER,          -- 1 or 2; NULL = no bet (EV below threshold)
    stake           REAL,             -- NULL = no bet
    bankroll_before REAL    NOT NULL,
    result          INTEGER,          -- 1 or 2 after match; NULL = pending
    pnl             REAL,             -- NULL until result known
    bankroll_after  REAL              -- NULL until result known
);
"""

_CREATE_DAILY_SUMMARY = """
CREATE TABLE IF NOT EXISTS daily_summary (
    date            TEXT    PRIMARY KEY,
    n_bets          INTEGER NOT NULL DEFAULT 0,
    n_wins          INTEGER NOT NULL DEFAULT 0,
    total_staked    REAL    NOT NULL DEFAULT 0.0,
    total_pnl       REAL    NOT NULL DEFAULT 0.0,
    bankroll_eod    REAL,
    roi_pct         REAL
);
"""


@dataclass
class PaperTraderConfig:
    """Configuration for the paper-trading daemon.

    Attributes
    ----------
    db_path : str
        Path to SQLite database file (created if absent).
    initial_bankroll : float
        Starting paper bankroll in any currency unit.
    kelly_fraction : float
        Fractional Kelly multiplier (default 0.25 = quarter-Kelly).
    min_ev : float
        Minimum EV threshold to place a bet (default 0.02 = 2%).
    max_kelly : float
        Hard cap on stake as fraction of bankroll (default 0.05 = 5%).
    predict_cron : str
        Cron expression for the prediction cycle (default: 6 AM UTC daily).
    result_cron : str
        Cron expression for the result-collection cycle (default: 11 PM UTC daily).
    use_betfair : bool
        Use Betfair as primary odds source. Falls back to OddsPortal if False.
    """

    db_path: str = "paper_trades.db"
    initial_bankroll: float = 1_000.0
    kelly_fraction: float = 0.25
    min_ev: float = 0.02
    max_kelly: float = 0.05
    predict_cron: str = "0 6 * * *"    # 06:00 UTC daily
    result_cron: str  = "0 23 * * *"   # 23:00 UTC daily
    use_betfair: bool = False           # False = OddsPortal (no account needed)


class PaperTrader:
    """Manages paper-trading lifecycle: predict → log → settle → report.

    Parameters
    ----------
    config : PaperTraderConfig
    model : BasePredictor
        Fitted model that accepts feature arrays and returns predict_proba output.
    feature_builder : callable | None
        ``feature_builder(match_odds: MatchOdds) -> np.ndarray | None``
        Returns a (1, n_features) array or None if features unavailable.
        None = log prediction with p1_win_prob=NaN (odds-only mode).
    odds_fetcher : BetfairFetcher | OddsPortalFetcher | None
        Pre-constructed fetcher. Built from env vars if None.
    """

    def __init__(
        self,
        config: PaperTraderConfig,
        model: BasePredictor,
        feature_builder: Optional[object] = None,
        odds_fetcher: Optional[object] = None,
    ) -> None:
        self._cfg = config
        self._model = model
        self._feature_builder = feature_builder
        self._fetcher = odds_fetcher or self._build_fetcher()
        self._bankroll = config.initial_bankroll
        self._scheduler: object | None = None

        self._init_db()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the APScheduler daemon. Blocks until stop() or Ctrl-C."""
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError as exc:
            raise ImportError("apscheduler not installed. Run: poetry add apscheduler") from exc

        scheduler = BlockingScheduler(timezone="UTC")
        predict_parts = self._cfg.predict_cron.split()
        result_parts  = self._cfg.result_cron.split()

        scheduler.add_job(
            self.run_predict_cycle,
            CronTrigger(
                minute=predict_parts[0], hour=predict_parts[1],
                day=predict_parts[2],   month=predict_parts[3],
                day_of_week=predict_parts[4],
            ),
            id="predict",
            name="Predict upcoming matches",
        )
        scheduler.add_job(
            self.run_result_cycle,
            CronTrigger(
                minute=result_parts[0], hour=result_parts[1],
                day=result_parts[2],   month=result_parts[3],
                day_of_week=result_parts[4],
            ),
            id="results",
            name="Settle completed matches",
        )

        self._scheduler = scheduler
        logger.info(
            "PaperTrader started. Predict: %s  Results: %s",
            self._cfg.predict_cron, self._cfg.result_cron,
        )
        scheduler.start()

    def stop(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)  # type: ignore[union-attr]
            self._scheduler = None
        logger.info("PaperTrader stopped.")

    # ------------------------------------------------------------------
    # Core cycles
    # ------------------------------------------------------------------

    def run_predict_cycle(self) -> list[int]:
        """Fetch odds, build predictions, persist to DB. Returns inserted IDs."""
        logger.info("Prediction cycle started.")
        markets = self._fetch_odds()
        if not markets:
            logger.info("No markets fetched — skipping cycle.")
            return []

        inserted: list[int] = []
        for m in markets:
            row_id = self._process_market(m)
            if row_id is not None:
                inserted.append(row_id)

        logger.info("Prediction cycle complete: %d predictions logged.", len(inserted))
        return inserted

    def run_result_cycle(self, results: Optional[list[dict]] = None) -> dict:
        """Settle pending predictions.

        Parameters
        ----------
        results : list[dict] | None
            Each dict must have keys: ``player1``, ``player2``, ``winner`` (1 or 2).
            Pass None in production — in that case result fetching must be
            implemented via a live data source (e.g. tennis-abstract scraper).

        Returns
        -------
        dict
            Summary: ``{"settled": n, "wins": n, "pnl": float}``.
        """
        if results is None:
            logger.warning(
                "run_result_cycle called without results — "
                "implement a live result source and pass results explicitly."
            )
            return {"settled": 0, "wins": 0, "pnl": 0.0}

        settled = wins = 0
        total_pnl = 0.0

        with self._db() as conn:
            pending = conn.execute(
                "SELECT id, player1, player2, bet_side, stake, bankroll_before "
                "FROM predictions WHERE result IS NULL AND stake IS NOT NULL"
            ).fetchall()

        for pred_id, p1, p2, bet_side, stake, bankroll_before in pending:
            result = self._find_result(results, p1, p2)
            if result is None:
                continue

            won = result == bet_side
            p1_odds, p2_odds = self._get_odds_from_db(pred_id)
            odds_bet = p1_odds if bet_side == 1 else p2_odds
            pnl = stake * (odds_bet - 1) if won else -stake
            bankroll_after = bankroll_before + pnl

            with self._db() as conn:
                conn.execute(
                    "UPDATE predictions SET result=?, pnl=?, bankroll_after=? WHERE id=?",
                    (result, pnl, bankroll_after, pred_id),
                )

            self._bankroll = bankroll_after
            settled += 1
            if won:
                wins += 1
            total_pnl += pnl

        self._write_daily_summary(date.today(), settled, wins, total_pnl)
        logger.info(
            "Result cycle: settled=%d wins=%d pnl=%.2f", settled, wins, total_pnl
        )
        return {"settled": settled, "wins": wins, "pnl": total_pnl}

    # ------------------------------------------------------------------
    # Per-market logic
    # ------------------------------------------------------------------

    def _process_market(self, m: MatchOdds) -> Optional[int]:
        """Compute EV + stake for one market and persist."""
        p1_win_prob = self._get_model_prob(m)
        if p1_win_prob is None:
            return None

        p2_win_prob = 1.0 - p1_win_prob
        ev_p1 = expected_value(p1_win_prob, m.p1_odds)
        ev_p2 = expected_value(p2_win_prob, m.p2_odds)

        bet_side: Optional[int] = None
        stake: Optional[float] = None
        ev: float = max(ev_p1, ev_p2)

        if ev_p1 >= ev_p2 and ev_p1 > self._cfg.min_ev:
            bet_side = 1
            stake = min(
                compute_kelly_stake(p1_win_prob, m.p1_odds, self._bankroll, self._cfg.kelly_fraction),
                self._bankroll * self._cfg.max_kelly,
            )
        elif ev_p2 > ev_p1 and ev_p2 > self._cfg.min_ev:
            bet_side = 2
            stake = min(
                compute_kelly_stake(p2_win_prob, m.p2_odds, self._bankroll, self._cfg.kelly_fraction),
                self._bankroll * self._cfg.max_kelly,
            )

        if stake is not None and stake <= 0:
            bet_side = None
            stake = None

        with self._db() as conn:
            cursor = conn.execute(
                """INSERT INTO predictions
                   (fetched_at, tournament, player1, player2, p1_win_prob,
                    p1_odds, p2_odds, ev, bet_side, stake, bankroll_before)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    m.fetched_at.isoformat(),
                    m.tournament,
                    m.player1,
                    m.player2,
                    p1_win_prob,
                    m.p1_odds,
                    m.p2_odds,
                    ev,
                    bet_side,
                    stake,
                    self._bankroll,
                ),
            )
            return cursor.lastrowid

    def _get_model_prob(self, m: MatchOdds) -> Optional[float]:
        """Return P(player1 wins) from model, or None if features unavailable."""
        if self._feature_builder is None:
            # Odds-only mode: use de-vigged market probability as proxy
            imp1 = 1.0 / m.p1_odds
            imp2 = 1.0 / m.p2_odds
            return float(np.clip(imp1 / (imp1 + imp2), 1e-6, 1.0 - 1e-6))

        try:
            X = self._feature_builder(m)  # type: ignore[call-arg]
        except Exception as exc:
            logger.warning("feature_builder failed for %s vs %s: %s", m.player1, m.player2, exc)
            return None

        if X is None:
            return None

        probs = self._model.predict_proba(X)  # shape (1, 2)
        return float(np.clip(probs[0, 0], 1e-6, 1.0 - 1e-6))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_odds(self) -> list[MatchOdds]:
        try:
            if isinstance(self._fetcher, BetfairFetcher):
                return self._fetcher.fetch_tennis_markets()
            return self._fetcher.fetch_upcoming_atp()  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("Odds fetch error: %s", exc)
            return []

    def _build_fetcher(self) -> object:
        if self._cfg.use_betfair:
            try:
                return BetfairFetcher.from_env()
            except ValueError as exc:
                logger.warning("Betfair credentials missing (%s); falling back to OddsPortal.", exc)
        return OddsPortalFetcher()

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._cfg.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._db() as conn:
            conn.execute(_CREATE_PREDICTIONS)
            conn.execute(_CREATE_DAILY_SUMMARY)

    def _find_result(
        self, results: list[dict], p1: str, p2: str
    ) -> Optional[int]:
        """Match a result dict to this player pair. Returns 1, 2, or None."""
        p1_sn = p1.split()[-1].lower()
        p2_sn = p2.split()[-1].lower()
        for r in results:
            r1 = r.get("player1", "").split()[-1].lower()
            r2 = r.get("player2", "").split()[-1].lower()
            if r1 == p1_sn and r2 == p2_sn:
                return int(r["winner"])
            # Check swapped names
            if r1 == p2_sn and r2 == p1_sn:
                return 2 if int(r["winner"]) == 1 else 1
        return None

    def _get_odds_from_db(self, pred_id: int) -> tuple[float, float]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT p1_odds, p2_odds FROM predictions WHERE id=?", (pred_id,)
            ).fetchone()
        return (row[0], row[1]) if row else (1.0, 1.0)

    def _write_daily_summary(
        self, day: date, n_bets: int, n_wins: int, total_pnl: float
    ) -> None:
        with self._db() as conn:
            existing = conn.execute(
                "SELECT n_bets, n_wins, total_staked, total_pnl FROM daily_summary WHERE date=?",
                (day.isoformat(),),
            ).fetchone()

            if existing:
                nb = existing[0] + n_bets
                nw = existing[1] + n_wins
                ts = existing[2]
                tp = existing[3] + total_pnl
            else:
                nb, nw, ts, tp = n_bets, n_wins, 0.0, total_pnl

            roi = (tp / ts * 100.0) if ts > 0 else None
            conn.execute(
                """INSERT OR REPLACE INTO daily_summary
                   (date, n_bets, n_wins, total_staked, total_pnl, bankroll_eod, roi_pct)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (day.isoformat(), nb, nw, ts, tp, self._bankroll, roi),
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return aggregate performance metrics from the SQLite database."""
        with self._db() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE stake IS NOT NULL)     AS n_bets,
                    COUNT(*) FILTER (WHERE result = bet_side)     AS n_wins,
                    SUM(stake)  FILTER (WHERE stake IS NOT NULL)  AS total_staked,
                    SUM(pnl)    FILTER (WHERE pnl IS NOT NULL)    AS total_pnl,
                    MIN(fetched_at), MAX(fetched_at)
                FROM predictions
            """).fetchone()

        n_bets, n_wins, staked, pnl, first, last = row
        n_bets  = n_bets  or 0
        n_wins  = n_wins  or 0
        staked  = staked  or 0.0
        pnl     = pnl     or 0.0

        return {
            "n_bets":      n_bets,
            "n_wins":      n_wins,
            "win_rate":    n_wins / n_bets if n_bets else 0.0,
            "total_staked": staked,
            "total_pnl":   pnl,
            "roi_pct":     pnl / staked * 100.0 if staked else 0.0,
            "bankroll":    self._bankroll,
            "first_bet":   first,
            "last_bet":    last,
        }
