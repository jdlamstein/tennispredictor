"""Unit tests for paper_trader and odds_fetcher (mock-based — no live API calls)."""

from __future__ import annotations

import os
import sys
import sqlite3
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.betting.odds_fetcher import MatchOdds, OddsPortalFetcher
from src.betting.paper_trader import PaperTrader, PaperTraderConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match(p1="Novak Djokovic", p2="Roger Federer",
           p1_odds=1.50, p2_odds=2.80) -> MatchOdds:
    return MatchOdds(
        player1=p1, player2=p2,
        p1_odds=p1_odds, p2_odds=p2_odds,
        source="test", market_id="mkt-1",
        fetched_at=datetime.now(timezone.utc),
    )


def _mock_model(p1_prob: float) -> MagicMock:
    model = MagicMock()
    model.predict_proba.return_value = np.array([[p1_prob, 1.0 - p1_prob]])
    return model


def _trader(db_path: str, p1_prob: float = 0.70, min_ev: float = 0.02) -> PaperTrader:
    cfg = PaperTraderConfig(
        db_path=db_path,
        initial_bankroll=1000.0,
        kelly_fraction=0.25,
        min_ev=min_ev,
        max_kelly=0.05,
        use_betfair=False,
    )
    model = _mock_model(p1_prob)
    feature_builder = lambda m: np.zeros((1, 10))  # noqa: E731
    fetcher = MagicMock()
    fetcher.fetch_upcoming_atp.return_value = []
    return PaperTrader(cfg, model=model, feature_builder=feature_builder, odds_fetcher=fetcher)


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

class TestDBInit:
    def test_tables_created_on_init(self, tmp_path):
        db = str(tmp_path / "test.db")
        _trader(db)
        conn = sqlite3.connect(db)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert "predictions" in tables
        assert "daily_summary" in tables

    def test_init_idempotent(self, tmp_path):
        db = str(tmp_path / "test.db")
        _trader(db)
        _trader(db)  # second init must not raise or duplicate tables


# ---------------------------------------------------------------------------
# Prediction cycle
# ---------------------------------------------------------------------------

class TestPredictCycle:
    def test_positive_ev_inserts_prediction(self, tmp_path):
        """EV > min_ev → prediction logged with non-null stake."""
        db = str(tmp_path / "test.db")
        trader = _trader(db, p1_prob=0.80)  # p1_odds=1.50 → EV = 0.80*1.50-1 = 0.20 ✓
        trader._fetcher.fetch_upcoming_atp.return_value = [_match(p1_odds=1.50, p2_odds=2.80)]

        ids = trader.run_predict_cycle()

        assert len(ids) == 1
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT stake, bet_side, p1_win_prob FROM predictions").fetchone()
        conn.close()
        assert row is not None
        assert row[0] > 0          # stake placed
        assert row[1] == 1         # bet on p1
        assert abs(row[2] - 0.80) < 1e-4

    def test_below_min_ev_no_stake(self, tmp_path):
        """p1_prob=0.5 at fair odds → EV=0 → no bet."""
        db = str(tmp_path / "test.db")
        trader = _trader(db, p1_prob=0.50, min_ev=0.02)
        trader._fetcher.fetch_upcoming_atp.return_value = [_match(p1_odds=2.0, p2_odds=2.0)]

        trader.run_predict_cycle()

        conn = sqlite3.connect(db)
        row = conn.execute("SELECT stake, bet_side FROM predictions").fetchone()
        conn.close()
        assert row is not None
        assert row[0] is None   # no stake
        assert row[1] is None   # no bet side

    def test_stake_capped_at_max_kelly(self, tmp_path):
        """Kelly formula may exceed 5%; cap must apply."""
        db = str(tmp_path / "test.db")
        trader = _trader(db, p1_prob=0.99)  # extreme confidence → huge raw Kelly
        trader._fetcher.fetch_upcoming_atp.return_value = [_match(p1_odds=1.10, p2_odds=10.0)]

        trader.run_predict_cycle()

        conn = sqlite3.connect(db)
        stake = conn.execute("SELECT stake FROM predictions WHERE stake IS NOT NULL").fetchone()
        conn.close()
        assert stake is not None
        assert stake[0] <= 1000.0 * 0.05 + 1e-9  # ≤ 5% of initial bankroll

    def test_empty_markets_returns_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        trader = _trader(db)
        trader._fetcher.fetch_upcoming_atp.return_value = []
        ids = trader.run_predict_cycle()
        assert ids == []

    def test_odds_only_mode_uses_devigged_prob(self, tmp_path):
        """feature_builder=None → market de-vigged probability used."""
        cfg = PaperTraderConfig(db_path=str(tmp_path / "t.db"), initial_bankroll=500.0)
        model = _mock_model(0.70)
        fetcher = MagicMock()
        fetcher.fetch_upcoming_atp.return_value = [_match(p1_odds=1.91, p2_odds=1.91)]
        trader = PaperTrader(cfg, model=model, feature_builder=None, odds_fetcher=fetcher)
        trader.run_predict_cycle()
        conn = sqlite3.connect(cfg.db_path)
        prob = conn.execute("SELECT p1_win_prob FROM predictions").fetchone()[0]
        conn.close()
        assert abs(prob - 0.5) < 1e-4  # symmetric odds → 50%


# ---------------------------------------------------------------------------
# Result cycle
# ---------------------------------------------------------------------------

class TestResultCycle:
    def _place_and_settle(self, tmp_path, winner: int, bet_side: int = 1,
                          p1_odds: float = 1.80, p2_odds: float = 2.20) -> dict:
        db = str(tmp_path / "test.db")
        trader = _trader(db, p1_prob=0.80)
        trader._fetcher.fetch_upcoming_atp.return_value = [
            _match("Novak Djokovic", "Roger Federer", p1_odds=p1_odds, p2_odds=p2_odds)
        ]
        trader.run_predict_cycle()

        results = [{"player1": "Novak Djokovic", "player2": "Roger Federer", "winner": winner}]
        return trader.run_result_cycle(results)

    def test_win_positive_pnl(self, tmp_path):
        summary = self._place_and_settle(tmp_path, winner=1)
        assert summary["wins"] == 1
        assert summary["pnl"] > 0

    def test_loss_negative_pnl(self, tmp_path):
        # p1_prob=0.80, bet on p1 (side=1), but winner=2
        summary = self._place_and_settle(tmp_path, winner=2)
        assert summary["wins"] == 0
        assert summary["pnl"] < 0

    def test_settle_updates_db(self, tmp_path):
        db = str(tmp_path / "test.db")
        trader = _trader(db, p1_prob=0.80)
        trader._fetcher.fetch_upcoming_atp.return_value = [
            _match("Novak Djokovic", "Roger Federer", p1_odds=1.80, p2_odds=2.20)
        ]
        trader.run_predict_cycle()
        trader.run_result_cycle([
            {"player1": "Novak Djokovic", "player2": "Roger Federer", "winner": 1}
        ])
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT result, pnl, bankroll_after FROM predictions").fetchone()
        conn.close()
        assert row[0] == 1          # result recorded
        assert row[1] is not None   # pnl set
        assert row[2] is not None   # bankroll_after set

    def test_no_results_returns_zero_settled(self, tmp_path):
        db = str(tmp_path / "test.db")
        trader = _trader(db)
        out = trader.run_result_cycle(results=None)
        assert out["settled"] == 0

    def test_summary_accumulates(self, tmp_path):
        db = str(tmp_path / "test.db")
        trader = _trader(db, p1_prob=0.80)
        for i in range(3):
            trader._fetcher.fetch_upcoming_atp.return_value = [
                _match(f"Player{i} A", f"Opponent{i} B", p1_odds=1.80, p2_odds=2.20)
            ]
            trader.run_predict_cycle()
            trader.run_result_cycle([
                {"player1": f"Player{i} A", "player2": f"Opponent{i} B", "winner": 1}
            ])
        s = trader.summary()
        assert s["n_bets"] == 3
        assert s["n_wins"] == 3


# ---------------------------------------------------------------------------
# OddsPortal scraper (mock HTML)
# ---------------------------------------------------------------------------

class TestOddsPortalFetcher:
    _FAKE_HTML = """
    <html><head></head><body>
    <script id="__NEXT_DATA__" type="application/json">
    {
        "props": {
            "pageProps": {
                "initialData": {
                    "tournamentEvents": {
                        "events": [
                            {
                                "id": "evt-1",
                                "home-name": "Djokovic N.",
                                "away-name": "Federer R.",
                                "tournament-name": "Wimbledon",
                                "odds": {"b365": [1.45, 2.90]}
                            },
                            {
                                "id": "evt-2",
                                "home-name": "Alcaraz C.",
                                "away-name": "Sinner J.",
                                "tournament-name": "Roland Garros",
                                "odds": {"b365": [1.60, 2.40]}
                            }
                        ]
                    }
                }
            }
        }
    }
    </script>
    </body></html>
    """

    def test_parse_returns_match_odds(self):
        fetcher = OddsPortalFetcher()
        results = fetcher._parse_atp_page(self._FAKE_HTML)
        assert len(results) == 2
        assert results[0].player1 == "Djokovic N."
        assert results[0].player2 == "Federer R."
        assert abs(results[0].p1_odds - 1.45) < 1e-6
        assert abs(results[0].p2_odds - 2.90) < 1e-6

    def test_parse_empty_html_returns_empty(self):
        fetcher = OddsPortalFetcher()
        assert fetcher._parse_atp_page("<html></html>") == []

    def test_network_error_returns_empty(self):
        import requests
        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("timeout")
        fetcher = OddsPortalFetcher(session=session)
        assert fetcher.fetch_upcoming_atp() == []

    def test_source_label_is_oddsportal(self):
        fetcher = OddsPortalFetcher()
        results = fetcher._parse_atp_page(self._FAKE_HTML)
        assert all(r.source == "oddsportal" for r in results)

    def test_skips_invalid_odds(self):
        html = self._FAKE_HTML.replace('"b365": [1.45, 2.90]', '"b365": [0.0, 2.90]')
        fetcher = OddsPortalFetcher()
        results = fetcher._parse_atp_page(html)
        # Only the second event (valid odds) should remain
        assert len(results) == 1
        assert results[0].player1 == "Alcaraz C."
