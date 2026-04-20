"""Tests for TheOddsAPIFetcher.

No real network calls — all responses are injected via a mock session.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.betting.odds_fetcher import MatchOdds, TheOddsAPIFetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    home: str = "Novak Djokovic",
    away: str = "Carlos Alcaraz",
    bookmakers: list[dict] | None = None,
    event_id: str = "abc123",
) -> dict:
    if bookmakers is None:
        bookmakers = [_make_book("pinnacle", home, 1.65, away, 2.30)]
    return {
        "id": event_id,
        "sport_key": "tennis_atp",
        "sport_title": "ATP Tennis",
        "home_team": home,
        "away_team": away,
        "bookmakers": bookmakers,
    }


def _make_book(key: str, home: str, home_price: float, away: str, away_price: float) -> dict:
    return {
        "key": key,
        "title": key.title(),
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": home, "price": home_price},
                    {"name": away, "price": away_price},
                ],
            }
        ],
    }


_ATP_SPORTS_RESP = [{"key": "tennis_atp_madrid_open", "title": "ATP Madrid Open", "active": True}]


def _mock_session(events: list[dict], status_code: int = 200) -> MagicMock:
    """Mock session that returns ATP sport list first, then events for the sport."""
    sports_resp = MagicMock()
    sports_resp.status_code = 200
    sports_resp.json.return_value = _ATP_SPORTS_RESP
    sports_resp.raise_for_status.return_value = None

    odds_resp = MagicMock()
    odds_resp.status_code = status_code
    odds_resp.json.return_value = events
    odds_resp.headers = {"x-requests-remaining": "499"}
    if status_code >= 400:
        import requests
        odds_resp.raise_for_status.side_effect = requests.HTTPError(response=odds_resp)
    else:
        odds_resp.raise_for_status.return_value = None

    session = MagicMock()
    # First call = /sports/, subsequent = /odds/
    session.get.side_effect = [sports_resp, odds_resp, odds_resp, odds_resp]
    return session


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------

class TestFromEnv:
    def test_returns_none_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        assert TheOddsAPIFetcher.from_env() is None

    def test_returns_fetcher_when_key_set(self, monkeypatch):
        monkeypatch.setenv("ODDS_API_KEY", "test-key-123")
        fetcher = TheOddsAPIFetcher.from_env()
        assert isinstance(fetcher, TheOddsAPIFetcher)


# ---------------------------------------------------------------------------
# fetch_upcoming_atp
# ---------------------------------------------------------------------------

class TestFetchUpcomingAtp:
    def test_maps_event_to_match_odds(self):
        events = [_make_event("Djokovic", "Alcaraz")]
        fetcher = TheOddsAPIFetcher(api_key="key", session=_mock_session(events))
        results = fetcher.fetch_upcoming_atp()

        assert len(results) == 1
        m = results[0]
        assert isinstance(m, MatchOdds)
        assert m.player1 == "Djokovic"
        assert m.player2 == "Alcaraz"
        assert m.p1_odds == pytest.approx(1.65)
        assert m.p2_odds == pytest.approx(2.30)
        assert m.source == "the-odds-api"
        assert m.market_id == "abc123"
        assert m.tournament == "ATP Tennis"

    def test_returns_empty_on_http_error(self):
        fetcher = TheOddsAPIFetcher(api_key="key", session=_mock_session([], status_code=401))
        assert fetcher.fetch_upcoming_atp() == []

    def test_returns_empty_on_network_error(self):
        import requests
        session = MagicMock()
        # Sports list call fails → no sport keys → empty result
        session.get.side_effect = requests.ConnectionError("timeout")
        fetcher = TheOddsAPIFetcher(api_key="key", session=session)
        assert fetcher.fetch_upcoming_atp() == []

    def test_skips_event_missing_player_names(self):
        events = [{"id": "x", "home_team": "", "away_team": "Alcaraz", "bookmakers": []}]
        fetcher = TheOddsAPIFetcher(api_key="key", session=_mock_session(events))
        assert fetcher.fetch_upcoming_atp() == []

    def test_skips_event_with_no_valid_odds(self):
        events = [_make_event(bookmakers=[])]
        fetcher = TheOddsAPIFetcher(api_key="key", session=_mock_session(events))
        assert fetcher.fetch_upcoming_atp() == []

    def test_multiple_events_all_returned(self):
        events = [
            _make_event("Djokovic", "Alcaraz", event_id="1"),
            _make_event("Sinner", "Medvedev", event_id="2",
                        bookmakers=[_make_book("bet365", "Sinner", 1.80, "Medvedev", 2.00)]),
        ]
        fetcher = TheOddsAPIFetcher(api_key="key", session=_mock_session(events))
        results = fetcher.fetch_upcoming_atp()
        assert len(results) == 2


# ---------------------------------------------------------------------------
# _pick_odds — bookmaker preference
# ---------------------------------------------------------------------------

class TestPickOdds:
    def test_prefers_pinnacle_over_bet365(self):
        home, away = "Djokovic", "Alcaraz"
        bookmakers = [
            _make_book("bet365", home, 1.60, away, 2.40),
            _make_book("pinnacle", home, 1.65, away, 2.30),
        ]
        fetcher = TheOddsAPIFetcher(api_key="key")
        p1, p2 = fetcher._pick_odds(bookmakers, home, away)
        assert p1 == pytest.approx(1.65)  # pinnacle wins
        assert p2 == pytest.approx(2.30)

    def test_falls_back_to_non_preferred_when_preferred_absent(self):
        home, away = "Djokovic", "Alcaraz"
        bookmakers = [_make_book("unibet_unknown_bookie", home, 1.70, away, 2.10)]
        fetcher = TheOddsAPIFetcher(api_key="key")
        p1, p2 = fetcher._pick_odds(bookmakers, home, away)
        assert p1 == pytest.approx(1.70)

    def test_returns_zeros_when_no_bookmakers(self):
        fetcher = TheOddsAPIFetcher(api_key="key")
        p1, p2 = fetcher._pick_odds([], "A", "B")
        assert p1 == 0.0
        assert p2 == 0.0

    def test_skips_non_h2h_markets(self):
        home, away = "Djokovic", "Alcaraz"
        book = {
            "key": "pinnacle",
            "markets": [
                {"key": "spreads", "outcomes": [{"name": home, "price": 1.90}, {"name": away, "price": 1.90}]},
            ],
        }
        fetcher = TheOddsAPIFetcher(api_key="key")
        p1, p2 = fetcher._pick_odds([book], home, away)
        assert p1 == 0.0  # spreads market ignored
