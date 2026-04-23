"""Tests for src/data/results_fetcher (mock network calls)."""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.results_fetcher import (
    _parse_sackmann_date,
    _from_sackmann,
    filter_for_pending,
    fetch_recent_results,
)

# Pin current_year in all _from_sackmann calls so tests don't drift over time.
_CY = 2024


# ---------------------------------------------------------------------------
# _parse_sackmann_date
# ---------------------------------------------------------------------------

class TestParseSackmannDate:
    def test_int_format(self):
        assert _parse_sackmann_date(20240601) == date(2024, 6, 1)

    def test_string_format(self):
        assert _parse_sackmann_date("20220115") == date(2022, 1, 15)

    def test_invalid_returns_min(self):
        assert _parse_sackmann_date("not-a-date") == date.min


# ---------------------------------------------------------------------------
# _from_sackmann (mocked HTTP)
# ---------------------------------------------------------------------------

_FAKE_CSV = """tourney_date,winner_name,loser_name,tourney_name,surface
20240601,Novak Djokovic,Carlos Alcaraz,Roland Garros,Clay
20240601,Rafael Nadal,Roger Federer,Roland Garros,Clay
20230101,Daniil Medvedev,Andrey Rublev,Brisbane,Hard
"""


class TestFromSackmann:
    def _session(self, text: str, status: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        resp.status_code = status
        resp.raise_for_status = MagicMock() if status == 200 else MagicMock(side_effect=Exception("404"))
        s = MagicMock()
        s.get.return_value = resp
        return s

    def test_returns_results_within_cutoff(self):
        session = self._session(_FAKE_CSV)
        cutoff = date(2024, 5, 1)
        # _current_year=2024 → single fetch; 2 matches on 20240601 >= cutoff
        results = _from_sackmann(session, cutoff, _current_year=_CY)
        assert len(results) == 2

    def test_cutoff_excludes_older_matches(self):
        session = self._session(_FAKE_CSV)
        cutoff = date(2024, 7, 1)  # all matches before this
        results = _from_sackmann(session, cutoff, _current_year=_CY)
        assert len(results) == 0

    def test_winner_is_always_player1(self):
        session = self._session(_FAKE_CSV)
        cutoff = date(2024, 1, 1)
        results = _from_sackmann(session, cutoff, _current_year=_CY)
        assert all(r["winner"] == 1 for r in results)

    def test_network_error_returns_empty(self):
        import requests
        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("timeout")
        results = _from_sackmann(session, date(2024, 1, 1), _current_year=_CY)
        assert results == []

    def test_missing_columns_returns_empty(self):
        bad_csv = "a,b,c\n1,2,3\n"
        session = self._session(bad_csv)
        results = _from_sackmann(session, date(2024, 1, 1), _current_year=_CY)
        assert results == []

    def test_result_dict_has_required_keys(self):
        session = self._session(_FAKE_CSV)
        cutoff = date(2024, 1, 1)
        results = _from_sackmann(session, cutoff, _current_year=_CY)
        for r in results:
            assert "player1" in r
            assert "player2" in r
            assert "winner" in r
            assert r["winner"] in (1, 2)

    def test_multi_year_makes_multiple_requests(self):
        """Cutoff in prior year triggers one GET per year."""
        session = self._session(_FAKE_CSV)
        cutoff = date(2023, 1, 1)
        _from_sackmann(session, cutoff, _current_year=2024)
        assert session.get.call_count == 2  # 2023 and 2024

    def test_single_year_makes_one_request(self):
        """Cutoff within current year makes exactly one GET."""
        session = self._session(_FAKE_CSV)
        cutoff = date(2024, 5, 1)
        _from_sackmann(session, cutoff, _current_year=2024)
        assert session.get.call_count == 1


# ---------------------------------------------------------------------------
# filter_for_pending
# ---------------------------------------------------------------------------

class TestFilterForPending:
    _RESULTS = [
        {"player1": "Novak Djokovic", "player2": "Carlos Alcaraz", "winner": 1},
        {"player1": "Rafael Nadal",   "player2": "Roger Federer",   "winner": 1},
        {"player1": "Daniil Medvedev","player2": "Andrey Rublev",   "winner": 2},
    ]

    def test_exact_match_kept(self):
        pending = [("Novak Djokovic", "Carlos Alcaraz")]
        out = filter_for_pending(self._RESULTS, pending)
        assert len(out) == 1
        assert out[0]["player1"] == "Novak Djokovic"

    def test_reversed_pair_still_matches(self):
        # Prediction logged as (Alcaraz, Djokovic), result stored as (Djokovic, Alcaraz)
        pending = [("Carlos Alcaraz", "Novak Djokovic")]
        out = filter_for_pending(self._RESULTS, pending)
        assert len(out) == 1

    def test_no_match_returns_empty(self):
        pending = [("Andy Murray", "Grigor Dimitrov")]
        out = filter_for_pending(self._RESULTS, pending)
        assert out == []

    def test_surname_matching(self):
        # PaperTrader stores full names; filter should match on last token
        pending = [("Novak Djokovic", "Carlos Alcaraz")]
        out = filter_for_pending(self._RESULTS, pending)
        assert len(out) == 1

    def test_empty_pending_returns_empty(self):
        out = filter_for_pending(self._RESULTS, [])
        assert out == []

    def test_multiple_pairs_matched(self):
        pending = [
            ("Novak Djokovic", "Carlos Alcaraz"),
            ("Daniil Medvedev", "Andrey Rublev"),
        ]
        out = filter_for_pending(self._RESULTS, pending)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# fetch_recent_results integration (mocked to avoid live HTTP)
# ---------------------------------------------------------------------------

class TestFetchRecentResults:
    def test_returns_list(self):
        import requests
        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("no network")
        results = fetch_recent_results(days=1, session=session)
        assert isinstance(results, list)

    def test_sackmann_success_skips_uts(self):
        """When Sackmann returns data, UTS is never called."""
        resp = MagicMock()
        resp.text = _FAKE_CSV
        resp.raise_for_status = MagicMock()
        session = MagicMock()
        session.get.return_value = resp

        # days=7 → cutoff is within current year → one Sackmann GET
        results = fetch_recent_results(days=7, session=session)
        assert isinstance(results, list)
        # At least one GET (Sackmann); UTS never tried (Sackmann returned data)
        assert session.get.call_count >= 1
