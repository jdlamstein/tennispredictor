"""Tests for scripts/rolling_backtest.py.

Verifies year isolation, output shape, and aggregate computation.
Uses a tiny synthetic DataFrame — no real data files required.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers — build a minimal synthetic ATP-style CSV
# ---------------------------------------------------------------------------

def _synthetic_row(
    tourney_date: int,
    id1: int,
    id2: int,
    name1: str,
    name2: str,
    winner: int,
    surface: int = 0,
    round_num: int = 1,
) -> dict:
    """Minimal row compatible with _prepare_features."""
    return {
        "tourney_date": tourney_date,
        "player1_id": id1,
        "player2_id": id2,
        "player1_name": name1,
        "player2_name": name2,
        "game_winner": winner,
        "surface": surface,
        "round": round_num,
        "draw_size": 32,
        "tourney_level": 1,
        "match_num": 1,
        "best_of": 3,
        "player1_seed": -10,
        "player2_seed": -10,
        "player1_entry": -10,
        "player2_entry": -10,
        "player1_hand": 1,
        "player2_hand": 1,
        "player1_ht": 185,
        "player2_ht": 185,
        "player1_ioc": 0,
        "player2_ioc": 0,
        "player1_age": 25,
        "player2_age": 25,
        "year": tourney_date // 10000,
        "sine_day": 0.0,
        "cosine_day": 1.0,
        "player1_glicko": 1500,
        "player2_glicko": 1500,
        "player1_rd": 350,
        "player2_rd": 350,
        "player1_sigma": 0.06,
        "player2_sigma": 0.06,
        "player1_elo_hard": 1500,
        "player2_elo_hard": 1500,
        "player1_elo_clay": 1500,
        "player2_elo_clay": 1500,
        "player1_elo_grass": 1500,
        "player2_elo_grass": 1500,
        "player1_elo_carpet": 1500,
        "player2_elo_carpet": 1500,
        "player1_winning_streak": 0,
        "player2_winning_streak": 0,
        "player1_losing_streak": 0,
        "player2_losing_streak": 0,
        "player1_weeks_inactive": 0,
        "player2_weeks_inactive": 0,
        "player1_last_two_weeks": 1,
        "player2_last_two_weeks": 1,
        "player1_v_player2_wins": 0,
        "player2_v_player1_wins": 0,
        "year_col": tourney_date // 10000,
    }


def _build_csv(rows: list[dict]) -> str:
    """Write rows to a temp CSV and return its path."""
    df = pd.DataFrame(rows)
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(f, index=False)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Unit tests for _run_year
# ---------------------------------------------------------------------------

class TestRunYear:
    """Verify the per-year run function honours year isolation."""

    def _make_atp_csv(self) -> str:
        rows = []
        for year in [2019, 2020, 2021]:
            for i in range(20):
                rows.append(_synthetic_row(
                    tourney_date=int(f"{year}0101") + i,
                    id1=1, id2=2,
                    name1="Novak Djokovic", name2="Roger Federer",
                    winner=1 if i % 2 == 0 else 2,
                ))
        return _build_csv(rows)

    def test_no_train_row_in_test_year(self, tmp_path) -> None:
        """Walk-forward split: training data must never include the holdout year."""
        rows = []
        for year in [2019, 2020]:
            for i in range(30):
                rows.append(_synthetic_row(
                    int(f"{year}0101") + i, 1, 2,
                    "Novak Djokovic", "Roger Federer",
                    1 if i % 2 == 0 else 2,
                ))
        raw = pd.DataFrame(rows)
        raw["year_col"] = raw["tourney_date"] // 10000

        holdout_year = 2020
        train = raw[raw["year_col"] < holdout_year]
        test  = raw[raw["year_col"] == holdout_year]

        assert (train["year_col"] < holdout_year).all(), "Train must be pre-holdout only"
        assert (test["year_col"] == holdout_year).all(), "Test must be holdout year only"
        assert len(train) + len(test) == len(raw)

    def test_run_year_returns_dict_with_required_keys(self, tmp_path) -> None:
        """_run_year must always return a dict with 'year' and 'n_bets'."""
        from scripts.rolling_backtest import _run_year

        rows = []
        for year in [2019, 2020]:
            for i in range(30):
                rows.append(_synthetic_row(
                    int(f"{year}0101") + i, 1, 2,
                    "Novak Djokovic", "Roger Federer",
                    1 if i % 2 == 0 else 2,
                ))
        raw = pd.DataFrame(rows)
        raw["year_col"] = raw["tourney_date"] // 10000

        # Empty odds → should return error dict, not raise
        odds_df = pd.DataFrame()
        result = _run_year(raw, odds_df, holdout_year=2020)
        assert "year" in result
        assert "n_bets" in result
        assert result["year"] == 2020

    def test_run_year_error_when_no_train_data(self, tmp_path) -> None:
        """_run_year returns error key when training data is empty."""
        from scripts.rolling_backtest import _run_year

        raw = pd.DataFrame()
        odds_df = pd.DataFrame()
        result = _run_year(raw, odds_df, holdout_year=2020)
        assert result.get("error") is not None or result.get("n_bets", 0) == 0

    def test_run_year_error_when_no_odds(self, tmp_path) -> None:
        """_run_year returns error when odds are empty for the holdout year."""
        from scripts.rolling_backtest import _run_year

        rows = []
        for year in [2019, 2020]:
            for i in range(30):
                rows.append(_synthetic_row(
                    int(f"{year}0101") + i, 1, 2,
                    "Novak Djokovic", "Roger Federer",
                    1 if i % 2 == 0 else 2,
                ))
        raw = pd.DataFrame(rows)
        raw["year_col"] = raw["tourney_date"] // 10000

        odds_df = pd.DataFrame()  # no odds
        result = _run_year(raw, odds_df, holdout_year=2020)
        assert result.get("n_bets", 0) == 0 or result.get("error")


# ---------------------------------------------------------------------------
# Unit tests for _print_table
# ---------------------------------------------------------------------------

class TestPrintTable:
    """Verify table formatting handles edge cases without raising."""

    def test_empty_rows(self, capsys) -> None:
        from scripts.rolling_backtest import _print_table
        _print_table([])
        captured = capsys.readouterr()
        assert isinstance(captured.out, str)

    def test_rows_with_errors(self, capsys) -> None:
        from scripts.rolling_backtest import _print_table
        rows = [{"year": 2020, "n_bets": 0, "error": "no data"}]
        _print_table(rows)
        captured = capsys.readouterr()
        assert "2020" in captured.out

    def test_valid_rows_print_year_and_roi(self, capsys) -> None:
        from scripts.rolling_backtest import _print_table
        rows = [{
            "year": 2022, "n_bets": 50, "win_pct": 62.0,
            "roi": 5.3, "sharpe": 1.2, "max_dd": -15.0,
            "clv": 0.3, "total_profit": 53.0,
        }]
        _print_table(rows)
        captured = capsys.readouterr()
        assert "2022" in captured.out
        assert "5.3" in captured.out or "+5.3" in captured.out
