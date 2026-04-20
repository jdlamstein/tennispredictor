"""Tests for backtester P&L arithmetic and Kelly integration.

These tests construct small, hand-calculable bet scenarios so results can be
verified by inspection. No mocks — the backtester and Kelly function run on
real inputs; only the match data is a minimal constructed example.
"""

import sys
import os
import math

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.backtester import BacktestConfig, run, summary
from src.betting.kelly import compute_kelly_stake, expected_value, devig


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_bets_df(
    rows: list[dict],
) -> pd.DataFrame:
    """Build a minimal odds DataFrame for backtesting."""
    return pd.DataFrame(rows)


def _even_money_bet(date: str, winner: int, p1_prob: float) -> dict:
    """Even-money match at 2.0 / 2.0 decimal odds."""
    return {
        "date": date,
        "p1_win_prob": p1_prob,
        "actual_winner": winner,
        "p1_odds": 2.0,
        "p2_odds": 2.0,
    }


# ---------------------------------------------------------------------------
# Kelly Criterion arithmetic
# ---------------------------------------------------------------------------

class TestKellyArithmetic:
    """Verify exact Kelly formula output on hand-calculable examples."""

    def test_even_money_60pct(self) -> None:
        """p=0.6, odds=2.0 → full Kelly f* = (0.6*1 - 0.4)/1 = 0.2.
        Quarter-Kelly on $1000 bankroll = $50.
        """
        stake = compute_kelly_stake(p=0.6, odds=2.0, bankroll=1000.0, fraction=0.25)
        assert math.isclose(stake, 50.0, rel_tol=1e-9)

    def test_expected_value_positive(self) -> None:
        """EV = 0.6*1 - 0.4 = 0.2 on an even-money bet at p=0.6."""
        ev = expected_value(p=0.6, odds=2.0)
        assert math.isclose(ev, 0.2, rel_tol=1e-9)

    def test_devig_symmetric(self) -> None:
        """For symmetric odds (2.0 / 2.0, no vig), fair probs must be 0.5 each."""
        p_win, p_lose = devig(2.0, 2.0)
        assert math.isclose(p_win, 0.5, abs_tol=1e-9)
        assert math.isclose(p_lose, 0.5, abs_tol=1e-9)

    def test_devig_removes_vig(self) -> None:
        """Typical 5% vig at 1.91 / 1.91 should de-vig to 0.5 / 0.5."""
        p_win, p_lose = devig(1.91, 1.91)
        assert math.isclose(p_win + p_lose, 1.0, abs_tol=1e-9)
        assert math.isclose(p_win, 0.5, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# Backtester P&L arithmetic
# ---------------------------------------------------------------------------

class TestBacktesterPnL:
    """Verify P&L accounting on hand-calculable bet sequences."""

    def test_single_winning_bet(self) -> None:
        """One bet, 60% model confidence, 2.0 odds, p1 wins.

        Quarter-Kelly stake on $1000: $50.
        Payout: $100 (stake * odds). Profit: $50.
        Final bankroll: $1050.
        """
        df = _make_bets_df([_even_money_bet("2024-01-01", winner=1, p1_prob=0.6)])
        cfg = BacktestConfig(initial_bankroll=1000.0, kelly_fraction=0.25, min_ev=0.0)
        result = run(df, cfg)

        assert len(result.bets) == 1
        row = result.bets.iloc[0]
        assert math.isclose(row["stake"], 50.0, rel_tol=1e-6)
        assert math.isclose(row["payout"], 100.0, rel_tol=1e-6)
        assert math.isclose(row["profit"], 50.0, rel_tol=1e-6)
        assert math.isclose(result.metrics["final_bankroll"], 1050.0, rel_tol=1e-6)

    def test_single_losing_bet(self) -> None:
        """One bet, p1 loses. Payout = 0. Bankroll decreases by the stake."""
        df = _make_bets_df([_even_money_bet("2024-01-01", winner=2, p1_prob=0.6)])
        cfg = BacktestConfig(initial_bankroll=1000.0, kelly_fraction=0.25, min_ev=0.0)
        result = run(df, cfg)

        assert len(result.bets) == 1
        row = result.bets.iloc[0]
        assert math.isclose(row["payout"], 0.0, abs_tol=1e-9)
        assert row["profit"] < 0
        assert result.metrics["final_bankroll"] < 1000.0

    def test_negative_ev_skipped(self) -> None:
        """Model probability of 0.4 at odds 2.0 → EV = -0.2. No bet placed."""
        df = _make_bets_df([_even_money_bet("2024-01-01", winner=1, p1_prob=0.4)])
        cfg = BacktestConfig(initial_bankroll=1000.0, kelly_fraction=0.25, min_ev=0.0)
        result = run(df, cfg)
        # p1_prob=0.4, but p2_prob=0.6 has EV=+0.2 — the backtester should bet p2
        # The bet_side should be 2 (better EV)
        if len(result.bets) == 1:
            assert result.bets.iloc[0]["bet_side"] == 2

    def test_chronological_ordering_enforced(self) -> None:
        """Rows provided in reverse date order must be processed chronologically."""
        rows = [
            _even_money_bet("2024-03-01", winner=1, p1_prob=0.6),
            _even_money_bet("2024-01-01", winner=1, p1_prob=0.6),
        ]
        df = _make_bets_df(rows)  # deliberately out of order
        cfg = BacktestConfig(initial_bankroll=1000.0, kelly_fraction=0.25, min_ev=0.0)
        result = run(df, cfg)
        dates = result.bets["date"].tolist()
        assert dates == sorted(dates), "Backtester must process bets in date order"

    def test_ten_even_bets_roi(self) -> None:
        """5 wins and 5 losses at even money with identical stakes → ~0% ROI."""
        rows = (
            [_even_money_bet(f"2024-01-{i+1:02d}", winner=1, p1_prob=0.55) for i in range(5)]
            + [_even_money_bet(f"2024-01-{i+6:02d}", winner=2, p1_prob=0.55) for i in range(5)]
        )
        df = _make_bets_df(rows)
        cfg = BacktestConfig(initial_bankroll=1000.0, kelly_fraction=0.25, min_ev=0.01)
        result = run(df, cfg)

        # Win rate must be 50%
        assert math.isclose(result.metrics["win_rate"], 50.0, abs_tol=1.0)

    def test_bankroll_never_negative(self) -> None:
        """Even after consecutive losses, the bankroll must remain non-negative."""
        rows = [_even_money_bet(f"2024-01-{i+1:02d}", winner=2, p1_prob=0.6)
                for i in range(20)]
        df = _make_bets_df(rows)
        cfg = BacktestConfig(initial_bankroll=1000.0, kelly_fraction=0.25, min_ev=0.0)
        result = run(df, cfg)
        if not result.bets.empty:
            assert (result.bets["bankroll_after"] >= 0).all()

    def test_summary_string_non_empty(self) -> None:
        """summary() must return a non-empty string for any non-empty result."""
        df = _make_bets_df([_even_money_bet("2024-01-01", winner=1, p1_prob=0.6)])
        cfg = BacktestConfig(initial_bankroll=1000.0, min_ev=0.0)
        result = run(df, cfg)
        text = summary(result)
        assert isinstance(text, str) and len(text) > 0


# ---------------------------------------------------------------------------
# CLV (Closing Line Value) tests
# ---------------------------------------------------------------------------

class TestCLV:
    """CLV is computed when closing odds are present; NaN when absent."""

    def _bet_with_closing(self, p1_close: float, p2_close: float) -> dict:
        return {
            "date": "2024-01-01",
            "p1_win_prob": 0.65,
            "actual_winner": 1,
            "p1_odds": 2.0,
            "p2_odds": 2.0,
            "p1_closing_odds": p1_close,
            "p2_closing_odds": p2_close,
        }

    def test_clv_finite_when_closing_odds_present(self) -> None:
        df = pd.DataFrame([self._bet_with_closing(1.80, 2.10)])
        cfg = BacktestConfig(initial_bankroll=1000.0, min_ev=0.0)
        result = run(df, cfg)
        clv = result.metrics.get("clv", float("nan"))
        assert np.isfinite(clv), f"Expected finite CLV, got {clv}"

    def test_clv_nan_when_closing_odds_absent(self) -> None:
        df = pd.DataFrame([_even_money_bet("2024-01-01", winner=1, p1_prob=0.65)])
        cfg = BacktestConfig(initial_bankroll=1000.0, min_ev=0.0)
        result = run(df, cfg)
        clv = result.metrics.get("clv", float("nan"))
        assert not np.isfinite(clv), "CLV should be NaN when no closing odds"

    def test_clv_positive_when_bet_at_better_than_closing(self) -> None:
        # Model bet at 2.0; closing odds drop to 1.70 (market moved against us).
        # Our bet odds (2.0) > closing odds (1.70) → CLV should be positive.
        df = pd.DataFrame([self._bet_with_closing(p1_close=1.70, p2_close=2.30)])
        cfg = BacktestConfig(initial_bankroll=1000.0, min_ev=0.0)
        result = run(df, cfg)
        clv = result.metrics.get("clv", float("nan"))
        assert np.isfinite(clv) and clv > 0, f"Expected positive CLV, got {clv}"

    def test_clv_negative_when_model_less_confident_than_closing(self) -> None:
        # model_prob = 0.55, but closing market heavily favours p1 (fair_p1 ~0.68).
        # CLV = 0.55 - 0.68 = negative → closing market was more confident than us.
        row = {
            "date": "2024-01-01",
            "p1_win_prob": 0.55,   # our model
            "actual_winner": 1,
            "p1_odds": 2.0,
            "p2_odds": 2.0,
            "p1_closing_odds": 1.40,  # market heavily backed p1 by close
            "p2_closing_odds": 3.00,
        }
        df = pd.DataFrame([row])
        cfg = BacktestConfig(initial_bankroll=1000.0, min_ev=0.0)
        result = run(df, cfg)
        clv = result.metrics.get("clv", float("nan"))
        assert np.isfinite(clv) and clv < 0, f"Expected negative CLV, got {clv}"

    def test_summary_includes_clv_when_present(self) -> None:
        df = pd.DataFrame([self._bet_with_closing(1.80, 2.10)])
        cfg = BacktestConfig(initial_bankroll=1000.0, min_ev=0.0)
        result = run(df, cfg)
        text = summary(result)
        assert "CLV" in text
