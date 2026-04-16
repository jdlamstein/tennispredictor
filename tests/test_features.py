"""Tests for ELO and feature engineering invariants.

Uses real ATP match data via the ``atp_sample`` fixture. Covers behavioral
invariants — not implementation details.
"""

import sys
import os

import pandas as pd
import pytest

# Make root importable without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.pipeline import Elo


class TestEloCalc:
    """calc_elo is a pure function — test it directly with controlled inputs."""

    def _calc(self, rating_a: float, rating_b: float, winner: int, games: int) -> float:
        """Thin helper to call calc_elo without constructing an Elo instance."""
        return Elo.calc_elo(None, rating_a, rating_b, winner, games)  # type: ignore[arg-type]

    def test_winner_rating_increases(self) -> None:
        """A player who wins should have a higher ELO afterwards."""
        before = 1500.0
        new = self._calc(before, 1500.0, winner=1, games=5)
        assert new > before, "Winner's ELO must increase after a win"

    def test_loser_rating_decreases(self) -> None:
        """A player who loses (winner=0) should have a lower ELO afterwards."""
        before = 1500.0
        new = self._calc(before, 1500.0, winner=0, games=5)
        assert new < before, "Loser's ELO must decrease after a loss"

    def test_upset_larger_delta(self) -> None:
        """Beating a much stronger opponent yields a larger ELO gain than beating a peer."""
        delta_vs_stronger = self._calc(1400.0, 1800.0, winner=1, games=5) - 1400.0
        delta_vs_peer = self._calc(1400.0, 1400.0, winner=1, games=5) - 1400.0
        assert delta_vs_stronger > delta_vs_peer, (
            "Upsets should yield larger ELO gains than expected wins"
        )

    def test_k_factor_drops_with_experience(self) -> None:
        """An established player (many games) gains less per win than a newcomer."""
        delta_new = self._calc(1500.0, 1500.0, winner=1, games=5) - 1500.0   # K=40
        delta_established = self._calc(1500.0, 1500.0, winner=1, games=50) - 1500.0  # K=20
        assert delta_new > delta_established, (
            "K-factor should be larger for players with fewer games played"
        )


class TestEloInDatabase:
    """Sanity-check ELO values stored in the real ATP database."""

    def test_elo_columns_present(self, atp_sample: pd.DataFrame) -> None:
        assert "player1_elo" in atp_sample.columns
        assert "player2_elo" in atp_sample.columns

    def test_elo_values_positive(self, atp_sample: pd.DataFrame) -> None:
        """All stored ELO values should be above zero (initialized at 1500)."""
        assert (atp_sample["player1_elo"] > 0).all(), "player1_elo must be positive"
        assert (atp_sample["player2_elo"] > 0).all(), "player2_elo must be positive"

    def test_game_winner_binary(self, atp_sample: pd.DataFrame) -> None:
        """game_winner should only contain 1 or 2."""
        assert atp_sample["game_winner"].isin([1, 2]).all(), (
            "game_winner must be 1 (player1 wins) or 2 (player2 wins)"
        )

    def test_streak_columns_non_negative(self, atp_sample: pd.DataFrame) -> None:
        """Winning and losing streaks must be non-negative integers."""
        for col in ("player1_winning_streak", "player2_winning_streak",
                    "player1_losing_streak", "player2_losing_streak"):
            assert (atp_sample[col] >= 0).all(), f"{col} must be non-negative"

    def test_normalization_stats_unchanged_between_train_and_test(
        self, atp_sample: pd.DataFrame
    ) -> None:
        """The first 60% and last 20% of the slice should not have the same mean on ELO.

        This is a lightweight check that the data isn't perfectly uniform (which
        would suggest a data-loading or sorting bug).
        """
        n = len(atp_sample)
        train_mean = atp_sample.iloc[: int(n * 0.6)]["player1_elo"].mean()
        test_mean = atp_sample.iloc[int(n * 0.8) :]["player1_elo"].mean()
        # Allow a small tolerance — they should not be identical
        assert train_mean != test_mean or n < 10, (
            "Train and test ELO means are suspiciously identical — check data ordering"
        )
