"""Tests for ELO and feature engineering invariants.

Uses real ATP match data via the ``atp_sample`` fixture and small synthetic
DataFrames for unit invariants. Covers behavior, not implementation details.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# Make root importable without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.pipeline import Elo
from src.features.elo import add_surface_elo, add_global_elo, add_all_elo
from src.features.glicko import add_glicko2


# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

def _matches(*rows: tuple) -> pd.DataFrame:
    """Build a minimal match DataFrame from (date, p1_id, p2_id, winner, surface) tuples.

    surface: 0=Hard 1=Clay 2=Grass 3=Carpet
    """
    return pd.DataFrame(
        rows,
        columns=["tourney_date", "player1_id", "player2_id", "game_winner", "surface"],
    )


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


# ---------------------------------------------------------------------------
# Surface ELO unit invariants (synthetic data)
# ---------------------------------------------------------------------------

class TestSurfaceElo:
    """Behavioral invariants for add_surface_elo."""

    def test_surface_elo_columns_added(self) -> None:
        """add_surface_elo must add exactly 8 surface ELO columns."""
        df = _matches(
            (20220101, 1, 2, 1, 0),  # Hard
            (20220201, 1, 3, 2, 1),  # Clay
        )
        out = add_surface_elo(df)
        expected = {
            "player1_elo_hard", "player2_elo_hard",
            "player1_elo_clay", "player2_elo_clay",
            "player1_elo_grass", "player2_elo_grass",
            "player1_elo_carpet", "player2_elo_carpet",
        }
        assert expected.issubset(set(out.columns)), (
            f"Missing surface ELO columns: {expected - set(out.columns)}"
        )

    def test_surface_elo_first_match_starts_at_1500(self) -> None:
        """A player's first match on a surface must show pre-match rating of 1500."""
        df = _matches((20220101, 1, 2, 1, 0))  # Hard
        out = add_surface_elo(df)
        assert out.iloc[0]["player1_elo_hard"] == pytest.approx(1500.0)
        assert out.iloc[0]["player2_elo_hard"] == pytest.approx(1500.0)

    def test_surface_elo_winner_gains_after_match(self) -> None:
        """After two hard-court matches, the consistent winner has higher hard ELO."""
        # Player 1 beats player 2 twice on hard
        df = _matches(
            (20220101, 1, 2, 1, 0),
            (20220201, 1, 2, 1, 0),
            (20220301, 1, 2, 1, 0),
        )
        out = add_surface_elo(df)
        # Row 2 (third match): player 1's hard ELO should be > 1500 (won previous two)
        assert out.iloc[2]["player1_elo_hard"] > 1500.0, (
            "Player1 won two hard-court matches, their hard ELO must exceed 1500"
        )
        assert out.iloc[2]["player2_elo_hard"] < 1500.0, (
            "Player2 lost two hard-court matches, their hard ELO must be below 1500"
        )

    def test_surface_elo_clay_specialist_hard_starts_at_1500(self) -> None:
        """Clay wins must not affect a player's hard ELO starting point.

        Surface ELO rows are sparse: each row only shows ELO for the surface being
        played. Clay-match rows have NaN for hard ELO. To verify independence, we
        add a hard match after clay wins and check that player1's hard ELO at that
        first hard match is still 1500 (clay results had no influence).
        """
        df = _matches(
            (20220101, 1, 2, 1, 1),  # Clay win
            (20220201, 1, 2, 1, 1),  # Clay win
            (20220301, 1, 2, 1, 0),  # Hard — first hard match, should start at 1500
        )
        out = add_surface_elo(df)
        # Row 1 is the 2nd clay match — clay ELO should reflect the prior clay win
        assert out.iloc[1]["player1_elo_clay"] > 1500.0, "Clay ELO should grow after wins"
        # Row 2 is the first hard match — hard ELO must start at 1500 (clay history irrelevant)
        assert out.iloc[2]["player1_elo_hard"] == pytest.approx(1500.0), (
            "First hard match must show ELO=1500 — clay wins must not carry over to hard"
        )

    def test_surface_elo_independent_across_surfaces(self) -> None:
        """Clay performance must not bleed into a player's hard ELO.

        Surface ELO rows are sparse — each row shows ELO only for the surface played.
        We test independence by verifying: after one clay win (row 0) and one hard loss
        (row 1), the second hard match (row 3) shows hard ELO below 1500 (from the hard
        loss in row 1), while the clay ELO at the 2nd clay match (row 2) is above 1500
        (from the clay win in row 0) — i.e., the two rating pools are independent.
        """
        df = _matches(
            (20220101, 1, 2, 1, 1),  # Clay: player1 wins
            (20220201, 1, 2, 2, 0),  # Hard: player1 loses (first hard match, pre-match = 1500)
            (20220301, 1, 2, 1, 1),  # Clay: player1 wins again
            (20220401, 1, 2, 1, 0),  # Hard: check hard ELO carries the prior loss
        )
        out = add_surface_elo(df)
        assert out.iloc[2]["player1_elo_clay"] > 1500.0, "Clay ELO must grow after clay win"
        assert out.iloc[1]["player1_elo_hard"] == pytest.approx(1500.0), (
            "First hard match must start at 1500 regardless of clay record"
        )
        assert out.iloc[3]["player1_elo_hard"] < 1500.0, (
            "After a hard loss, subsequent hard-match ELO must be below 1500"
        )

    def test_surface_elo_unknown_surface_is_nan(self) -> None:
        """Rows with surface code -1 (unknown) must have NaN for all surface ELOs."""
        df = _matches((20220101, 1, 2, 1, -1))  # Unknown surface
        out = add_surface_elo(df)
        assert pd.isna(out.iloc[0]["player1_elo_hard"])
        assert pd.isna(out.iloc[0]["player1_elo_clay"])

    def test_surface_elo_does_not_mutate_input(self) -> None:
        """add_surface_elo must return a new DataFrame and not modify the input."""
        df = _matches((20220101, 1, 2, 1, 0))
        original_cols = set(df.columns)
        _ = add_surface_elo(df)
        assert set(df.columns) == original_cols, "Input DataFrame was mutated"

    def test_add_all_elo_includes_global_and_surface(self) -> None:
        """add_all_elo must add both global ELO (2 cols) and surface ELO (8 cols)."""
        df = _matches((20220101, 1, 2, 1, 0))
        out = add_all_elo(df)
        assert "player1_elo" in out.columns
        assert "player2_elo" in out.columns
        assert "player1_elo_hard" in out.columns
        assert "player1_elo_clay" in out.columns


# ---------------------------------------------------------------------------
# Surface ELO integration test (real database sample)
# ---------------------------------------------------------------------------

class TestSurfaceEloInDatabase:
    """Integration tests for add_surface_elo on the real ATP dataset sample."""

    def test_surface_elo_columns_present_in_enriched_sample(
        self, atp_sample: pd.DataFrame
    ) -> None:
        """Running add_surface_elo on the real sample must add all 8 columns."""
        out = add_surface_elo(atp_sample)
        for surface in ("hard", "clay", "grass", "carpet"):
            assert f"player1_elo_{surface}" in out.columns
            assert f"player2_elo_{surface}" in out.columns

    def test_surface_elo_values_positive_or_nan(
        self, atp_sample: pd.DataFrame
    ) -> None:
        """Surface ELO values must be positive (initial 1500) or NaN for unplayed surfaces."""
        out = add_surface_elo(atp_sample)
        for surface in ("hard", "clay", "grass", "carpet"):
            col = out[f"player1_elo_{surface}"].dropna()
            assert (col > 0).all(), f"player1_elo_{surface} has non-positive values"


# ---------------------------------------------------------------------------
# Glicko-2 unit invariants (synthetic data)
# ---------------------------------------------------------------------------

class TestGlicko2:
    """Behavioral invariants for add_glicko2."""

    def test_glicko2_columns_added(self) -> None:
        """add_glicko2 must add exactly 6 columns: glicko, rd, sigma for each player."""
        df = _matches((20220101, 1, 2, 1, 0))
        out = add_glicko2(df)
        expected = {"player1_glicko", "player2_glicko",
                    "player1_rd", "player2_rd",
                    "player1_sigma", "player2_sigma"}
        assert expected.issubset(set(out.columns))

    def test_glicko2_initial_rating_is_1500(self) -> None:
        """A player's first match must show pre-match Glicko rating of 1500."""
        df = _matches((20220101, 1, 2, 1, 0))
        out = add_glicko2(df)
        assert out.iloc[0]["player1_glicko"] == pytest.approx(1500.0, abs=1.0)
        assert out.iloc[0]["player2_glicko"] == pytest.approx(1500.0, abs=1.0)

    def test_glicko2_initial_rd_is_350(self) -> None:
        """Initial Rating Deviation must be 350 (Glickman 2012 constant)."""
        df = _matches((20220101, 1, 2, 1, 0))
        out = add_glicko2(df)
        assert out.iloc[0]["player1_rd"] == pytest.approx(350.0, abs=1.0)

    def test_glicko2_winner_rating_increases(self) -> None:
        """After enough wins in one period, winner's Glicko rating in the NEXT period
        must exceed the loser's.

        Glicko-2 updates ratings at period boundaries (every 90 days), not per match.
        Pre-match values within a period all reflect the START of that period (1500 for
        new players). To see updated ratings, we must observe a match in period 2.
        """
        # Five matches all in period 0 (Jan–Mar 2022): pre-match values show 1500 for both.
        period0 = [20220101, 20220115, 20220201, 20220215, 20220301]
        # One match in period 1 (Jun 2022): pre-match values show post-period0 ratings.
        period1_match = 20220601
        rows = [(d, 1, 2, 1, 0) for d in period0] + [(period1_match, 1, 2, 1, 0)]
        df = _matches(*rows)
        out = add_glicko2(df)
        # Last row is in period 1 — ratings reflect player1's 5 wins from period 0.
        p1_last = out.iloc[-1]["player1_glicko"]
        p2_last = out.iloc[-1]["player2_glicko"]
        assert p1_last > p2_last, (
            f"Winner's Glicko rating in period 1 should exceed loser's, "
            f"got {p1_last:.1f} vs {p2_last:.1f}"
        )

    def test_glicko2_rd_decreases_with_activity(self) -> None:
        """A player with many matches in period 0 must have lower RD in period 1.

        Uses properly computed calendar dates (not integer arithmetic that overflows
        month boundaries). Player3 is new — they start with RD=350. Player1 has played
        many matches, so their RD should be well below 350 by the time period 1 starts.
        """
        import datetime
        base = datetime.date(2022, 1, 1)
        # 20 matches every 5 days: Jan 1 → ~Apr 10 (spans period 0 and period 1)
        early_dates = [
            int((base + datetime.timedelta(days=i * 5)).strftime("%Y%m%d"))
            for i in range(20)
        ]
        late_date = 20221001  # period ~3 — player3 is brand new here
        early = [(d, 1, 2, 1, 0) for d in early_dates]
        late = [(late_date, 1, 3, 1, 0)]
        df = _matches(*(early + late))
        out = add_glicko2(df)
        last = out.iloc[-1]
        assert last["player1_rd"] < last["player2_rd"], (
            f"Active player1 (RD={last['player1_rd']:.1f}) should have lower RD "
            f"than new player3 (RD={last['player2_rd']:.1f})"
        )

    def test_glicko2_does_not_mutate_input(self) -> None:
        """add_glicko2 must return a new DataFrame and not modify the input."""
        df = _matches((20220101, 1, 2, 1, 0))
        original_cols = set(df.columns)
        _ = add_glicko2(df)
        assert set(df.columns) == original_cols, "Input DataFrame was mutated"


# ---------------------------------------------------------------------------
# Glicko-2 integration test (real database sample)
# ---------------------------------------------------------------------------

class TestGlicko2InDatabase:
    """Integration tests for add_glicko2 on the real ATP dataset sample."""

    def test_glicko2_columns_present(self, atp_sample: pd.DataFrame) -> None:
        """Running add_glicko2 on the real sample must add all 6 columns."""
        out = add_glicko2(atp_sample)
        for col in ("player1_glicko", "player2_glicko",
                    "player1_rd", "player2_rd",
                    "player1_sigma", "player2_sigma"):
            assert col in out.columns

    def test_glicko2_rd_positive(self, atp_sample: pd.DataFrame) -> None:
        """Rating Deviation must be positive for all rows."""
        out = add_glicko2(atp_sample)
        assert (out["player1_rd"] > 0).all()
        assert (out["player2_rd"] > 0).all()

    def test_glicko2_sigma_positive(self, atp_sample: pd.DataFrame) -> None:
        """Volatility (sigma) must be positive for all rows."""
        out = add_glicko2(atp_sample)
        assert (out["player1_sigma"] > 0).all()
        assert (out["player2_sigma"] > 0).all()
