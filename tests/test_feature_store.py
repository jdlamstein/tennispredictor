"""Tests for src/features/feature_store.py.

All tests use synthetic in-memory DataFrames — no real ATP database needed.
"""

from __future__ import annotations

import io
import math
import os
import sys
import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.feature_store import FeatureStore, PlayerState, _update_elo, _INITIAL_ELO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    tourney_date: int,
    id1: int,
    id2: int,
    name1: str,
    name2: str,
    winner: int,
    surface: int = 0,
    hand1: float = 1.0,
    hand2: float = 1.0,
    ht1: float = 185.0,
    ht2: float = 185.0,
    ioc1: float = 0.0,
    ioc2: float = 0.0,
) -> dict:
    return {
        "tourney_date": tourney_date,
        "player1_id": id1,
        "player2_id": id2,
        "player1_name": name1,
        "player2_name": name2,
        "game_winner": winner,
        "surface": surface,
        "player1_hand": hand1,
        "player2_hand": hand2,
        "player1_ht": ht1,
        "player2_ht": ht2,
        "player1_ioc": ioc1,
        "player2_ioc": ioc2,
    }


def _build_store(rows: list[dict]) -> FeatureStore:
    df = pd.DataFrame(rows)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        path = f.name
    try:
        return FeatureStore.build(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# PlayerState unit tests
# ---------------------------------------------------------------------------

class TestPlayerState:
    def test_initial_elo_is_1500(self):
        ps = PlayerState(player_id=1)
        assert ps.elo_hard == _INITIAL_ELO
        assert ps.elo_clay == _INITIAL_ELO
        assert ps.elo_grass == _INITIAL_ELO

    def test_glicko_r_converts_scale(self):
        ps = PlayerState(player_id=1)
        assert abs(ps.glicko_r() - 1500.0) < 1e-6

    def test_surface_elo_returns_correct_attribute(self):
        ps = PlayerState(player_id=1)
        ps.elo_clay = 1600.0
        assert ps.surface_elo("clay") == 1600.0
        assert ps.surface_elo("hard") == _INITIAL_ELO


class TestUpdateElo:
    def test_winner_rating_increases(self):
        new_r = _update_elo(1500.0, 1500.0, a_won=True, games_a=50)
        assert new_r > 1500.0

    def test_loser_rating_decreases(self):
        new_r = _update_elo(1500.0, 1500.0, a_won=False, games_a=50)
        assert new_r < 1500.0

    def test_upset_gives_larger_update(self):
        # Underdog beating favourite → bigger ELO gain
        gain_upset = _update_elo(1400.0, 1600.0, a_won=True, games_a=50) - 1400.0
        gain_normal = _update_elo(1600.0, 1400.0, a_won=True, games_a=50) - 1600.0
        assert gain_upset > gain_normal


# ---------------------------------------------------------------------------
# FeatureStore.build tests
# ---------------------------------------------------------------------------

class TestFeatureStoreBuild:
    def test_players_registered(self):
        rows = [
            _row(20220101, 1, 2, "Novak Djokovic", "Carlos Alcaraz", 1),
        ]
        store = _build_store(rows)
        assert store._resolve("Djokovic") is not None
        assert store._resolve("Alcaraz") is not None

    def test_winner_elo_increases(self):
        rows = [
            _row(20220101, 1, 2, "Novak Djokovic", "Carlos Alcaraz", 1, surface=0),
        ]
        store = _build_store(rows)
        dj = store._resolve("Djokovic")
        al = store._resolve("Alcaraz")
        assert dj.elo_hard > _INITIAL_ELO
        assert al.elo_hard < _INITIAL_ELO

    def test_loser_elo_decreases(self):
        rows = [
            _row(20220101, 1, 2, "Rafael Nadal", "Roger Federer", 1, surface=1),
        ]
        store = _build_store(rows)
        fed = store._resolve("Federer")
        assert fed.elo_clay < _INITIAL_ELO

    def test_winning_streak_tracked(self):
        rows = [
            _row(20220101, 1, 2, "Novak Djokovic", "Carlos Alcaraz", 1),
            _row(20220201, 1, 3, "Novak Djokovic", "Daniil Medvedev", 1),
        ]
        store = _build_store(rows)
        dj = store._resolve("Djokovic")
        assert dj.winning_streak == 2

    def test_losing_streak_resets_after_win(self):
        rows = [
            _row(20220101, 1, 2, "Rafael Nadal", "Roger Federer", 2),  # Nadal loses
            _row(20220201, 1, 3, "Rafael Nadal", "Andy Murray", 1),     # Nadal wins
        ]
        store = _build_store(rows)
        nadal = store._resolve("Nadal")
        assert nadal.losing_streak == 0
        assert nadal.winning_streak == 1

    def test_h2h_counts_correct(self):
        rows = [
            _row(20220101, 1, 2, "Novak Djokovic", "Roger Federer", 1),
            _row(20220201, 1, 2, "Novak Djokovic", "Roger Federer", 1),
            _row(20220301, 1, 2, "Novak Djokovic", "Roger Federer", 2),  # Fed wins
        ]
        store = _build_store(rows)
        dj = store._resolve("Djokovic")
        fed = store._resolve("Federer")
        assert dj.h2h[fed.player_id] == 2
        assert fed.h2h[dj.player_id] == 1

    def test_holdout_year_filters_data(self):
        rows = [
            _row(20210101, 1, 2, "Novak Djokovic", "Roger Federer", 1),
            _row(20220601, 1, 2, "Novak Djokovic", "Roger Federer", 1),  # after cutoff
        ]
        df = pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            store = FeatureStore.build(path, holdout_year=2022)
            dj = store._resolve("Djokovic")
            # Only 1 match in 2021 → winning_streak = 1
            assert dj.winning_streak == 1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# FeatureStore._resolve tests
# ---------------------------------------------------------------------------

class TestResolve:
    def test_full_name_lookup(self):
        rows = [_row(20220101, 1, 2, "Novak Djokovic", "Roger Federer", 1)]
        store = _build_store(rows)
        assert store._resolve("Novak Djokovic") is not None

    def test_surname_only_lookup(self):
        rows = [_row(20220101, 1, 2, "Novak Djokovic", "Roger Federer", 1)]
        store = _build_store(rows)
        assert store._resolve("Djokovic") is not None

    def test_odds_format_lookup(self):
        # "Djokovic N." → first token is "Djokovic"
        rows = [_row(20220101, 1, 2, "Novak Djokovic", "Roger Federer", 1)]
        store = _build_store(rows)
        assert store._resolve("Djokovic N.") is not None

    def test_unknown_player_returns_none(self):
        rows = [_row(20220101, 1, 2, "Novak Djokovic", "Roger Federer", 1)]
        store = _build_store(rows)
        assert store._resolve("Andy Murray") is None

    def test_surname_collision_resolved_by_initial(self):
        # Two "Smith" players — Smith A (id=1) vs Smith B (id=2)
        rows = [
            _row(20220101, 1, 3, "Adam Smith", "Roger Federer", 1),
            _row(20220201, 2, 3, "Brian Smith", "Roger Federer", 2),
        ]
        store = _build_store(rows)
        ps = store._resolve("Adam Smith")
        assert ps is not None
        assert ps.player_id == 1


# ---------------------------------------------------------------------------
# FeatureStore.make_features tests
# ---------------------------------------------------------------------------

class TestMakeFeatures:
    def setup_method(self):
        rows = [
            _row(20220101, 1, 2, "Novak Djokovic", "Carlos Alcaraz", 1),
            _row(20220201, 1, 3, "Novak Djokovic", "Rafael Nadal", 1),
        ]
        self.store = _build_store(rows)

    def test_returns_numpy_array(self):
        X = self.store.make_features("Novak Djokovic", "Carlos Alcaraz")
        assert isinstance(X, np.ndarray)

    def test_shape_is_1_by_n(self):
        X = self.store.make_features("Novak Djokovic", "Carlos Alcaraz")
        assert X.ndim == 2
        assert X.shape[0] == 1
        assert X.shape[1] > 10  # sanity: non-trivial feature count

    def test_feature_count_matches_expected(self):
        # 40 features matching _prepare_features output
        X = self.store.make_features("Novak Djokovic", "Carlos Alcaraz")
        assert X.shape[1] == 40

    def test_unknown_player1_returns_none(self):
        result = self.store.make_features("Andy Murray", "Carlos Alcaraz")
        assert result is None

    def test_unknown_player2_returns_none(self):
        result = self.store.make_features("Novak Djokovic", "Pete Sampras")
        assert result is None

    def test_no_nan_in_output(self):
        X = self.store.make_features("Novak Djokovic", "Rafael Nadal")
        assert not np.any(np.isnan(X))

    def test_surface_param_stored(self):
        X_hard = self.store.make_features("Novak Djokovic", "Rafael Nadal", surface=0)
        X_clay = self.store.make_features("Novak Djokovic", "Rafael Nadal", surface=1)
        # First feature is surface code
        assert X_hard[0, 0] == pytest.approx(0.0)
        assert X_clay[0, 0] == pytest.approx(1.0)

    def test_elo_values_in_expected_range(self):
        X = self.store.make_features("Novak Djokovic", "Carlos Alcaraz")
        # ELO columns (indices 18, 19) should be close to 1500 ± 200 after a few matches
        assert 1300 < X[0, 18] < 1700
        assert 1300 < X[0, 19] < 1700

    def test_winner_has_higher_elo_than_loser(self):
        # Djokovic won all matches → should have higher ELO than Alcaraz
        X = self.store.make_features("Novak Djokovic", "Carlos Alcaraz")
        assert X[0, 18] > X[0, 19]  # p1_elo > p2_elo

    def test_glicko_r_near_1500_initially(self):
        # After few matches Glicko-2 shouldn't deviate wildly from 1500
        X = self.store.make_features("Novak Djokovic", "Carlos Alcaraz")
        assert 1200 < X[0, 20] < 1900  # player1_glicko

    def test_all_features_finite(self):
        X = self.store.make_features("Novak Djokovic", "Carlos Alcaraz")
        assert np.all(np.isfinite(X))


# ---------------------------------------------------------------------------
# FeatureStore.player_info tests
# ---------------------------------------------------------------------------

class TestPlayerInfo:
    def test_returns_dict_for_known_player(self):
        rows = [_row(20220101, 1, 2, "Novak Djokovic", "Carlos Alcaraz", 1)]
        store = _build_store(rows)
        info = store.player_info("Djokovic")
        assert isinstance(info, dict)
        assert "elo_hard" in info
        assert "glicko" in info
        assert "winning_streak" in info

    def test_returns_none_for_unknown(self):
        rows = [_row(20220101, 1, 2, "Novak Djokovic", "Carlos Alcaraz", 1)]
        store = _build_store(rows)
        assert store.player_info("Pete Sampras") is None

    def test_winner_has_positive_winning_streak(self):
        rows = [_row(20220101, 1, 2, "Novak Djokovic", "Carlos Alcaraz", 1)]
        store = _build_store(rows)
        info = store.player_info("Djokovic")
        assert info["winning_streak"] >= 1
