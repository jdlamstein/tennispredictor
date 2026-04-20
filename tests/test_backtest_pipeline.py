"""Regression tests for the backtest pipeline — catches the 5 alignment bugs.

Each test is mapped to the specific bug it would have prevented. All tests use
synthetic deterministic DataFrames; no real data files are required.

Bug catalogue
-------------
Bug 1 – Label extraction after drop:
    game_winner was in _DROP_EXACT and then df.pop("game_winner") was called.
Bug 2 – Case B probability flip:
    p1_win_prob was flipped (1 - p) in Case B, causing bets at loser's odds
    with winner's probability.
Bug 3 – Wrong probability column:
    probs[:, 1] used instead of probs[:, 0] for P(player1 wins).
Bug 4 – Name-collision mismatches:
    No game_winner consistency check after merging on surnames.
Bug 5 – Row-order mismatch:
    _prepare_features sorts internally; match_odds received unsorted DataFrame,
    so probs[i] was assigned to the wrong match.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.backtest import _prepare_features, match_odds
from src.models.classifiers import SklearnPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_row(
    tourney_date: int,
    player1_name: str,
    player2_name: str,
    game_winner: int,
    round_num: int = 1,
) -> dict:
    """Minimal ATP-style row with all columns expected by _prepare_features."""
    return {
        "tourney_date": tourney_date,
        "player1_name": player1_name,
        "player2_name": player2_name,
        "player1_id": 1,
        "player2_id": 2,
        "game_winner": game_winner,
        "surface": 0,
        "round": round_num,
        "year": tourney_date // 10000,
    }


def _odds_row(winner_name: str, loser_name: str, year: int,
              p1_odds: float = 1.5, p2_odds: float = 2.8) -> dict:
    """Minimal odds row matching tennis-data.co.uk format after odds_loader parsing."""
    date_int = year * 10000 + 601
    return {
        "date_int": date_int,
        "p1_name": winner_name,   # "Surname I." format
        "p2_name": loser_name,
        "p1_odds": p1_odds,
        "p2_odds": p2_odds,
        "actual_winner": 1,
    }


def _make_atp_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_odds_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date_int"].astype(str), format="%Y%m%d", errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Bug 5 – Row-order mismatch
# ---------------------------------------------------------------------------

class TestRowOrderAlignment:
    """Tests that catch Bug 5: probs[i] assigned to wrong match when
    test_df is not sorted to match the order used to build X_test."""

    def test_prepare_features_output_is_sorted_by_tourney_date(self) -> None:
        """Row 0 of X must belong to the match with the earliest tourney_date."""
        rows = [
            _base_row(20220501, "Alpha A", "Beta B", 1, round_num=3),
            _base_row(20220101, "Gamma G", "Delta D", 2, round_num=7),  # earliest
            _base_row(20220301, "Epsilon E", "Zeta Z", 1, round_num=2),
        ]
        df = _make_atp_df(rows)
        X, y, feat_cols = _prepare_features(df)

        round_idx = feat_cols.index("round")
        # Earliest date row has round=7; it must be first after sort.
        assert X[0, round_idx] == pytest.approx(7.0), (
            "Row 0 of X must correspond to the match with the smallest tourney_date."
        )

    def test_match_odds_sorts_assertion_fires_on_unsorted_input(self) -> None:
        """match_odds must raise AssertionError when test_df is not sorted."""
        rows = [
            _base_row(20220501, "Djokovic N", "Federer R", 1),
            _base_row(20220101, "Alcaraz C", "Sinner J", 1),  # earlier but placed second
        ]
        df_unsorted = _make_atp_df(rows)  # NOT sorted by tourney_date
        probs = np.array([[0.8, 0.2], [0.7, 0.3]])
        odds = _make_odds_df([_odds_row("Djokovic N.", "Federer R.", 2022)])

        with pytest.raises(AssertionError, match="sorted by tourney_date"):
            match_odds(df_unsorted, odds, probs)

    def test_match_odds_probs_aligned_to_sorted_df(self) -> None:
        """p1_win_prob in output must match the probability for that player-pair,
        not for whatever row happened to be at the same index in the unsorted df."""
        # ATP "First Surname" format: surname = last token.
        # Odds "Surname I." format: surname = first token.
        # "John Smith" → p1_sn = "smith" matches odds winner "Smith J." first token "smith".
        rows = [
            _base_row(20220601, "John Smith", "Kevin Jones", 1),
            _base_row(20220101, "Dmitri Novak", "Roger Federer", 2),  # earlier
        ]
        test_sorted = _make_atp_df(rows).sort_values("tourney_date").reset_index(drop=True)
        # After sort: row 0 = Dmitri Novak (20220101), row 1 = John Smith (20220601)
        probs = np.array([
            [0.35, 0.65],  # row 0 → Dmitri Novak row
            [0.90, 0.10],  # row 1 → John Smith row: p1_win_prob = 0.90
        ])
        odds = _make_odds_df([
            _odds_row("Smith J.", "Jones K.", 2022),  # only Smith row in odds
        ])
        result = match_odds(test_sorted, odds, probs)

        # John Smith (sorted row 1) should have p1_win_prob ≈ 0.90
        smith_rows = result[result["p1_sn"] == "smith"]
        assert not smith_rows.empty, "John Smith should be in merged output"
        assert smith_rows.iloc[0]["p1_win_prob"] == pytest.approx(0.90, abs=1e-4)

    def test_prepare_features_X_y_lengths_match(self) -> None:
        """len(X) must equal len(y) — basic alignment sanity check."""
        rows = [_base_row(20220101 + i * 100, f"P{i} A", f"Q{i} B", (i % 2) + 1)
                for i in range(7)]
        df = _make_atp_df(rows)
        X, y, _ = _prepare_features(df)
        assert len(X) == len(y) == 7


# ---------------------------------------------------------------------------
# Bug 1 – Label extraction after drop
# ---------------------------------------------------------------------------

class TestLabelEncoding:
    """Tests that catch Bug 1: game_winner dropped before label extraction."""

    def test_label_encoding_game_winner_to_y(self) -> None:
        """y = game_winner - 1: player1 wins → 0, player2 wins → 1."""
        rows = [
            _base_row(20220101, "A B", "C D", game_winner=1),
            _base_row(20220201, "E F", "G H", game_winner=2),
            _base_row(20220301, "I J", "K L", game_winner=1),
            _base_row(20220401, "M N", "O P", game_winner=2),
        ]
        _, y, _ = _prepare_features(_make_atp_df(rows))
        np.testing.assert_array_equal(y, [0, 1, 0, 1])

    def test_prepare_features_no_keyerror_despite_drop_exact(self) -> None:
        """_prepare_features must not raise KeyError for game_winner even though
        it appears in _DROP_EXACT."""
        rows = [_base_row(20220101, "A B", "C D", 1)]
        try:
            X, y, feat_cols = _prepare_features(_make_atp_df(rows))
        except KeyError as exc:
            pytest.fail(f"_prepare_features raised KeyError: {exc}")

    def test_feature_columns_do_not_contain_game_winner(self) -> None:
        """game_winner must not appear in the returned feature column list."""
        rows = [_base_row(20220101, "A B", "C D", 1)]
        _, _, feat_cols = _prepare_features(_make_atp_df(rows))
        assert "game_winner" not in feat_cols, (
            "'game_winner' must not be a feature — it is the label"
        )
        assert "tourney_date" not in feat_cols
        assert "player1_name" not in feat_cols


# ---------------------------------------------------------------------------
# Bug 3 – Wrong probability column
# ---------------------------------------------------------------------------

class TestProbabilityColumnConvention:
    """Tests that catch Bug 3: probs[:, 1] used instead of probs[:, 0]."""

    def test_predict_proba_col0_is_player1_wins(self) -> None:
        """SklearnPredictor wrapping GaussianNB: column 0 = P(player1 wins).

        Trivially separable data: feature=0 → player1 wins (label=0),
        feature=10 → player2 wins (label=1).
        """
        rng = np.random.default_rng(0)
        X_train = np.concatenate([
            rng.normal(0, 0.1, (100, 1)),   # class 0: player1 wins
            rng.normal(10, 0.1, (100, 1)),  # class 1: player2 wins
        ])
        y_train = np.array([0] * 100 + [1] * 100)

        predictor = SklearnPredictor(GaussianNB(), name="test")
        predictor.fit(X_train, y_train)

        # Predict on a sample clearly in class 0 territory (player1 wins)
        X_test = np.array([[0.05]])
        probs = predictor.predict_proba(X_test)

        assert probs.shape == (1, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert probs[0, 0] > 0.9, (
            f"probs[:, 0] should be P(player1 wins) ≈ 1.0, got {probs[0, 0]:.3f}. "
            "If this fails, the column convention is wrong — check predict_proba."
        )

    def test_match_odds_uses_col0_not_col1(self) -> None:
        """match_odds assigns probs[:, 0] to p1_win_prob.

        If probs[:, 1] were used (Bug 3), the value 0.35 would appear instead of 0.75.
        """
        # "Novak Djokovic" → p1_sn = "djokovic" matches odds "Djokovic N." first token "djokovic".
        rows = [_base_row(20220101, "Novak Djokovic", "Roger Federer", game_winner=1)]
        test_sorted = _make_atp_df(rows).sort_values("tourney_date").reset_index(drop=True)
        # col 0 = 0.75 (P player1 wins), col 1 = 0.25
        probs = np.array([[0.75, 0.25]])
        odds = _make_odds_df([_odds_row("Djokovic N.", "Federer R.", 2022)])

        result = match_odds(test_sorted, odds, probs)
        assert not result.empty
        assert result.iloc[0]["p1_win_prob"] == pytest.approx(0.75, abs=1e-4), (
            "p1_win_prob must equal probs[:, 0] (= 0.75), not probs[:, 1] (= 0.25). "
            "Using the wrong column inverts all predictions."
        )


# ---------------------------------------------------------------------------
# Bug 2 – Case B probability flip
# ---------------------------------------------------------------------------

class TestCaseBNoProbabilityFlip:
    """Tests that catch Bug 2: p1_win_prob flipped in Case B."""

    def test_case_b_p1_win_prob_unchanged(self) -> None:
        """In Case B (our p1 = odds loser), p1_win_prob must NOT be flipped.

        The model predicts low P(player1 wins) because player1 is the loser.
        Flipping to 1 - p would make it appear player1 is the winner, causing
        the backtester to bet at loser's odds with winner's probability.
        """
        # player1 = "John Smith" (loser), player2 = "Kevin Jones" (winner)
        # ATP p1_sn = "smith" (last token), odds loser p2_name = "Smith J." first token "smith" ✓
        rows = [_base_row(20220101, "John Smith", "Kevin Jones", game_winner=2)]
        test_sorted = _make_atp_df(rows).sort_values("tourney_date").reset_index(drop=True)

        original_p1_win_prob = 0.20  # model correctly says player1 (loser) unlikely to win
        probs = np.array([[original_p1_win_prob, 1.0 - original_p1_win_prob]])

        # Odds file has Jones as winner, Smith as loser → Case B match
        odds = _make_odds_df([_odds_row("Jones K.", "Smith J.", 2022)])
        result = match_odds(test_sorted, odds, probs)

        assert not result.empty, "Case B row should be in matched output"
        stored = result.iloc[0]["p1_win_prob"]
        assert stored == pytest.approx(original_p1_win_prob, abs=1e-4), (
            f"p1_win_prob must remain {original_p1_win_prob} (not flipped to "
            f"{1.0 - original_p1_win_prob}). Flipping inverts all Case B bets."
        )

    def test_case_b_actual_winner_is_two(self) -> None:
        """Case B rows must have actual_winner = 2 (player2 won)."""
        rows = [_base_row(20220101, "John Smith", "Kevin Jones", game_winner=2)]
        test_sorted = _make_atp_df(rows).sort_values("tourney_date").reset_index(drop=True)
        probs = np.array([[0.20, 0.80]])
        odds = _make_odds_df([_odds_row("Jones K.", "Smith J.", 2022)])
        result = match_odds(test_sorted, odds, probs)
        assert not result.empty
        assert result.iloc[0]["actual_winner"] == 2


# ---------------------------------------------------------------------------
# Bug 4 – Name-collision mismatches
# ---------------------------------------------------------------------------

class TestNameCollisionFilter:
    """Tests that catch Bug 4: no game_winner consistency check after merge."""

    def test_case_a_filters_out_game_winner_2_rows(self) -> None:
        """In Case A (p1_sn = winner surname), only game_winner=1 rows must survive.

        A row where player1 surname matches the odds winner but game_winner=2
        is a name collision — the wrong match was pulled in. It must be dropped.
        """
        # ATP "First Surname" format: "John Smith" → p1_sn = "smith".
        # Odds "Surname I." format: "Smith J." → first token = "smith" ✓
        rows = [
            _base_row(20220101, "John Smith", "Kevin Jones", game_winner=1),  # keep
            _base_row(20220102, "John Smith", "Leo Brown", game_winner=2),    # same p1 surname, wrong winner → drop
        ]
        test_sorted = _make_atp_df(rows).sort_values("tourney_date").reset_index(drop=True)
        probs = np.array([[0.8, 0.2], [0.8, 0.2]])
        odds = _make_odds_df([_odds_row("Smith J.", "Jones K.", 2022)])

        result = match_odds(test_sorted, odds, probs)
        # Only the row with game_winner=1 survives
        assert (result["actual_winner"] == 1).all(), (
            "Case A rows with game_winner=2 are name-collision mismatches and must be dropped."
        )

    def test_dedup_prevents_double_counting_same_physical_match(self) -> None:
        """The same physical match appearing in both Case A and Case B must
        produce exactly one row in the output (not two)."""
        # Djokovic beat Federer. ATP DB has two rows (one per ordering):
        rows = [
            _base_row(20220101, "Novak Djokovic", "Roger Federer", game_winner=1),  # Case A
            _base_row(20220101, "Roger Federer", "Novak Djokovic", game_winner=2),  # Case B
        ]
        test_sorted = _make_atp_df(rows).sort_values("tourney_date").reset_index(drop=True)
        probs = np.array([[0.75, 0.25], [0.25, 0.75]])
        odds = _make_odds_df([_odds_row("Djokovic N.", "Federer R.", 2022)])

        result = match_odds(test_sorted, odds, probs)
        assert len(result) == 1, (
            f"The same physical match appeared {len(result)} times in output. "
            "Dedup on sorted surnames must reduce it to exactly 1."
        )

    def test_surname_extraction_atp_format_last_token(self) -> None:
        """ATP name 'Novak Djokovic' → surname = 'djokovic' (last token)."""
        rows = [_base_row(20220101, "Novak Djokovic", "Roger Federer", game_winner=1)]
        test_sorted = _make_atp_df(rows).sort_values("tourney_date").reset_index(drop=True)
        probs = np.array([[0.7, 0.3]])
        # Odds file uses "Surname I." format (first token = surname)
        odds = _make_odds_df([_odds_row("Djokovic N.", "Federer R.", 2022)])

        result = match_odds(test_sorted, odds, probs)
        assert not result.empty, (
            "ATP 'Novak Djokovic' (last token = 'djokovic') must match "
            "odds 'Djokovic N.' (first token = 'djokovic')."
        )
