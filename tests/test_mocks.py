"""Mock-based unit tests that run without real data files or trained models.

Covers:
- Backtester input validation and edge-case branching
- TemperatureScaling calibration softens overconfident probabilities
- CalibratedPredictor delegates correctly to a mocked base predictor
- OddsLoader column mapping and date_int construction
- BasePredictor interface contract enforcement
"""

import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.backtester import BacktestConfig, run, summary
from src.evaluation.calibration import CalibratedPredictor, TemperatureScaling, blend_with_market
from src.data.odds_loader import _parse_one


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_match_df(**overrides) -> pd.DataFrame:
    """One-row DataFrame with required backtester columns."""
    row = {
        "date": "2024-06-01",
        "p1_win_prob": 0.65,
        "actual_winner": 1,
        "p1_odds": 1.80,
        "p2_odds": 2.20,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _mock_predictor(probs: np.ndarray) -> MagicMock:
    """Return a BasePredictor mock whose predict_proba returns ``probs``."""
    pred = MagicMock()
    pred.predict_proba.return_value = probs
    return pred


# ---------------------------------------------------------------------------
# Backtester — input validation
# ---------------------------------------------------------------------------

class TestBacktesterValidation:
    """run() must raise or return gracefully on bad input."""

    def test_missing_required_column_raises(self) -> None:
        df = _make_match_df()
        df = df.drop(columns=["p1_odds"])
        with pytest.raises(ValueError, match="missing columns"):
            run(df)

    def test_empty_dataframe_returns_no_bets(self) -> None:
        df = pd.DataFrame(columns=["date", "p1_win_prob", "actual_winner", "p1_odds", "p2_odds"])
        result = run(df)
        assert result.bets.empty
        assert result.metrics.get("n_bets", 0) == 0

    def test_below_min_ev_no_bets_placed(self) -> None:
        """p1_win_prob=0.5 at 2.0/2.0 → EV=0 on both sides. min_ev=0.05 → no bet."""
        df = _make_match_df(p1_win_prob=0.5, p1_odds=2.0, p2_odds=2.0)
        cfg = BacktestConfig(min_ev=0.05)
        result = run(df, cfg)
        assert result.bets.empty

    def test_exhausted_bankroll_stops_early(self) -> None:
        """Bankroll of 0 after first losing bet — backtester must not crash."""
        rows = [
            {"date": f"2024-01-{i+1:02d}", "p1_win_prob": 0.9,
             "actual_winner": 2, "p1_odds": 1.5, "p2_odds": 3.0}
            for i in range(50)
        ]
        df = pd.DataFrame(rows)
        cfg = BacktestConfig(initial_bankroll=10.0, kelly_fraction=1.0, min_ev=0.0)
        result = run(df, cfg)
        assert result.metrics.get("final_bankroll", 0.0) >= 0.0


# ---------------------------------------------------------------------------
# TemperatureScaling — calibration behaviour
# ---------------------------------------------------------------------------

class TestTemperatureScaling:
    """Temperature scaling must soften overconfident probability outputs."""

    def _overconfident_probs(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Synthetic probs near 0 or 1; true outcomes are 70/30 (not 99/1)."""
        rng = np.random.default_rng(0)
        p = np.where(rng.random(n) < 0.7, 0.99, 0.01)
        probs = np.stack([p, 1.0 - p], axis=1)  # col0 = P(p1 wins), correct convention
        outcomes = (rng.random(n) < 0.7).astype(float)
        return probs, outcomes

    def test_temperature_greater_than_one_for_overconfident_model(self) -> None:
        """T > 1 means the calibrator softened the distribution."""
        probs, outcomes = self._overconfident_probs()
        ts = TemperatureScaling()
        ts.fit(probs, outcomes)
        assert ts.temperature_ is not None
        assert ts.temperature_ > 1.0, "Overconfident model should need T > 1"

    def test_transform_reduces_extreme_probabilities(self) -> None:
        """After calibration, probabilities should be less extreme."""
        probs, outcomes = self._overconfident_probs()
        ts = TemperatureScaling().fit(probs, outcomes)
        calibrated = ts.transform(probs)
        assert calibrated.shape == probs.shape
        # Max calibrated P(player1 wins) must be strictly lower than 0.99
        assert calibrated[:, 0].max() < 0.99

    def test_transform_rows_sum_to_one(self) -> None:
        probs, outcomes = self._overconfident_probs()
        ts = TemperatureScaling().fit(probs, outcomes)
        calibrated = ts.transform(probs)
        assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-9)

    def test_transform_before_fit_raises(self) -> None:
        ts = TemperatureScaling()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            ts.transform(np.array([[0.3, 0.7]]))


# ---------------------------------------------------------------------------
# CalibratedPredictor — delegates to mocked base predictor
# ---------------------------------------------------------------------------

class TestCalibratedPredictor:
    """CalibratedPredictor must call its base predictor and apply calibration."""

    def _raw_probs(self) -> np.ndarray:
        """Extreme probabilities (overconfident model output)."""
        return np.array([[0.01, 0.99], [0.99, 0.01], [0.05, 0.95]])

    def test_fit_delegates_to_base(self) -> None:
        base = _mock_predictor(self._raw_probs())
        cp = CalibratedPredictor(base)
        X = np.zeros((3, 4))
        y = np.array([1, 0, 1])
        cp.fit(X, y)
        base.fit.assert_called_once_with(X, y)

    def test_predict_proba_before_calibrate_returns_raw(self) -> None:
        raw = self._raw_probs()
        base = _mock_predictor(raw)
        cp = CalibratedPredictor(base)
        X = np.zeros((3, 4))
        out = cp.predict_proba(X)
        assert np.array_equal(out, raw)

    def test_calibrate_then_predict_softens_output(self) -> None:
        # Build overconfident model: always says 0.99, but only right 65% of the time
        rng = np.random.default_rng(42)
        n = 300
        raw = np.tile([0.99, 0.01], (n, 1)).astype(float)  # col0=P(p1)=overconfident
        outcomes = (rng.random(n) < 0.65).astype(float)  # true win rate = 65%, not 99%
        base = _mock_predictor(raw)
        cp = CalibratedPredictor(base)
        cp.calibrate(np.zeros((n, 4)), outcomes)
        calibrated = cp.predict_proba(np.zeros((n, 4)))
        # Calibrated p1_win_prob (col 0) must be softer than raw 0.99
        assert calibrated[:, 0].max() < 0.99

    def test_save_raises_not_implemented(self) -> None:
        base = _mock_predictor(self._raw_probs())
        cp = CalibratedPredictor(base)
        with pytest.raises(NotImplementedError):
            cp.save("/tmp/model.json")

    def test_load_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            CalibratedPredictor.load("/tmp/model.json")


# ---------------------------------------------------------------------------
# OddsLoader — column mapping and date_int with mocked Excel file
# ---------------------------------------------------------------------------

class TestOddsLoaderParsing:
    """_parse_one must map columns and compute date_int without real files."""

    def _make_raw_odds_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Date": ["01/06/2024", "15/07/2024"],
            "Winner": ["Djokovic N.", "Alcaraz C."],
            "Loser": ["Federer R.", "Sinner J."],
            "PSW": [1.45, 1.60],
            "PSL": [2.90, 2.40],
            "B365W": [1.44, 1.58],
            "B365L": [2.85, 2.35],
        })

    def test_column_rename_applied(self, tmp_path: Path) -> None:
        """p1_name, p1_odds, p2_odds must appear in output."""
        raw = self._make_raw_odds_df()
        fake_path = tmp_path / "atp_2024.xlsx"
        with patch("src.data.odds_loader.pd.read_excel", return_value=raw):
            df = _parse_one(fake_path)
        assert "p1_name" in df.columns
        assert "p1_odds" in df.columns
        assert "p2_odds" in df.columns

    def test_date_int_format(self, tmp_path: Path) -> None:
        """date_int must be YYYYMMDD integer."""
        raw = self._make_raw_odds_df()
        fake_path = tmp_path / "atp_2024.xlsx"
        with patch("src.data.odds_loader.pd.read_excel", return_value=raw):
            df = _parse_one(fake_path)
        assert df["date_int"].iloc[0] == 20240601
        assert df["date_int"].iloc[1] == 20240715

    def test_actual_winner_always_one(self, tmp_path: Path) -> None:
        """Winner column in odds file always corresponds to actual_winner=1."""
        raw = self._make_raw_odds_df()
        fake_path = tmp_path / "atp_2024.xlsx"
        with patch("src.data.odds_loader.pd.read_excel", return_value=raw):
            df = _parse_one(fake_path)
        assert (df["actual_winner"] == 1).all()

    def test_rows_with_missing_pinnacle_odds_dropped(self, tmp_path: Path) -> None:
        """Rows without PSW/PSL must be excluded."""
        raw = self._make_raw_odds_df()
        raw.loc[0, "PSW"] = float("nan")
        fake_path = tmp_path / "atp_2024.xlsx"
        with patch("src.data.odds_loader.pd.read_excel", return_value=raw):
            df = _parse_one(fake_path)
        assert len(df) == 1  # only the second row survives

    def test_no_pinnacle_cols_returns_empty(self, tmp_path: Path) -> None:
        """Files without PSW/PSL must return an empty DataFrame."""
        raw = pd.DataFrame({
            "Date": ["01/06/2024"],
            "Winner": ["Djokovic N."],
            "Loser": ["Federer R."],
        })
        fake_path = tmp_path / "atp_2024.xlsx"
        with patch("src.data.odds_loader.pd.read_excel", return_value=raw):
            df = _parse_one(fake_path)
        assert df.empty


# ---------------------------------------------------------------------------
# Market-informed ensemble: blend_with_market
# ---------------------------------------------------------------------------

def _matched_row(p_model: float, p1_odds: float, p2_odds: float) -> pd.DataFrame:
    """Minimal matched DataFrame row for blend_with_market tests."""
    return pd.DataFrame([{
        "p1_win_prob": p_model,
        "p1_odds": p1_odds,
        "p2_odds": p2_odds,
    }])


class TestBlendWithMarket:
    """Behavioral tests for blend_with_market."""

    def test_alpha_one_leaves_prob_unchanged(self) -> None:
        """alpha=1 (pure model) must return p1_win_prob unchanged."""
        df = _matched_row(p_model=0.70, p1_odds=1.80, p2_odds=2.20)
        result = blend_with_market(df, alpha=1.0)
        assert result.iloc[0]["p1_win_prob"] == pytest.approx(0.70, abs=1e-6)

    def test_alpha_zero_gives_fair_market_prob(self) -> None:
        """alpha=0 (pure market) must return the de-vigged implied probability.

        With symmetric odds (1.91 / 1.91), fair prob = 0.5.
        """
        df = _matched_row(p_model=0.80, p1_odds=1.91, p2_odds=1.91)
        result = blend_with_market(df, alpha=0.0)
        assert result.iloc[0]["p1_win_prob"] == pytest.approx(0.5, abs=1e-4)

    def test_alpha_half_averages_model_and_market(self) -> None:
        """alpha=0.5 must blend 50/50 between model and de-vigged market."""
        # p_market: 1/1.80 = 0.5556, 1/2.20 = 0.4545, total=1.0101
        # fair_p1 = 0.5556 / 1.0101 ≈ 0.55
        df = _matched_row(p_model=0.75, p1_odds=1.80, p2_odds=2.20)
        result = blend_with_market(df, alpha=0.5)
        implied_p1 = (1.0 / 1.80)
        implied_p2 = (1.0 / 2.20)
        fair_p1 = implied_p1 / (implied_p1 + implied_p2)
        expected = 0.5 * 0.75 + 0.5 * fair_p1
        assert result.iloc[0]["p1_win_prob"] == pytest.approx(expected, abs=1e-4)

    def test_result_clipped_to_open_interval(self) -> None:
        """Blended probabilities must never be exactly 0 or 1."""
        df = _matched_row(p_model=1.0, p1_odds=1.01, p2_odds=50.0)
        result = blend_with_market(df, alpha=1.0)
        p = result.iloc[0]["p1_win_prob"]
        assert 0.0 < p < 1.0

    def test_output_does_not_mutate_input(self) -> None:
        """blend_with_market must not modify the input DataFrame."""
        df = _matched_row(p_model=0.70, p1_odds=1.80, p2_odds=2.20)
        original = df["p1_win_prob"].iloc[0]
        _ = blend_with_market(df, alpha=0.5)
        assert df["p1_win_prob"].iloc[0] == original

    def test_invalid_alpha_raises(self) -> None:
        """alpha outside [0, 1] must raise ValueError."""
        df = _matched_row(p_model=0.70, p1_odds=1.80, p2_odds=2.20)
        with pytest.raises(ValueError, match="alpha"):
            blend_with_market(df, alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            blend_with_market(df, alpha=-0.1)

    def test_vectorised_multiple_rows(self) -> None:
        """blend_with_market must handle multiple rows correctly."""
        df = pd.DataFrame([
            {"p1_win_prob": 0.60, "p1_odds": 1.80, "p2_odds": 2.20},
            {"p1_win_prob": 0.40, "p1_odds": 2.50, "p2_odds": 1.60},
        ])
        result = blend_with_market(df, alpha=0.5)
        assert len(result) == 2
        # Both rows must be in valid probability range
        assert (result["p1_win_prob"] > 0).all()
        assert (result["p1_win_prob"] < 1).all()
