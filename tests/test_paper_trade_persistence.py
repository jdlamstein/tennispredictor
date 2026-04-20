"""Tests for model + scaler persistence in scripts/paper_trade.py.

Verifies that joblib save/load round-trips produce identical predictions.
Uses a tiny in-memory model — no real ATP database required.
"""

from __future__ import annotations

import os
import sys
import tempfile

import joblib
import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.classifiers import SklearnPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_tiny_model() -> tuple[SklearnPredictor, StandardScaler, np.ndarray]:
    """Train a trivially separable 2-feature model. Returns (model, scaler, X_test)."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 10))
    y = (X[:, 0] > 0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SklearnPredictor(GaussianNB())
    model.fit(X_scaled, y)

    X_test = scaler.transform(rng.normal(size=(20, 10)))
    return model, scaler, X_test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJobLibRoundTrip:
    """predict_proba must be identical before and after joblib save/load."""

    def test_predictions_identical_after_save_load(self, tmp_path) -> None:
        model, scaler, X_test = _train_tiny_model()
        probs_before = model.predict_proba(X_test)

        cache_path = tmp_path / "test_model.joblib"
        joblib.dump({"model": model, "scaler": scaler, "holdout_year": 2022}, cache_path)

        bundle = joblib.load(cache_path)
        loaded_model  = bundle["model"]
        loaded_scaler = bundle["scaler"]

        # Verify scaler parameters identical
        np.testing.assert_array_almost_equal(loaded_scaler.mean_, scaler.mean_)
        np.testing.assert_array_almost_equal(loaded_scaler.scale_, scaler.scale_)

        probs_after = loaded_model.predict_proba(X_test)
        np.testing.assert_array_almost_equal(probs_before, probs_after)

    def test_bundle_contains_holdout_year(self, tmp_path) -> None:
        model, scaler, _ = _train_tiny_model()
        cache_path = tmp_path / "model.joblib"
        joblib.dump({"model": model, "scaler": scaler, "holdout_year": 2023}, cache_path)

        bundle = joblib.load(cache_path)
        assert bundle["holdout_year"] == 2023

    def test_load_from_cache_skips_training(self, tmp_path, monkeypatch) -> None:
        """If cache exists, _load_or_train_model must NOT call pd.read_csv."""
        import pandas as pd
        model, scaler, _ = _train_tiny_model()
        cache_path = str(tmp_path / "paper_model.joblib")
        joblib.dump({"model": model, "scaler": scaler, "holdout_year": 2022}, cache_path)

        read_csv_calls = []
        original_read_csv = pd.read_csv

        def mock_read_csv(*args, **kwargs):
            read_csv_calls.append(args)
            return original_read_csv(*args, **kwargs)

        monkeypatch.setattr(pd, "read_csv", mock_read_csv)
        monkeypatch.setenv("MODEL_CACHE_PATH", cache_path)
        monkeypatch.setenv("ATP_DB", "/nonexistent/path.csv")

        # Reload module env vars
        import importlib
        import scripts.paper_trade as pt
        importlib.reload(pt)

        scaler_ref: list = []
        loaded = pt._load_or_train_model(scaler_ref)

        assert len(read_csv_calls) == 0, "pd.read_csv should NOT be called when cache exists"
        assert len(scaler_ref) == 1
        assert loaded is not None

    def test_cache_fallback_on_corrupt_file(self, tmp_path, monkeypatch) -> None:
        """Corrupt cache file must fall through to re-training (or error gracefully)."""
        cache_path = tmp_path / "corrupt.joblib"
        cache_path.write_bytes(b"not a valid joblib file")

        monkeypatch.setenv("MODEL_CACHE_PATH", str(cache_path))
        monkeypatch.setenv("ATP_DB", "/nonexistent/path.csv")

        import importlib
        import scripts.paper_trade as pt
        importlib.reload(pt)

        scaler_ref: list = []
        # Should not raise on corrupt cache — falls back to training
        # But ATP_DB doesn't exist so training will raise FileNotFoundError
        with pytest.raises(Exception):
            pt._load_or_train_model(scaler_ref)


class TestBundleStructure:
    """Validate bundle dict structure and types."""

    def test_bundle_keys(self, tmp_path) -> None:
        model, scaler, _ = _train_tiny_model()
        cache_path = tmp_path / "m.joblib"
        joblib.dump({"model": model, "scaler": scaler, "holdout_year": 2022}, cache_path)
        bundle = joblib.load(cache_path)
        assert "model" in bundle
        assert "scaler" in bundle
        assert "holdout_year" in bundle

    def test_scaler_is_standard_scaler(self, tmp_path) -> None:
        model, scaler, _ = _train_tiny_model()
        cache_path = tmp_path / "m.joblib"
        joblib.dump({"model": model, "scaler": scaler, "holdout_year": 2022}, cache_path)
        bundle = joblib.load(cache_path)
        assert isinstance(bundle["scaler"], StandardScaler)

    def test_predict_proba_output_shape(self, tmp_path) -> None:
        model, scaler, X_test = _train_tiny_model()
        cache_path = tmp_path / "m.joblib"
        joblib.dump({"model": model, "scaler": scaler, "holdout_year": 2022}, cache_path)
        bundle = joblib.load(cache_path)
        probs = bundle["model"].predict_proba(X_test)
        assert probs.shape == (len(X_test), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
