"""Unit tests for SklearnPredictor and classifier registry."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.classifiers import (
    CLASSIFIER_REGISTRY,
    SklearnPredictor,
    make_classifier,
    make_naive_bayes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def binary_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4))
    y = (rng.random(80) > 0.5).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# SklearnPredictor — core interface
# ---------------------------------------------------------------------------

class TestSklearnPredictor:
    def test_fit_and_predict_proba_shape(self, binary_data):
        X, y = binary_data
        pred = make_naive_bayes()
        pred.fit(X[:60], y[:60])
        probs = pred.predict_proba(X[60:])
        assert probs.shape == (20, 2)

    def test_predict_proba_rows_sum_to_one(self, binary_data):
        X, y = binary_data
        pred = make_naive_bayes()
        pred.fit(X, y)
        probs = pred.predict_proba(X)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)

    def test_col0_is_p_class0(self, binary_data):
        """Column convention: col 0 = P(class 0) = P(player1 wins)."""
        X, y = binary_data
        pred = make_naive_bayes()
        pred.fit(X, y)
        probs = pred.predict_proba(X)
        assert (probs[:, 0] >= 0).all() and (probs[:, 0] <= 1).all()
        assert (probs[:, 1] >= 0).all() and (probs[:, 1] <= 1).all()
        # col 0 + col 1 == 1 confirms they represent complementary classes
        assert np.allclose(probs[:, 0] + probs[:, 1], 1.0, atol=1e-9)

    def test_predict_proba_without_predict_proba_raises(self):
        from sklearn.svm import SVC
        # SVC without probability=True has no predict_proba
        pred = SklearnPredictor(SVC(probability=False), name="NoProb")
        with pytest.raises(ValueError, match="predict_proba"):
            pred.predict_proba(np.zeros((3, 4)))

    def test_save_and_load_roundtrip(self, tmp_path, binary_data):
        X, y = binary_data
        pred = make_naive_bayes()
        pred.fit(X, y)
        path = str(tmp_path / "model.joblib")
        pred.save(path)
        loaded = SklearnPredictor.load(path)
        original_probs = pred.predict_proba(X[:5])
        loaded_probs = loaded.predict_proba(X[:5])
        assert np.allclose(original_probs, loaded_probs, atol=1e-9)


# ---------------------------------------------------------------------------
# make_classifier and CLASSIFIER_REGISTRY
# ---------------------------------------------------------------------------

class TestClassifierRegistry:
    def test_known_name_returns_sklearn_predictor(self):
        pred = make_classifier("naive_bayes")
        assert isinstance(pred, SklearnPredictor)

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            make_classifier("nonexistent_model")

    def test_all_registry_classifiers_produce_valid_proba(self, binary_data):
        X, y = binary_data
        # Only test fast classifiers — skip SVM, RandomForest (slow)
        fast = {"naive_bayes", "adaboost", "decision_tree", "knn", "qda"}
        for name in fast:
            pred = make_classifier(name)
            pred.fit(X, y)
            probs = pred.predict_proba(X)
            assert probs.shape == (len(X), 2), f"{name}: wrong shape"
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6), f"{name}: rows don't sum to 1"

    def test_registry_has_expected_keys(self):
        expected = {"naive_bayes", "adaboost", "linear_svm", "decision_tree",
                    "random_forest", "mlp_sklearn", "qda", "knn"}
        assert expected == set(CLASSIFIER_REGISTRY.keys())
