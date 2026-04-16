"""Sklearn classifiers wrapped to implement BasePredictor.

Provides a generic ``SklearnPredictor`` that wraps any sklearn estimator
that supports ``predict_proba``. Also exposes named factory functions for the
eight classifiers used in the original codebase.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import joblib
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.models.base_predictor import BasePredictor

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

logger = logging.getLogger(__name__)


class SklearnPredictor(BasePredictor):
    """Wraps any sklearn classifier that supports ``predict_proba``.

    Parameters
    ----------
    clf : ClassifierMixin
        An sklearn estimator. Must implement ``predict_proba`` or
        ``decision_function`` (SVC with ``probability=True``).
    name : str
        Human-readable name used for logging.
    """

    def __init__(self, clf: "ClassifierMixin", name: str = "sklearn") -> None:
        self._clf = clf
        self.name = name

    # ------------------------------------------------------------------
    # BasePredictor interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier to training data.

        Parameters
        ----------
        X : (n_samples, n_features) normalized feature matrix.
        y : (n_samples,) binary labels — 0 or 1.
        """
        self._clf.fit(X, y)
        logger.info("Trained %s", self.name)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities.

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Column 0 → P(player 2 wins), Column 1 → P(player 1 wins).
        """
        if not hasattr(self._clf, "predict_proba"):
            raise ValueError(
                f"{self.name} does not support predict_proba. "
                "For SVC, instantiate with probability=True."
            )
        return self._clf.predict_proba(X)

    def save(self, path: str) -> None:
        """Persist the fitted classifier via joblib."""
        joblib.dump(self._clf, path)
        logger.info("Saved %s to %s", self.name, path)

    @classmethod
    def load(cls, path: str) -> SklearnPredictor:
        """Load a joblib-serialised sklearn classifier.

        Parameters
        ----------
        path : str
            Path written by ``save()``.
        """
        clf = joblib.load(path)
        name = type(clf).__name__
        logger.info("Loaded %s from %s", name, path)
        return cls(clf=clf, name=name)


# ------------------------------------------------------------------
# Named factory functions — mirror the original classifier.py choices
# ------------------------------------------------------------------

def make_naive_bayes() -> SklearnPredictor:
    """GaussianNB — 93.0% test accuracy in original experiments."""
    return SklearnPredictor(GaussianNB(), name="NaiveBayes")


def make_adaboost() -> SklearnPredictor:
    """AdaBoost with 100 estimators — 92.6% test accuracy."""
    return SklearnPredictor(AdaBoostClassifier(n_estimators=100), name="AdaBoost")


def make_linear_svm() -> SklearnPredictor:
    """Linear SVC — 92.7% test accuracy.

    ``probability=True`` enables predict_proba via Platt scaling.
    """
    return SklearnPredictor(
        SVC(kernel="linear", C=0.025, probability=True), name="LinearSVM"
    )


def make_decision_tree() -> SklearnPredictor:
    return SklearnPredictor(DecisionTreeClassifier(max_depth=100), name="DecisionTree")


def make_random_forest() -> SklearnPredictor:
    return SklearnPredictor(
        RandomForestClassifier(n_estimators=1000), name="RandomForest"
    )


def make_mlp_sklearn() -> SklearnPredictor:
    return SklearnPredictor(MLPClassifier(alpha=1, max_iter=1000), name="MLPClassifier")


def make_qda() -> SklearnPredictor:
    return SklearnPredictor(QuadraticDiscriminantAnalysis(), name="QDA")


def make_knn() -> SklearnPredictor:
    return SklearnPredictor(KNeighborsClassifier(n_neighbors=3), name="KNN")


# Map name → factory for config-driven selection
CLASSIFIER_REGISTRY: dict[str, "SklearnPredictor"] = {
    "naive_bayes": make_naive_bayes,
    "adaboost": make_adaboost,
    "linear_svm": make_linear_svm,
    "decision_tree": make_decision_tree,
    "random_forest": make_random_forest,
    "mlp_sklearn": make_mlp_sklearn,
    "qda": make_qda,
    "knn": make_knn,
}


def make_classifier(name: str) -> SklearnPredictor:
    """Instantiate a classifier by name from the registry.

    Parameters
    ----------
    name : str
        One of the keys in ``CLASSIFIER_REGISTRY``.

    Raises
    ------
    ValueError
        If ``name`` is not in the registry.
    """
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{name}'. Available: {list(CLASSIFIER_REGISTRY)}"
        )
    return CLASSIFIER_REGISTRY[name]()
