"""Abstract base class all predictors must implement.

Every model — MLP, sklearn classifier, XGBoost, ensemble — wraps itself in a class
that satisfies this interface. The evaluation pipeline, backtester, and paper-trader
depend only on this ABC, never on a concrete implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePredictor(ABC):
    """Common interface for all match-outcome predictors.

    Parameters
    ----------
    None — concrete subclasses define their own __init__.

    Notes
    -----
    ``predict_proba`` must return calibrated probabilities, not raw logits.
    Apply temperature scaling or Platt scaling before exposing a model through
    this interface if its raw outputs are not probability-calibrated.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on features X and binary labels y.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Normalized feature matrix.
        y : np.ndarray, shape (n_samples,)
            Binary labels — 0 if player 2 wins, 1 if player 1 wins.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return win-probability estimates.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Normalized feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Column 0 → P(player 2 wins), Column 1 → P(player 1 wins).
            Each row sums to 1.0.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to ``path``.

        Parameters
        ----------
        path : str
            File path. Extension convention is up to the subclass
            (.ckpt for Lightning, .joblib for sklearn, .json for XGBoost).
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> BasePredictor:
        """Load a previously saved model from ``path``.

        Parameters
        ----------
        path : str
            Same path passed to ``save``.

        Returns
        -------
        BasePredictor
            A ready-to-use predictor instance.
        """
