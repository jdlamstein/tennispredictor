"""XGBoost match-outcome predictor implementing BasePredictor.

XGBoost often outperforms deep learning on tabular data with < 1M rows and
provides native feature importance, making it easy to understand which
features drive predictions.

Reference
---------
Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System."
KDD 2016. https://arxiv.org/abs/1603.02754
"""

from __future__ import annotations

import logging

import numpy as np
from xgboost import XGBClassifier

from src.models.base_predictor import BasePredictor

logger = logging.getLogger(__name__)

# Default hyperparameters — tuned conservatively for tabular sports data
_DEFAULTS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)


class XGBoostPredictor(BasePredictor):
    """XGBoost binary classifier wrapped as a ``BasePredictor``.

    Parameters
    ----------
    **kwargs
        Forwarded to ``XGBClassifier``. Any key overrides the defaults above.

    Examples
    --------
    >>> pred = XGBoostPredictor(n_estimators=300, max_depth=5)
    >>> pred.fit(X_train, y_train)
    >>> probs = pred.predict_proba(X_test)   # shape (n, 2)
    """

    def __init__(self, **kwargs) -> None:
        params = {**_DEFAULTS, **kwargs}
        self._clf = XGBClassifier(**params)

    # ------------------------------------------------------------------
    # BasePredictor interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Train the XGBoost model.

        Parameters
        ----------
        X, y : training features and binary labels (0 or 1).
        X_val, y_val : optional validation set for early stopping.
            If supplied, training stops when val logloss stops improving.
        """
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._clf.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
        logger.info(
            "XGBoost trained — best iteration: %s",
            getattr(self._clf, "best_iteration", "N/A"),
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities.

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Column 0 → P(player 1 wins), Column 1 → P(player 2 wins).
            Follows sklearn convention: column k = P(class k), where class 0
            = game_winner 1 = player 1 wins.
        """
        return self._clf.predict_proba(X)

    def save(self, path: str) -> None:
        """Save model to a JSON file (XGBoost native format)."""
        self._clf.save_model(path)
        logger.info("Saved XGBoost model to %s", path)

    @classmethod
    def load(cls, path: str) -> XGBoostPredictor:
        """Load an XGBoost model from a JSON file.

        Parameters
        ----------
        path : str
            Path written by ``save()``.
        """
        predictor = cls()
        predictor._clf.load_model(path)
        logger.info("Loaded XGBoost model from %s", path)
        return predictor

    # ------------------------------------------------------------------
    # XGBoost-specific extras
    # ------------------------------------------------------------------

    def feature_importances(
        self, feature_names: list[str] | None = None
    ) -> dict[str, float]:
        """Return feature importances sorted descending.

        Parameters
        ----------
        feature_names : list[str] or None
            Names corresponding to columns of X. If None, uses ``f0``, ``f1``, …

        Returns
        -------
        dict[str, float]
            ``{feature_name: importance_score}`` sorted by importance descending.
        """
        scores = self._clf.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(scores))]
        ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return dict(ranked)
