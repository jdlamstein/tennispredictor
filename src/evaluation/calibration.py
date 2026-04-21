"""Probability calibration for match-outcome predictors.

Raw neural network logits are often overconfident — the model may output 90%
confidence on bets that win only 75% of the time. Calibration corrects this so
that EV calculations and Kelly sizing are based on accurate probabilities.

Method: Temperature Scaling (Guo et al., 2017)
    Divide logits by a learned scalar T before the softmax. T > 1 softens the
    distribution (reduces confidence); T < 1 sharpens it.

Reference
---------
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
"On Calibration of Modern Neural Networks." ICML 2017.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.models.base_predictor import BasePredictor

logger = logging.getLogger(__name__)


def blend_with_market(matched: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Blend model probability with de-vigged bookmaker implied probability.

    p_final = alpha * p_model + (1 - alpha) * p_market

    Parameters
    ----------
    matched : pd.DataFrame
        Output of ``match_odds()`` — must contain ``p1_win_prob``, ``p1_odds``,
        ``p2_odds``.
    alpha : float
        Model weight in [0, 1].
        - 1.0 → pure model (no change to p1_win_prob)
        - 0.0 → pure de-vigged market probability
        - 0.3–0.7 → typical blend range

    Returns
    -------
    pd.DataFrame
        New DataFrame with ``p1_win_prob`` replaced by the blended probability.
        Input is not mutated.

    Raises
    ------
    ValueError
        If ``alpha`` is outside [0, 1].
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    df = matched.copy()

    p_model = df["p1_win_prob"].values

    if alpha < 1.0:
        implied_p1 = 1.0 / df["p1_odds"].values
        implied_p2 = 1.0 / df["p2_odds"].values
        fair_p1 = implied_p1 / (implied_p1 + implied_p2)
        p_model = alpha * p_model + (1.0 - alpha) * fair_p1

    df["p1_win_prob"] = np.clip(p_model, 1e-6, 1.0 - 1e-6)
    return df


class TemperatureScaling:
    """Learns a single scalar T that minimises NLL on held-out probabilities.

    Parameters
    ----------
    init_temperature : float
        Starting value for the temperature search (default 1.0 = no change).

    Attributes
    ----------
    temperature_ : float | None
        The fitted temperature. Set after calling ``fit``.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        self.init_temperature = init_temperature
        self.temperature_: Optional[float] = None

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "TemperatureScaling":
        """Find the temperature that minimises NLL on validation data.

        Parameters
        ----------
        probs : np.ndarray, shape (n, 2)
            Raw (uncalibrated) softmax probabilities from the model.
            Column 0 is P(player 1 wins) — sklearn convention: col k = P(class k),
            class 0 = game_winner 1 = player1 wins. Do NOT pass col 1 as p1_win.
        outcomes : np.ndarray, shape (n,)
            Binary actual outcomes — 1 if player 1 won, 0 otherwise.

        Returns
        -------
        self
        """
        if probs.ndim != 2 or probs.shape[1] != 2:
            raise ValueError(f"probs must have shape (n, 2), got {probs.shape}")
        if len(outcomes) != len(probs):
            raise ValueError(
                f"probs and outcomes length mismatch: {len(probs)} vs {len(outcomes)}"
            )

        def nll(t: float) -> float:
            """Negative log-likelihood after applying temperature T."""
            if t <= 0:
                return float("inf")
            # probs[:,0] = P(player1 wins) per sklearn class-index convention
            log_p1 = np.log(np.clip(probs[:, 0], 1e-12, 1.0))
            log_p2 = np.log(np.clip(probs[:, 1], 1e-12, 1.0))
            scaled_log_p1 = log_p1 / t
            scaled_log_p2 = log_p2 / t
            # Renormalise
            log_sum = np.logaddexp(scaled_log_p1, scaled_log_p2)
            cal_p1 = np.exp(scaled_log_p1 - log_sum)
            cal_p1 = np.clip(cal_p1, 1e-12, 1.0 - 1e-12)
            loss = -np.mean(
                outcomes * np.log(cal_p1) + (1.0 - outcomes) * np.log(1.0 - cal_p1)
            )
            return float(loss)

        result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.temperature_ = float(result.x)
        logger.info("Temperature scaling fitted: T = %.4f", self.temperature_)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply fitted temperature to raw probabilities.

        Parameters
        ----------
        probs : np.ndarray, shape (n, 2)
            Raw softmax probabilities.

        Returns
        -------
        np.ndarray, shape (n, 2)
            Calibrated probabilities. Rows still sum to 1.0.
        """
        if self.temperature_ is None:
            raise RuntimeError("Call fit() before transform().")

        t = self.temperature_
        log_p1 = np.log(np.clip(probs[:, 0], 1e-12, 1.0)) / t
        log_p2 = np.log(np.clip(probs[:, 1], 1e-12, 1.0)) / t
        log_sum = np.logaddexp(log_p1, log_p2)
        cal_p1 = np.exp(log_p1 - log_sum)
        cal_p2 = 1.0 - cal_p1
        return np.stack([cal_p1, cal_p2], axis=1)


class CalibratedPredictor(BasePredictor):
    """Wraps any ``BasePredictor`` and applies temperature scaling at inference.

    Parameters
    ----------
    base : BasePredictor
        Already-fitted predictor whose raw outputs need calibration.
    """

    def __init__(self, base: BasePredictor) -> None:
        self._base = base
        self._calibrator = TemperatureScaling()
        self._fitted = False

    def calibrate(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "CalibratedPredictor":
        """Learn temperature on validation data.

        Parameters
        ----------
        X_val : (n_val, n_features) — validation feature matrix.
        y_val : (n_val,) — binary validation labels.

        Returns
        -------
        self
        """
        raw_probs = self._base.predict_proba(X_val)
        self._calibrator.fit(raw_probs, y_val)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # BasePredictor interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Delegate to the underlying predictor.

        Call ``calibrate(X_val, y_val)`` separately after fitting.
        """
        self._base.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities.

        If ``calibrate`` has not been called, returns raw probabilities with a
        warning.
        """
        raw = self._base.predict_proba(X)
        if not self._fitted:
            logger.warning(
                "CalibratedPredictor.predict_proba called before calibrate(); "
                "returning raw probabilities."
            )
            return raw
        return self._calibrator.transform(raw)

    def save(self, path: str) -> None:
        """Save is not yet supported for CalibratedPredictor."""
        raise NotImplementedError(
            "CalibratedPredictor.save is not implemented. "
            "Save the base predictor and temperature separately."
        )

    @classmethod
    def load(cls, path: str) -> "CalibratedPredictor":
        raise NotImplementedError("Use base predictor load + TemperatureScaling.")
