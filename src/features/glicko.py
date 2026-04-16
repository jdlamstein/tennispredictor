"""Glicko-2 rating system for ATP tennis matches.

Glicko-2 extends ELO with two extra per-player quantities:
  - Rating Deviation (RD): uncertainty in the player's rating.
    High RD = fewer recent matches, rating is less reliable.
  - Volatility (σ): measures consistency of performance.
    High σ = player has been performing unexpectedly (up or down).

These quantities improve prediction quality for players with sparse data
(Challengers, newly-turned-pro players) where ELO's uncertainty is implicit.

Reference
---------
Glickman, M. E. (2012). "Example of the Glicko-2 system."
http://www.glicko.net/glicko/glicko2.pdf

Algorithm constants follow the Glickman (2012) example exactly.
"""

from __future__ import annotations

import logging
import math
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Glicko-2 system constants
# ---------------------------------------------------------------------------
_SCALE: Final[float] = 173.7178       # converts between Glicko-1 and Glicko-2 scale

_INIT_R: Final[float] = 1500.0        # initial rating (Glicko-1 scale)
_INIT_RD: Final[float] = 350.0        # initial rating deviation (Glicko-1 scale)
_INIT_SIGMA: Final[float] = 0.06      # initial volatility

_TAU: Final[float] = 0.5              # system constant τ; controls volatility change speed
_EPSILON: Final[float] = 1e-6         # convergence threshold for Illinois algorithm


# ---------------------------------------------------------------------------
# Internal Glicko-2 math (works on μ/φ/σ scale)
# ---------------------------------------------------------------------------

def _g(phi: float) -> float:
    """g(φ) reduction factor."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / math.pi ** 2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """Expected score E(μ, μj, φj)."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _compute_v(mu: float, opponents: list[tuple[float, float]]) -> float:
    """Estimated variance v from a list of (μ_j, φ_j) opponent ratings."""
    v = 0.0
    for mu_j, phi_j in opponents:
        g_j = _g(phi_j)
        e_j = _E(mu, mu_j, phi_j)
        v += g_j ** 2 * e_j * (1.0 - e_j)
    if v == 0.0:
        return float("inf")
    return 1.0 / v


def _compute_delta(mu: float, v: float, opponents: list[tuple[float, float, float]]) -> float:
    """Estimated improvement Δ from a list of (μ_j, φ_j, s_j) results."""
    total = 0.0
    for mu_j, phi_j, s_j in opponents:
        total += _g(phi_j) * (s_j - _E(mu, mu_j, phi_j))
    return v * total


def _update_volatility(phi: float, sigma: float, delta: float, v: float) -> float:
    """Illinois algorithm to compute new volatility σ'.

    Finds σ' such that f(σ') = 0 using a bracketed root-finding approach.
    """
    a = math.log(sigma ** 2)
    d2 = delta ** 2
    phi2 = phi ** 2

    def f(x: float) -> float:
        ex = math.exp(x)
        denom = phi2 + v + ex
        return (ex * (d2 - phi2 - v - ex) / (2.0 * denom ** 2)
                - (x - a) / (_TAU ** 2))

    # Initialise bracket
    b = a - _TAU * math.sqrt(abs(d2 - phi2 - v)) if d2 > phi2 + v else a - _TAU

    fa, fb = f(a), f(b)

    # Illinois method iterations
    for _ in range(100):
        c = a + (a - b) * fa / (fb - fa)
        fc = f(c)
        if fc * fb < 0.0:
            a, fa = b, fb
        else:
            fa /= 2.0
        b, fb = c, fc
        if abs(b - a) < _EPSILON:
            break

    return math.exp(b / 2.0)


def _update_player(
    mu: float,
    phi: float,
    sigma: float,
    opponents: list[tuple[float, float, float]],  # (μ_j, φ_j, score_j)
) -> tuple[float, float, float]:
    """Compute updated (μ', φ', σ') after a batch of matches.

    Parameters
    ----------
    mu, phi, sigma : current player state on Glicko-2 scale.
    opponents : list of (opponent_mu, opponent_phi, score) — score is 1=win, 0=loss, 0.5=draw.

    Returns
    -------
    tuple[float, float, float]
        Updated (mu, phi, sigma).
    """
    if not opponents:
        # No matches: inflate RD to reflect increased uncertainty
        phi_star = math.sqrt(phi ** 2 + sigma ** 2)
        return mu, phi_star, sigma

    # Step 1: compute opponent pairs without scores (for variance)
    opp_no_score = [(mu_j, phi_j) for mu_j, phi_j, _ in opponents]

    v = _compute_v(mu, opp_no_score)
    delta = _compute_delta(mu, v, opponents)

    # Step 2: new volatility
    sigma_new = _update_volatility(phi, sigma, delta, v)

    # Step 3: new pre-rating-period RD
    phi_star = math.sqrt(phi ** 2 + sigma_new ** 2)

    # Step 4: new RD
    phi_new = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)

    # Step 5: new rating
    total = sum(_g(phi_j) * (s_j - _E(mu, mu_j, phi_j))
                for mu_j, phi_j, s_j in opponents)
    mu_new = mu + phi_new ** 2 * total

    return mu_new, phi_new, sigma_new


# ---------------------------------------------------------------------------
# DataFrame-level API
# ---------------------------------------------------------------------------

def add_glicko2(df: pd.DataFrame, rating_period_days: int = 90) -> pd.DataFrame:
    """Add Glicko-2 ratings to each match row.

    Ratings are updated in *rating periods* (batches of matches within a time
    window) rather than after every single match, following Glickman's spec.
    Within a rating period, all updates are computed simultaneously.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``player1_id``, ``player2_id``, ``game_winner`` (1 or 2),
        ``tourney_date`` (YYYYMMDD int). Must be sorted chronologically.
    rating_period_days : int
        Length of each rating period in calendar days. Default 90 (quarterly).

    Returns
    -------
    pd.DataFrame
        New DataFrame with six added columns:
        ``player{1,2}_glicko``, ``player{1,2}_rd``, ``player{1,2}_sigma``.
        Values reflect each player's state *before* the match.
    """
    df = df.copy()

    # Convert tourney_date (YYYYMMDD) to a date index for period grouping
    dates = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    period_start = dates.min()

    # Player state on Glicko-2 (μ, φ, σ) scale
    state_mu: dict[int, float] = {}
    state_phi: dict[int, float] = {}
    state_sigma: dict[int, float] = {}

    def _get(pid: int) -> tuple[float, float, float]:
        mu = state_mu.get(pid, (_INIT_R - 1500.0) / _SCALE)
        phi = state_phi.get(pid, _INIT_RD / _SCALE)
        sigma = state_sigma.get(pid, _INIT_SIGMA)
        return mu, phi, sigma

    # Pre-allocate output columns
    for col in ("player1_glicko", "player2_glicko", "player1_rd", "player2_rd",
                "player1_sigma", "player2_sigma"):
        df[col] = np.nan

    # Group matches into rating periods
    period_idx = ((dates - period_start).dt.days // rating_period_days).values

    for period in range(int(period_idx.max()) + 1):
        mask = period_idx == period
        period_rows = df[mask]

        if period_rows.empty:
            continue

        # Collect opponents for each player in this period (before updating)
        player_opponents: dict[int, list[tuple[float, float, float]]] = {}

        for idx, row in period_rows.iterrows():
            id1, id2 = int(row["player1_id"]), int(row["player2_id"])
            winner = int(row["game_winner"])

            mu1, phi1, _ = _get(id1)
            mu2, phi2, _ = _get(id2)

            # Record pre-match Glicko-2 values in output
            df.at[idx, "player1_glicko"] = mu1 * _SCALE + 1500.0
            df.at[idx, "player2_glicko"] = mu2 * _SCALE + 1500.0
            df.at[idx, "player1_rd"] = phi1 * _SCALE
            df.at[idx, "player2_rd"] = phi2 * _SCALE
            df.at[idx, "player1_sigma"] = state_sigma.get(id1, _INIT_SIGMA)
            df.at[idx, "player2_sigma"] = state_sigma.get(id2, _INIT_SIGMA)

            score1 = 1.0 if winner == 1 else 0.0
            score2 = 1.0 - score1

            player_opponents.setdefault(id1, []).append((mu2, phi2, score1))
            player_opponents.setdefault(id2, []).append((mu1, phi1, score2))

        # Batch-update all players who played in this period
        for pid, opps in player_opponents.items():
            mu, phi, sigma = _get(pid)
            mu_new, phi_new, sigma_new = _update_player(mu, phi, sigma, opps)
            state_mu[pid] = mu_new
            state_phi[pid] = phi_new
            state_sigma[pid] = sigma_new

    logger.info("Added Glicko-2 ratings to %d rows", len(df))
    return df
