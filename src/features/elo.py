"""ELO rating computation for ATP tennis matches.

Extracted from preprocessing/pipeline.py and extended with surface-specific
ratings (hard / clay / grass / carpet), which are documented to improve
prediction accuracy by ~1–2pp over a single global ELO.

Reference
---------
Kovalchik, S. (2016). "Searching for the GOAT of tennis win prediction."
Journal of Quantitative Analysis in Sports, 12(3), 127–138.

Design
------
All functions take a DataFrame and return a new DataFrame — no mutation,
no CSV side-effects. The caller is responsible for saving results.

Surfaces
--------
The ATP dataset encodes surface as integers after clean_data.py runs:
    0 → Hard, 1 → Clay, 2 → Grass, 3 → Carpet (rare, mostly pre-2009)

The surface map below translates these integers to the column suffix used
in the output (e.g., ``player1_elo_hard``).
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Initial ELO rating for every new player
_INITIAL_ELO: Final[float] = 1500.0

# K-factor thresholds (mirrors the original pipeline.py implementation)
_K_EXPERIENCED: Final[int] = 30      # games played to be considered experienced
_K_ELITE_RATING: Final[float] = 2400.0

# Integer surface codes used in atp_database.csv after encoding
_SURFACE_MAP: Final[dict[int, str]] = {
    0: "hard",
    1: "clay",
    2: "grass",
    3: "carpet",
}


def _k_factor(rating: float, games_played: int) -> float:
    """Return the K-factor for a player at their current rating and experience."""
    if games_played < _K_EXPERIENCED and rating < _K_ELITE_RATING:
        return 40.0
    if rating < _K_ELITE_RATING:
        return 20.0
    return 10.0


def _update_elo(rating_a: float, rating_b: float, a_won: bool, games_a: int) -> float:
    """Return the new ELO rating for player A after one match.

    Parameters
    ----------
    rating_a, rating_b : float — current ratings.
    a_won : bool — True if player A won.
    games_a : int — total games player A has played (determines K-factor).

    Returns
    -------
    float — updated rating for player A.
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    k = _k_factor(rating_a, games_a)
    score_a = 1.0 if a_won else 0.0
    return rating_a + k * (score_a - expected_a)


def add_global_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Add global (surface-agnostic) ELO ratings to each match row.

    This is a refactored, DataFrame-native version of the original
    ``preprocessing.pipeline.Elo.populate_elo`` — same algorithm, no CSV I/O.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``player1_id``, ``player2_id``, ``game_winner`` (1 or 2),
        sorted chronologically by ``tourney_date``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with added columns ``player1_elo`` and ``player2_elo``.
        Values reflect each player's rating *before* the match (pre-match ELO).
    """
    df = df.copy()
    ratings: dict[int, float] = {}   # player_id → current ELO
    games: dict[int, int] = {}        # player_id → games played

    p1_elos: list[float] = []
    p2_elos: list[float] = []

    for _, row in df.iterrows():
        id1, id2 = int(row["player1_id"]), int(row["player2_id"])
        winner = int(row["game_winner"])   # 1 = player1 won, 2 = player2 won

        r1 = ratings.get(id1, _INITIAL_ELO)
        r2 = ratings.get(id2, _INITIAL_ELO)
        g1 = games.get(id1, 0)
        g2 = games.get(id2, 0)

        # Record pre-match ratings
        p1_elos.append(r1)
        p2_elos.append(r2)

        # Update ratings
        ratings[id1] = _update_elo(r1, r2, a_won=(winner == 1), games_a=g1)
        ratings[id2] = _update_elo(r2, r1, a_won=(winner == 2), games_a=g2)
        games[id1] = g1 + 1
        games[id2] = g2 + 1

    df["player1_elo"] = p1_elos
    df["player2_elo"] = p2_elos
    return df


def add_surface_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Add surface-specific ELO ratings for hard, clay, grass, and carpet.

    Each player maintains four separate ELO pools — one per surface. A player
    who dominates on clay but struggles on grass will have diverging ratings
    that capture their surface specialisation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``player1_id``, ``player2_id``, ``game_winner``,
        ``surface`` (integer-encoded: 0=Hard, 1=Clay, 2=Grass, 3=Carpet).
        Must be sorted chronologically.

    Returns
    -------
    pd.DataFrame
        New DataFrame with eight added columns:
        ``player{1,2}_elo_{hard,clay,grass,carpet}``.
        Rows with an unrecognised surface code get ``NaN`` for all four pairs.
    """
    df = df.copy()

    surfaces = list(_SURFACE_MAP.values())      # hard, clay, grass, carpet
    # Per-surface rating dictionaries: surface → {player_id → elo}
    surf_ratings: dict[str, dict[int, float]] = {s: {} for s in surfaces}
    surf_games: dict[str, dict[int, int]] = {s: {} for s in surfaces}

    # Pre-allocate output columns
    for s in surfaces:
        df[f"player1_elo_{s}"] = np.nan
        df[f"player2_elo_{s}"] = np.nan

    for idx, row in df.iterrows():
        surface_code = int(row["surface"]) if not pd.isna(row["surface"]) else -1
        surface_name = _SURFACE_MAP.get(surface_code)

        id1, id2 = int(row["player1_id"]), int(row["player2_id"])
        winner = int(row["game_winner"])

        if surface_name is None:
            continue  # leave NaN for unknown surfaces

        r1 = surf_ratings[surface_name].get(id1, _INITIAL_ELO)
        r2 = surf_ratings[surface_name].get(id2, _INITIAL_ELO)
        g1 = surf_games[surface_name].get(id1, 0)
        g2 = surf_games[surface_name].get(id2, 0)

        df.at[idx, f"player1_elo_{surface_name}"] = r1
        df.at[idx, f"player2_elo_{surface_name}"] = r2

        surf_ratings[surface_name][id1] = _update_elo(r1, r2, a_won=(winner == 1), games_a=g1)
        surf_ratings[surface_name][id2] = _update_elo(r2, r1, a_won=(winner == 2), games_a=g2)
        surf_games[surface_name][id1] = g1 + 1
        surf_games[surface_name][id2] = g2 + 1

    return df


def add_all_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: add both global and surface-specific ELO.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted chronologically; requires ``player1_id``, ``player2_id``,
        ``game_winner``, ``surface``.

    Returns
    -------
    pd.DataFrame
        Input with ten added columns: ``player{1,2}_elo`` plus
        ``player{1,2}_elo_{hard,clay,grass,carpet}``.
    """
    df = add_global_elo(df)
    df = add_surface_elo(df)
    logger.info(
        "Added global + surface-specific ELO to %d rows (%d unique players)",
        len(df),
        df["player1_id"].nunique() + df["player2_id"].nunique(),
    )
    return df
