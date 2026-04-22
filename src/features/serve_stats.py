"""Serve-stat EMA features for the ATP enriched database.

Replays match history chronologically and computes per-player exponential
moving averages (EMA) of serve statistics. The pre-match value is written
to the row *before* the match outcome updates the player's state — no
look-ahead bias.

New columns added (14 total, 7 per player):
    player{1,2}_ema_first_serve_pct       # 1stIn / svpt
    player{1,2}_ema_first_serve_win_pct   # 1stWon / 1stIn
    player{1,2}_ema_second_serve_win_pct  # 2ndWon / (svpt - 1stIn)
    player{1,2}_ema_bp_save_pct           # bpSaved / bpFaced
    player{1,2}_ema_ace_rate              # ace / svpt
    player{1,2}_ema_df_rate               # df / svpt
    player{1,2}_ema_serve_games           # matches with valid serve data

EMA alpha = 0.15 (≈ 7-match half-life). Prior values are sensible ATP
tour averages, so early-career estimates are reasonable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ALPHA = 0.15  # EMA decay — matches feature_store.py

_SERVE_COLS = [
    "ema_first_serve_pct",
    "ema_first_serve_win_pct",
    "ema_second_serve_win_pct",
    "ema_bp_save_pct",
    "ema_ace_rate",
    "ema_df_rate",
    "ema_serve_games",
]


@dataclass
class _PlayerServeState:
    first_serve_pct: float = 0.60
    first_serve_win_pct: float = 0.72
    second_serve_win_pct: float = 0.50
    bp_save_pct: float = 0.65
    ace_rate: float = 0.07
    df_rate: float = 0.04
    serve_games: int = 0


def _update(ps: _PlayerServeState, svpt: float, first_in: float, first_won: float,
            second_won: float, bp_saved: float, bp_faced: float,
            ace: float, df: float) -> None:
    if svpt <= 0:
        return
    a = _ALPHA
    second_in = svpt - first_in
    ps.first_serve_pct = a * (first_in / svpt) + (1 - a) * ps.first_serve_pct
    ps.first_serve_win_pct = (
        a * (first_won / first_in if first_in > 0 else ps.first_serve_win_pct)
        + (1 - a) * ps.first_serve_win_pct
    )
    ps.second_serve_win_pct = (
        a * (second_won / second_in if second_in > 0 else ps.second_serve_win_pct)
        + (1 - a) * ps.second_serve_win_pct
    )
    ps.bp_save_pct = (
        a * (bp_saved / bp_faced if bp_faced > 0 else ps.bp_save_pct)
        + (1 - a) * ps.bp_save_pct
    )
    ps.ace_rate = a * (ace / svpt) + (1 - a) * ps.ace_rate
    ps.df_rate = a * (df / svpt) + (1 - a) * ps.df_rate
    ps.serve_games += 1


def _safe_float(val) -> float:
    try:
        f = float(val)
        return f if not np.isnan(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def add_serve_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Append 14 pre-match serve-stat EMA columns to *df*.

    Input must be sorted chronologically (by ``tourney_date``).
    Returns a new DataFrame — input is not modified.
    """
    if not {"player1_id", "player2_id", "player1_svpt", "player2_svpt"}.issubset(df.columns):
        logger.warning(
            "Serve stat columns not found in DataFrame — skipping add_serve_stats."
        )
        return df

    df = df.copy()

    # Pre-allocate output arrays
    n = len(df)
    out: dict[str, np.ndarray] = {
        f"player{p}_{c}": np.zeros(n, dtype=float)
        for p in (1, 2)
        for c in _SERVE_COLS
    }

    states: dict[int, _PlayerServeState] = {}

    for i, row in df.iterrows():
        id1 = int(_safe_float(row["player1_id"]))
        id2 = int(_safe_float(row["player2_id"]))

        if id1 not in states:
            states[id1] = _PlayerServeState()
        if id2 not in states:
            states[id2] = _PlayerServeState()

        ps1 = states[id1]
        ps2 = states[id2]

        # Write PRE-MATCH values
        for prefix, ps in [("player1", ps1), ("player2", ps2)]:
            out[f"{prefix}_ema_first_serve_pct"][i] = ps.first_serve_pct
            out[f"{prefix}_ema_first_serve_win_pct"][i] = ps.first_serve_win_pct
            out[f"{prefix}_ema_second_serve_win_pct"][i] = ps.second_serve_win_pct
            out[f"{prefix}_ema_bp_save_pct"][i] = ps.bp_save_pct
            out[f"{prefix}_ema_ace_rate"][i] = ps.ace_rate
            out[f"{prefix}_ema_df_rate"][i] = ps.df_rate
            out[f"{prefix}_ema_serve_games"][i] = float(ps.serve_games)

        # Update state post-match
        for prefix, ps in [("player1", ps1), ("player2", ps2)]:
            _update(
                ps,
                svpt=_safe_float(row.get(f"{prefix}_svpt")),
                first_in=_safe_float(row.get(f"{prefix}_1stIn")),
                first_won=_safe_float(row.get(f"{prefix}_1stWon")),
                second_won=_safe_float(row.get(f"{prefix}_2ndWon")),
                bp_saved=_safe_float(row.get(f"{prefix}_bpSaved")),
                bp_faced=_safe_float(row.get(f"{prefix}_bpFaced")),
                ace=_safe_float(row.get(f"{prefix}_ace")),
                df=_safe_float(row.get(f"{prefix}_df")),
            )

    for col, arr in out.items():
        df[col] = arr

    new_cols = [f"player{p}_{c}" for p in (1, 2) for c in _SERVE_COLS]
    logger.info("Added %d serve-stat EMA columns.", len(new_cols))
    return df
