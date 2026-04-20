"""Live feature store for paper-trading predictions.

Loads the enriched ATP database, replays the full ELO/Glicko-2 computation,
and caches each player's current ratings + recent form. Given two player names
and tournament context, returns a feature vector ready for model inference.

Design
------
- Built once at startup from the enriched CSV (takes ~30s for 824K rows).
- Immutable after build — call ``FeatureStore.build()`` for fresh data.
- Name matching uses surname (last token) — matches both ATP format
  ("Novak Djokovic") and odds format ("Djokovic N.").

Feature vector column order matches ``scripts/backtest._prepare_features``
output so the same trained XGBoost model can be used without retraining.

Usage
-----
    store = FeatureStore.build(atp_db_path, holdout_year=2022)
    X = store.make_features(
        p1_name="Novak Djokovic",
        p2_name="Carlos Alcaraz",
        surface=0,          # 0=Hard
        tourney_level=2,    # e.g. Grand Slam encoded value
        round_num=7,        # Final
        best_of=5,
        draw_size=128,
        current_date=date.today(),
    )
    probs = model.predict_proba(scaler.transform(X))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ELO constants (mirrors src/features/elo.py)
# ---------------------------------------------------------------------------
_INITIAL_ELO = 1500.0
_K_ELITE = 2100.0
_K_EXPERIENCED = 30


def _k_factor(rating: float, games: int) -> float:
    if games < _K_EXPERIENCED and rating < _K_ELITE:
        return 40.0
    if rating < _K_ELITE:
        return 20.0
    return 10.0


def _update_elo(r_a: float, r_b: float, a_won: bool, games_a: int) -> float:
    expected = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
    k = _k_factor(r_a, games_a)
    return r_a + k * ((1.0 if a_won else 0.0) - expected)


# ---------------------------------------------------------------------------
# Glicko-2 constants (mirrors src/features/glicko.py)
# ---------------------------------------------------------------------------
_SCALE = 173.7178
_INIT_R, _INIT_RD, _INIT_SIGMA = 1500.0, 350.0, 0.06
_TAU = 0.5
_EPS = 1e-6
_PERIOD_DAYS = 90


def _g2(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / math.pi ** 2)


def _E2(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g2(phi_j) * (mu - mu_j)))


_SURFACE_MAP = {0: "hard", 1: "clay", 2: "grass", 3: "carpet"}


# ---------------------------------------------------------------------------
# Per-player state
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    """All features we track per player."""

    player_id: int
    name: str = ""
    hand: float = -10.0
    ht: float = -10.0
    ioc: float = -10.0
    dob: Optional[date] = None          # for age computation

    # Surface ELO (current = post-last-match)
    elo_hard: float = _INITIAL_ELO
    elo_clay: float = _INITIAL_ELO
    elo_grass: float = _INITIAL_ELO
    elo_carpet: float = _INITIAL_ELO
    elo_games: dict = field(default_factory=dict)  # surface → games played

    # Glicko-2 state on Glicko-2 scale
    g2_mu: float = (_INIT_R - 1500.0) / _SCALE
    g2_phi: float = _INIT_RD / _SCALE
    g2_sigma: float = _INIT_SIGMA

    # Form
    winning_streak: int = 0
    losing_streak: int = 0
    weeks_inactive: float = 0.0
    last_match_date: Optional[date] = None
    recent_matches: int = 0             # matches in last 14 days

    # Per opponent H2H: {opp_player_id: wins}
    h2h: dict = field(default_factory=dict)

    def glicko_r(self) -> float:
        """Rating on Glicko-1 scale."""
        return self.g2_mu * _SCALE + 1500.0

    def glicko_rd(self) -> float:
        return self.g2_phi * _SCALE

    def glicko_sigma(self) -> float:
        return self.g2_sigma

    def surface_elo(self, surface_name: str) -> float:
        return getattr(self, f"elo_{surface_name}", _INITIAL_ELO)


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------

class FeatureStore:
    """Replay ATP history and cache current player states.

    Parameters
    ----------
    players : dict[int, PlayerState]
        Keyed by player_id.
    surname_to_id : dict[str, list[int]]
        Surname (lowercase) → list of player_ids (may collide for common surnames).
    built_at : date
        Date the store was built (= last date in training data).
    """

    def __init__(
        self,
        players: dict[int, PlayerState],
        surname_to_id: dict[str, list[int]],
        built_at: date,
    ) -> None:
        self._players = players
        self._sn_map = surname_to_id
        self._built_at = built_at

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        db_path: str,
        holdout_year: Optional[int] = None,
    ) -> "FeatureStore":
        """Replay history and build the feature store.

        Parameters
        ----------
        db_path : str
            Path to enriched ATP database CSV (from ``scripts/enrich_features.py``).
        holdout_year : int | None
            If set, only use matches before this year (avoids leaking holdout
            data into the live feature cache).

        Returns
        -------
        FeatureStore
        """
        logger.info("Building FeatureStore from %s …", db_path)
        df = pd.read_csv(db_path, low_memory=False)
        df = df.sort_values("tourney_date").reset_index(drop=True)

        if holdout_year is not None:
            df = df[df["tourney_date"] // 10000 < holdout_year].copy()

        players: dict[int, PlayerState] = {}
        surname_to_id: dict[str, list[int]] = {}

        # Glicko-2 period tracking
        dates = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
        period_start = dates.min()
        period_idx = ((dates - period_start).dt.days // _PERIOD_DAYS).fillna(-1).astype(int).values

        # Accumulate Glicko-2 period results: player_id → list[(opp_mu, opp_phi, score)]
        g2_period_buf: dict[int, list[tuple[float, float, float]]] = {}
        current_period = -1

        def _flush_glicko() -> None:
            """Apply accumulated Glicko-2 updates from the current period."""
            for pid, results in g2_period_buf.items():
                if pid not in players:
                    continue
                ps = players[pid]
                mu, phi, sigma = ps.g2_mu, ps.g2_phi, ps.g2_sigma
                if not results:
                    # Inactive this period: increase RD
                    phi_new = min(math.sqrt(phi ** 2 + sigma ** 2), _INIT_RD / _SCALE)
                    players[pid].g2_phi = phi_new
                    continue
                # Simplified update (skip full Illinois for speed; approximate)
                v = 0.0
                for mu_j, phi_j, _ in results:
                    g_j = _g2(phi_j)
                    e_j = _E2(mu, mu_j, phi_j)
                    v += g_j ** 2 * e_j * (1.0 - e_j)
                v = 1.0 / v if v > 0 else float("inf")
                delta_sum = sum(
                    _g2(phi_j) * (s_j - _E2(mu, mu_j, phi_j))
                    for mu_j, phi_j, s_j in results
                )
                delta = v * delta_sum
                phi_star = math.sqrt(phi ** 2 + sigma ** 2)
                phi_new = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v) if v < float("inf") else phi_star
                mu_new = mu + phi_new ** 2 * delta_sum
                players[pid].g2_mu = mu_new
                players[pid].g2_phi = phi_new
            g2_period_buf.clear()

        def _get_or_create(pid: int, name: str = "") -> PlayerState:
            if pid not in players:
                players[pid] = PlayerState(player_id=pid, name=name)
                sn = name.split()[-1].lower() if name else ""
                if sn:
                    surname_to_id.setdefault(sn, [])
                    if pid not in surname_to_id[sn]:
                        surname_to_id[sn].append(pid)
            return players[pid]

        last_date = date.min

        for i, row in df.iterrows():
            p = int(period_idx[i])

            # Flush Glicko-2 at period boundary
            if p != current_period:
                if current_period >= 0:
                    _flush_glicko()
                current_period = p

            id1 = int(row["player1_id"])
            id2 = int(row["player2_id"])
            name1 = str(row.get("player1_name", ""))
            name2 = str(row.get("player2_name", ""))
            winner = int(row["game_winner"])
            surface_code = int(row["surface"]) if pd.notna(row.get("surface")) else -1
            surf = _SURFACE_MAP.get(surface_code)

            ps1 = _get_or_create(id1, name1)
            ps2 = _get_or_create(id2, name2)

            # Update names (take most recent)
            if name1:
                ps1.name = name1
            if name2:
                ps2.name = name2

            # Static info (overwrite with latest)
            for attr, col in [("hand", "player1_hand"), ("ht", "player1_ht"), ("ioc", "player1_ioc")]:
                val = row.get(col)
                if pd.notna(val):
                    setattr(ps1, attr, float(val))
            for attr, col in [("hand", "player2_hand"), ("ht", "player2_ht"), ("ioc", "player2_ioc")]:
                val = row.get(col)
                if pd.notna(val):
                    setattr(ps2, attr, float(val))

            # Match date
            try:
                tdate_str = str(int(row["tourney_date"]))
                match_date = date(int(tdate_str[:4]), int(tdate_str[4:6]), int(tdate_str[6:8]))
            except Exception:
                match_date = last_date
            last_date = max(last_date, match_date)

            # Surface ELO (post-match update = current ELO for next match)
            if surf:
                r1 = ps1.surface_elo(surf)
                r2 = ps2.surface_elo(surf)
                g1 = ps1.elo_games.get(surf, 0)
                g2 = ps2.elo_games.get(surf, 0)
                setattr(ps1, f"elo_{surf}", _update_elo(r1, r2, winner == 1, g1))
                setattr(ps2, f"elo_{surf}", _update_elo(r2, r1, winner == 2, g2))
                ps1.elo_games[surf] = g1 + 1
                ps2.elo_games[surf] = g2 + 1

            # Glicko-2 period accumulation
            mu1, phi1, _ = ps1.g2_mu, ps1.g2_phi, ps1.g2_sigma
            mu2, phi2, _ = ps2.g2_mu, ps2.g2_phi, ps2.g2_sigma
            g2_period_buf.setdefault(id1, []).append((mu2, phi2, 1.0 if winner == 1 else 0.0))
            g2_period_buf.setdefault(id2, []).append((mu1, phi1, 1.0 if winner == 2 else 0.0))

            # Form
            for ps, won in [(ps1, winner == 1), (ps2, winner == 2)]:
                if won:
                    ps.winning_streak += 1
                    ps.losing_streak = 0
                else:
                    ps.losing_streak += 1
                    ps.winning_streak = 0
                two_weeks_ago = match_date - timedelta(days=14)
                if ps.last_match_date and ps.last_match_date >= two_weeks_ago:
                    ps.recent_matches += 1
                else:
                    ps.recent_matches = 1
                ps.last_match_date = match_date

            # H2H
            ps1.h2h[id2] = ps1.h2h.get(id2, 0) + (1 if winner == 1 else 0)
            ps2.h2h[id1] = ps2.h2h.get(id1, 0) + (1 if winner == 2 else 0)

        # Final Glicko-2 flush
        _flush_glicko()

        logger.info(
            "FeatureStore built: %d players, last date=%s", len(players), last_date
        )
        return cls(players, surname_to_id, last_date)

    # ------------------------------------------------------------------
    # Name lookup
    # ------------------------------------------------------------------

    def _resolve(self, name: str) -> Optional[PlayerState]:
        """Find PlayerState by name (full name or 'Surname I.' format)."""
        sn = name.split()[0].lower()  # odds format: "Djokovic N." → "djokovic"
        if sn not in self._sn_map:
            # Try last-token (ATP format: "Novak Djokovic" → "djokovic")
            sn = name.split()[-1].lower()
        ids = self._sn_map.get(sn, [])
        if not ids:
            return None
        if len(ids) == 1:
            return self._players[ids[0]]
        # Collision: disambiguate by first initial if available
        initial = name.split()[1][0].upper() if len(name.split()) > 1 else ""
        for pid in ids:
            ps = self._players[pid]
            ps_initial = ps.name.split()[0][0].upper() if ps.name else ""
            if initial and ps_initial == initial:
                return ps
        return self._players[ids[0]]  # fall back to first

    # ------------------------------------------------------------------
    # Feature vector construction
    # ------------------------------------------------------------------

    def make_features(
        self,
        p1_name: str,
        p2_name: str,
        surface: int = 0,
        tourney_level: float = 0.0,
        round_num: float = 1.0,
        best_of: float = 3.0,
        draw_size: float = 32.0,
        current_date: Optional[date] = None,
        p1_seed: float = -10.0,
        p2_seed: float = -10.0,
    ) -> Optional[np.ndarray]:
        """Return a (1, n_features) array for model inference.

        Column order matches ``scripts/backtest._prepare_features`` output
        for the enriched database.

        Returns None if either player is not found in the store.
        """
        ps1 = self._resolve(p1_name)
        ps2 = self._resolve(p2_name)
        if ps1 is None:
            logger.warning("Player not found: %s", p1_name)
            return None
        if ps2 is None:
            logger.warning("Player not found: %s", p2_name)
            return None

        today = current_date or date.today()
        surf_name = _SURFACE_MAP.get(surface, "hard")

        def _age(ps: PlayerState) -> float:
            if ps.dob:
                return (today - ps.dob).days / 365.25
            return -10.0

        def _weeks_inactive(ps: PlayerState) -> float:
            if ps.last_match_date is None:
                return -10.0
            return max(0.0, (today - ps.last_match_date).days / 7.0)

        def _h2h(ps_a: PlayerState, ps_b: PlayerState) -> float:
            return float(ps_a.h2h.get(ps_b.player_id, 0))

        year = today.year
        yday = today.timetuple().tm_yday
        sine_day = math.sin(2 * math.pi * yday / 365.0)
        cosine_day = math.cos(2 * math.pi * yday / 365.0)

        # Feature order matches _prepare_features output for enriched DB.
        # Unknown / unavailable fields filled with -10 (same sentinel as backtest).
        feats = [
            float(surface),                  # surface
            float(draw_size),                 # draw_size
            float(tourney_level),             # tourney_level
            -10.0,                            # match_num (unavailable pre-match)
            float(best_of),                   # best_of
            float(round_num),                 # round
            float(p1_seed),                   # player1_seed
            -10.0,                            # player1_entry
            ps1.hand,                         # player1_hand
            ps1.ht,                           # player1_ht
            ps1.ioc,                          # player1_ioc
            _age(ps1),                        # player1_age
            float(p2_seed),                   # player2_seed
            -10.0,                            # player2_entry
            ps2.hand,                         # player2_hand
            ps2.ht,                           # player2_ht
            ps2.ioc,                          # player2_ioc
            _age(ps2),                        # player2_age
            # Surface ELOs (pre-match for NEXT match = post-match from last match)
            ps1.surface_elo(surf_name),       # player1_elo_{surf}
            ps2.surface_elo(surf_name),       # player2_elo_{surf}
            # Glicko-2
            ps1.glicko_r(),                   # player1_glicko
            ps1.glicko_rd(),                  # player1_rd
            ps1.glicko_sigma(),               # player1_sigma
            ps2.glicko_r(),                   # player2_glicko
            ps2.glicko_rd(),                  # player2_rd
            ps2.glicko_sigma(),               # player2_sigma
            float(year),                      # year
            sine_day,                         # sine_day
            cosine_day,                       # cosine_day
            # Form
            float(ps1.winning_streak),        # player1_winning_streak
            float(ps2.winning_streak),        # player2_winning_streak
            float(ps1.losing_streak),         # player1_losing_streak
            float(ps2.losing_streak),         # player2_losing_streak
            _weeks_inactive(ps1),             # player1_weeks_inactive
            _weeks_inactive(ps2),             # player2_weeks_inactive
            float(ps1.recent_matches),        # player1_last_two_weeks
            float(ps2.recent_matches),        # player2_last_two_weeks
            # H2H
            _h2h(ps1, ps2),                   # player1_v_player2_wins
            _h2h(ps2, ps1),                   # player2_v_player1_wins
            float(year),                      # year_col
        ]
        return np.array(feats, dtype=float).reshape(1, -1)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def player_info(self, name: str) -> Optional[dict]:
        """Return a summary dict for a player (for debugging / display)."""
        ps = self._resolve(name)
        if ps is None:
            return None
        return {
            "name":           ps.name,
            "elo_hard":       round(ps.elo_hard, 1),
            "elo_clay":       round(ps.elo_clay, 1),
            "elo_grass":      round(ps.elo_grass, 1),
            "glicko":         round(ps.glicko_r(), 1),
            "glicko_rd":      round(ps.glicko_rd(), 1),
            "winning_streak": ps.winning_streak,
            "losing_streak":  ps.losing_streak,
            "last_match":     ps.last_match_date,
        }
