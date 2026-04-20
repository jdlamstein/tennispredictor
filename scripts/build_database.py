"""Build atp_database.csv from raw Sackmann ATP match CSVs.

Reads all atp_matches_{year}.csv files from the data directory, converts from
Sackmann winner/loser format to player1/player2 format, and computes form
features (winning streaks, H2H, basic ELO, date features) chronologically.

Run enrich_features.py afterwards to add surface-specific ELO and Glicko-2.

Usage
-----
    poetry run python scripts/build_database.py
    poetry run python scripts/build_database.py --data-dir ~/Data/tennis/tennis_data

Environment variables
---------------------
    ATP_ROOTDIR   Parent directory (default: ~/Data/tennis)
"""

import argparse
import hashlib
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ATP_ROOTDIR = os.path.expanduser(os.environ.get("ATP_ROOTDIR", "~/Data/tennis"))
_DEFAULT_DATA_DIR = os.path.join(ATP_ROOTDIR, "tennis_data")
_DEFAULT_OUTPUT = os.path.join(_DEFAULT_DATA_DIR, "atp_database.csv")

# Basic ELO constants
_ELO_INIT = 1500.0
_ELO_K = 32.0

# Sackmann surface strings → integer codes (matches elo.py: 0=Hard, 1=Clay, 2=Grass, 3=Carpet)
_SURFACE_MAP = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}

# Racket hand: R=0, L=1, unknown/ambidextrous=-10
_HAND_MAP = {"R": 0.0, "L": 1.0, "U": -10.0, "A": -10.0}

# Tourney level ordinal (higher = more prestigious)
_LEVEL_MAP = {"F": 6, "G": 5, "M": 4, "A": 3, "C": 2, "S": 1, "D": 0, "O": 0}

# Round ordinal (higher = later in draw)
_ROUND_MAP = {
    "F": 8, "SF": 7, "QF": 6, "R16": 5, "R32": 4, "R64": 3, "R128": 2,
    "RR": 1, "BR": 1, "Q3": 1, "Q2": 1, "Q1": 1, "ER": 0,
}

def _encode_ioc(code: str | None) -> float:
    """Stable numeric encoding for 3-letter IOC country codes via ASCII hash."""
    if not code or not isinstance(code, str) or len(code) != 3:
        return -10.0
    c = code.upper()
    return float(ord(c[0]) * 676 + ord(c[1]) * 26 + ord(c[2]))


@dataclass
class _PlayerState:
    winning_streak: int = 0
    losing_streak: int = 0
    recent_matches: int = 0
    last_match_date: date | None = None
    elo: float = _ELO_INIT
    h2h: dict = field(default_factory=dict)


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def _k_factor(n_games: int) -> float:
    if n_games < 30:
        return 40.0
    if n_games < 100:
        return 32.0
    return 24.0


def _assign_sides(winner_id: int, loser_id: int, tourney_id: str, match_num: int) -> bool:
    """Return True if winner should be player1. Deterministic via hash."""
    key = f"{tourney_id}_{match_num}_{winner_id}_{loser_id}"
    digest = hashlib.md5(key.encode()).digest()[0]
    return digest % 2 == 0


def _parse_date(val) -> date | None:
    try:
        s = str(int(val))
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except Exception:
        return None


def _load_sackmann_files(data_dir: str) -> pd.DataFrame:
    """Load and concatenate all atp_matches_*.csv files."""
    import glob
    pattern = os.path.join(data_dir, "atp_matches_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No atp_matches_*.csv files found in {data_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
            logger.info("  Loaded %s (%d rows)", os.path.basename(f), len(df))
        except Exception as exc:
            logger.warning("  Skipping %s: %s", f, exc)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Total Sackmann rows: %d", len(combined))
    return combined


def build(data_dir: str, output_path: str) -> None:
    logger.info("Loading Sackmann CSV files from %s ...", data_dir)
    raw = _load_sackmann_files(data_dir)

    # Require minimum columns
    required = {"tourney_date", "winner_id", "loser_id", "winner_name", "loser_name",
                "tourney_id", "match_num"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort chronologically
    raw = raw.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)

    # State tracking per player
    player_state: dict[int, _PlayerState] = {}
    player_games: dict[int, int] = {}  # total matches played (for K-factor)

    def _get(pid: int) -> _PlayerState:
        if pid not in player_state:
            player_state[pid] = _PlayerState()
            player_games[pid] = 0
        return player_state[pid]

    rows = []

    for _, r in raw.iterrows():
        try:
            w_id = int(r["winner_id"])
            l_id = int(r["loser_id"])
        except (ValueError, TypeError):
            continue

        w1 = _assign_sides(w_id, l_id, str(r.get("tourney_id", "")), int(r.get("match_num", 0)))
        p1_id, p2_id = (w_id, l_id) if w1 else (l_id, w_id)
        game_winner = 1 if w1 else 2

        # Prefix helpers
        def _w(col: str):
            return r.get(col)
        def _l(col: str):
            return r.get(col)

        # Sackmann stat cols: w_ace, l_ace etc.
        if w1:
            p1_ace = r.get("w_ace"); p1_df = r.get("w_df"); p1_svpt = r.get("w_svpt")
            p1_1in = r.get("w_1stIn"); p1_1won = r.get("w_1stWon"); p1_2won = r.get("w_2ndWon")
            p1_svg = r.get("w_SvGms"); p1_bps = r.get("w_bpSaved"); p1_bpf = r.get("w_bpFaced")
            p2_ace = r.get("l_ace"); p2_df = r.get("l_df"); p2_svpt = r.get("l_svpt")
            p2_1in = r.get("l_1stIn"); p2_1won = r.get("l_1stWon"); p2_2won = r.get("l_2ndWon")
            p2_svg = r.get("l_SvGms"); p2_bps = r.get("l_bpSaved"); p2_bpf = r.get("l_bpFaced")
            p1_rank = r.get("winner_rank"); p1_rp = r.get("winner_rank_points")
            p2_rank = r.get("loser_rank"); p2_rp = r.get("loser_rank_points")
            p1_seed = r.get("winner_seed"); p1_entry = r.get("winner_entry")
            p1_name = r.get("winner_name"); p1_hand = r.get("winner_hand")
            p1_ht = r.get("winner_ht"); p1_ioc = r.get("winner_ioc"); p1_age = r.get("winner_age")
            p2_seed = r.get("loser_seed"); p2_entry = r.get("loser_entry")
            p2_name = r.get("loser_name"); p2_hand = r.get("loser_hand")
            p2_ht = r.get("loser_ht"); p2_ioc = r.get("loser_ioc"); p2_age = r.get("loser_age")
        else:
            p1_ace = r.get("l_ace"); p1_df = r.get("l_df"); p1_svpt = r.get("l_svpt")
            p1_1in = r.get("l_1stIn"); p1_1won = r.get("l_1stWon"); p1_2won = r.get("l_2ndWon")
            p1_svg = r.get("l_SvGms"); p1_bps = r.get("l_bpSaved"); p1_bpf = r.get("l_bpFaced")
            p2_ace = r.get("w_ace"); p2_df = r.get("w_df"); p2_svpt = r.get("w_svpt")
            p2_1in = r.get("w_1stIn"); p2_1won = r.get("w_1stWon"); p2_2won = r.get("w_2ndWon")
            p2_svg = r.get("w_SvGms"); p2_bps = r.get("w_bpSaved"); p2_bpf = r.get("w_bpFaced")
            p1_rank = r.get("loser_rank"); p1_rp = r.get("loser_rank_points")
            p2_rank = r.get("winner_rank"); p2_rp = r.get("winner_rank_points")
            p1_seed = r.get("loser_seed"); p1_entry = r.get("loser_entry")
            p1_name = r.get("loser_name"); p1_hand = r.get("loser_hand")
            p1_ht = r.get("loser_ht"); p1_ioc = r.get("loser_ioc"); p1_age = r.get("loser_age")
            p2_seed = r.get("winner_seed"); p2_entry = r.get("winner_entry")
            p2_name = r.get("winner_name"); p2_hand = r.get("winner_hand")
            p2_ht = r.get("winner_ht"); p2_ioc = r.get("winner_ioc"); p2_age = r.get("winner_age")

        # Date features
        match_date = _parse_date(r.get("tourney_date"))
        if match_date is None:
            continue
        year = match_date.year
        yday = match_date.timetuple().tm_yday
        sine_day = math.sin(2 * math.pi * yday / 365.0)
        cosine_day = math.cos(2 * math.pi * yday / 365.0)

        ps1 = _get(p1_id)
        ps2 = _get(p2_id)
        w_ps = ps1 if w1 else ps2
        l_ps = ps2 if w1 else ps1

        # Pre-match form features
        two_weeks_ago = match_date - timedelta(days=14)

        def _weeks_inactive(ps: _PlayerState) -> float:
            if ps.last_match_date is None:
                return -10.0
            return max(0.0, (match_date - ps.last_match_date).days / 7.0)

        p1_win_streak = ps1.winning_streak
        p1_lose_streak = ps1.losing_streak
        p2_win_streak = ps2.winning_streak
        p2_lose_streak = ps2.losing_streak
        p1_inactive = _weeks_inactive(ps1)
        p2_inactive = _weeks_inactive(ps2)
        p1_recent = ps1.recent_matches
        p2_recent = ps2.recent_matches
        p1v2 = ps1.h2h.get(p2_id, 0)
        p2v1 = ps2.h2h.get(p1_id, 0)

        # Pre-match ELO
        p1_elo = ps1.elo
        p2_elo = ps2.elo

        # --- Update state ---
        # ELO
        exp_w = _elo_expected(w_ps.elo, l_ps.elo)
        k_w = _k_factor(player_games.get(w_id, 0))
        k_l = _k_factor(player_games.get(l_id, 0))
        w_ps.elo += k_w * (1.0 - exp_w)
        l_ps.elo += k_l * (0.0 - (1.0 - exp_w))

        # Form
        w_ps.winning_streak += 1
        w_ps.losing_streak = 0
        l_ps.losing_streak += 1
        l_ps.winning_streak = 0

        for ps in (ps1, ps2):
            if ps.last_match_date and ps.last_match_date >= two_weeks_ago:
                ps.recent_matches += 1
            else:
                ps.recent_matches = 1
            ps.last_match_date = match_date

        # H2H
        ps1.h2h[p2_id] = ps1.h2h.get(p2_id, 0) + (1 if game_winner == 1 else 0)
        ps2.h2h[p1_id] = ps2.h2h.get(p1_id, 0) + (1 if game_winner == 2 else 0)

        # Games played
        player_games[w_id] = player_games.get(w_id, 0) + 1
        player_games[l_id] = player_games.get(l_id, 0) + 1

        surface_str = str(r.get("surface", "")) if pd.notna(r.get("surface")) else ""
        surface_code = _SURFACE_MAP.get(surface_str, -1)

        level_str = str(r.get("tourney_level", "")) if pd.notna(r.get("tourney_level")) else ""
        level_code = _LEVEL_MAP.get(level_str, -10)

        round_str = str(r.get("round", "")) if pd.notna(r.get("round")) else ""
        round_code = _ROUND_MAP.get(round_str, -10)

        def _hand(val) -> float:
            if pd.isna(val):
                return -10.0
            return _HAND_MAP.get(str(val), -10.0)

        def _entry(val) -> float:
            return -10.0 if pd.isna(val) else float(hash(str(val)) % 100)

        rows.append({
            "tourney_id": r.get("tourney_id"),
            "tourney_name": r.get("tourney_name"),
            "surface": surface_code,
            "draw_size": r.get("draw_size"),
            "tourney_level": level_code,
            "tourney_date": r.get("tourney_date"),
            "match_num": r.get("match_num"),
            "best_of": r.get("best_of"),
            "round": round_code,
            "minutes": r.get("minutes"),
            "player1_id": p1_id,
            "player1_seed": p1_seed,
            "player1_entry": _entry(p1_entry),
            "player1_name": p1_name,
            "player1_hand": _hand(p1_hand),
            "player1_ht": p1_ht,
            "player1_ioc": _encode_ioc(str(p1_ioc) if pd.notna(p1_ioc) else None),
            "player1_age": p1_age,
            "player2_id": p2_id,
            "player2_seed": p2_seed,
            "player2_entry": _entry(p2_entry),
            "player2_name": p2_name,
            "player2_hand": _hand(p2_hand),
            "player2_ht": p2_ht,
            "player2_ioc": _encode_ioc(str(p2_ioc) if pd.notna(p2_ioc) else None),
            "player2_age": p2_age,
            "player1_ace": p1_ace, "player1_df": p1_df, "player1_svpt": p1_svpt,
            "player1_1stIn": p1_1in, "player1_1stWon": p1_1won, "player1_2ndWon": p1_2won,
            "player1_SvGms": p1_svg, "player1_bpSaved": p1_bps, "player1_bpFaced": p1_bpf,
            "player2_ace": p2_ace, "player2_df": p2_df, "player2_svpt": p2_svpt,
            "player2_1stIn": p2_1in, "player2_1stWon": p2_1won, "player2_2ndWon": p2_2won,
            "player2_SvGms": p2_svg, "player2_bpSaved": p2_bps, "player2_bpFaced": p2_bpf,
            "player1_rank": p1_rank, "player1_rank_points": p1_rp,
            "player2_rank": p2_rank, "player2_rank_points": p2_rp,
            "game_winner": game_winner,
            "player1_elo": p1_elo,
            "player2_elo": p2_elo,
            "year": year,
            "sine_day": sine_day,
            "cosine_day": cosine_day,
            "month": match_date.month,
            "day": match_date.day,
            "yday": yday,
            "player1_winning_streak": p1_win_streak,
            "player2_winning_streak": p2_win_streak,
            "player1_losing_streak": p1_lose_streak,
            "player2_losing_streak": p2_lose_streak,
            "player1_weeks_inactive": p1_inactive,
            "player2_weeks_inactive": p2_inactive,
            "player1_last_two_weeks": p1_recent,
            "player2_last_two_weeks": p2_recent,
            "player1_v_player2_wins": p1v2,
            "player2_v_player1_wins": p2v1,
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(out), output_path)
    logger.info("Date range: %s to %s", out['tourney_date'].min(), out['tourney_date'].max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ATP database from Sackmann CSVs.")
    parser.add_argument("--data-dir", default=_DEFAULT_DATA_DIR, help="Directory with atp_matches_*.csv files")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    build(args.data_dir, args.output)
    logger.info("Next step: poetry run python scripts/enrich_features.py")


if __name__ == "__main__":
    main()
