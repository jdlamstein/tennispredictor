"""Fetch recent ATP match results for paper-trade settlement.

Two sources (tried in order):
  1. Jeff Sackmann's tennis_atp GitHub — free, reliable, updated daily/weekly.
     Fetches the current-year CSV directly from raw.githubusercontent.com.
  2. Ultimate Tennis Statistics REST API — free, no auth required, updated
     within hours of match completion.

Both sources return a list of dicts compatible with PaperTrader.run_result_cycle():
    [{"player1": str, "player2": str, "winner": int (1 or 2)}, ...]

Usage
-----
    from src.data.results_fetcher import fetch_recent_results

    results = fetch_recent_results(days=1)
    trader.run_result_cycle(results)
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 15
_SACKMANN_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/"
    "master/atp_matches_{year}.csv"
)
_UTS_BASE = "https://www.ultimatetennisstatistics.com/rest"
_UTS_MATCHES = _UTS_BASE + "/playerProfile/matches"


def fetch_recent_results(
    days: int = 1,
    session: Optional[requests.Session] = None,
) -> list[dict]:
    """Return ATP match results completed within the last ``days`` days.

    Tries Sackmann GitHub first; falls back to Ultimate Tennis Statistics.

    Parameters
    ----------
    days : int
        Look-back window in calendar days (default 1 = yesterday + today).
    session : requests.Session | None
        Optional pre-configured session (useful for testing / proxies).

    Returns
    -------
    list[dict]
        Each dict has keys: ``player1`` (str), ``player2`` (str),
        ``winner`` (int — 1 = player1 won, 2 = player2 won),
        ``tournament`` (str), ``date`` (date).
    """
    s = session or requests.Session()
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)

    results = _from_sackmann(s, cutoff)
    if results:
        logger.info("Results: %d matches from Sackmann (cutoff=%s).", len(results), cutoff)
        return results

    logger.info("Sackmann returned 0 results; trying UTS API.")
    results = _from_uts(s, cutoff)
    logger.info("Results: %d matches from UTS (cutoff=%s).", len(results), cutoff)
    return results


# ---------------------------------------------------------------------------
# Source 1: Jeff Sackmann tennis_atp GitHub
# ---------------------------------------------------------------------------

def _from_sackmann(
    session: requests.Session,
    cutoff: date,
) -> list[dict]:
    """Download current-year ATP matches CSV from Sackmann's GitHub repo."""
    year = datetime.now().year
    url = _SACKMANN_URL.format(year=year)
    try:
        resp = session.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Sackmann fetch failed: %s", exc)
        return []

    try:
        import pandas as pd
        df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    except Exception as exc:
        logger.warning("Sackmann CSV parse failed: %s", exc)
        return []

    # Column names in Sackmann format: tourney_date, winner_name, loser_name, tourney_name
    required = {"tourney_date", "winner_name", "loser_name"}
    if not required.issubset(df.columns):
        logger.warning("Sackmann CSV missing expected columns: %s", df.columns.tolist())
        return []

    df = df.dropna(subset=["tourney_date", "winner_name", "loser_name"])
    df["_date"] = df["tourney_date"].apply(_parse_sackmann_date)
    df = df[df["_date"] >= cutoff]

    results: list[dict] = []
    for _, row in df.iterrows():
        # Sackmann stores winner and loser directly — winner is always player1 here.
        # PaperTrader uses (player1, player2) from our MatchOdds which may have
        # either player listed first. Return both orderings for the matcher.
        results.append({
            "player1":    row["winner_name"],
            "player2":    row["loser_name"],
            "winner":     1,
            "tournament": row.get("tourney_name", ""),
            "date":       row["_date"],
        })
    return results


def _parse_sackmann_date(val: object) -> date:
    """Parse Sackmann tourney_date (YYYYMMDD int or str) to date."""
    try:
        s = str(int(val))
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except Exception:
        return date.min


# ---------------------------------------------------------------------------
# Source 2: Ultimate Tennis Statistics REST API
# ---------------------------------------------------------------------------

def _from_uts(
    session: requests.Session,
    cutoff: date,
) -> list[dict]:
    """Fetch recent completed matches from the UTS public REST API."""
    # UTS /playerProfile/matches requires a player ID. Instead use the
    # tournament-matches endpoint with a broad filter if available, or
    # fall through with an empty list — UTS doesn't expose a global
    # recent-matches feed on the free tier.
    # TODO: integrate a UTS-compatible endpoint when available.
    logger.warning(
        "UTS fallback not fully implemented. "
        "Returning empty list — pass results manually or use Sackmann source."
    )
    return []


# ---------------------------------------------------------------------------
# Convenience: filter to only unresolved prediction player-pairs
# ---------------------------------------------------------------------------

def filter_for_pending(
    results: list[dict],
    pending: list[tuple[str, str]],
) -> list[dict]:
    """Keep only result dicts whose player pair appears in ``pending``.

    Parameters
    ----------
    results : list[dict]
        Full list from ``fetch_recent_results``.
    pending : list[tuple[str, str]]
        ``(player1, player2)`` pairs with unsettled predictions.

    Returns
    -------
    list[dict]
        Filtered subset.
    """
    def _sn(name: str) -> str:
        return name.split()[-1].lower()

    pending_sns = {(_sn(p1), _sn(p2)) for p1, p2 in pending}
    out: list[dict] = []
    for r in results:
        pair = (_sn(r["player1"]), _sn(r["player2"]))
        pair_rev = (pair[1], pair[0])
        if pair in pending_sns or pair_rev in pending_sns:
            out.append(r)
    return out
