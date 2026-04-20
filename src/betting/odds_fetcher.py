"""Live odds fetching for ATP tennis matches.

Two sources:
  1. Betfair Exchange (primary) — free API via betfairlightweight; requires
     a funded Betfair account for non-delayed prices.
  2. OddsPortal scraper (fallback) — no account required; fragile to layout
     changes; rate-limit cautiously.

Usage
-----
    from src.betting.odds_fetcher import BetfairFetcher, OddsPortalFetcher

    # Betfair (preferred)
    fetcher = BetfairFetcher.from_env()
    markets = fetcher.fetch_tennis_markets()

    # OddsPortal fallback
    op = OddsPortalFetcher()
    matches = op.fetch_upcoming_atp()

Environment variables (Betfair)
--------------------------------
    BETFAIR_USERNAME   Betfair account username
    BETFAIR_PASSWORD   Betfair account password
    BETFAIR_APP_KEY    Application key from Betfair developer portal
    BETFAIR_CERT_PATH  Path to SSL cert file (default: ~/.betfair/client-2048.crt)
    BETFAIR_KEY_PATH   Path to SSL key file (default: ~/.betfair/client-2048.key)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_ODDSPORTAL_BASE = "https://www.oddsportal.com"
_ODDSPORTAL_ATP = "/tennis/atp/"
_REQUEST_TIMEOUT = 10  # seconds
_SCRAPE_DELAY = 2.0    # seconds between OddsPortal requests (rate limiting)


@dataclass(frozen=True)
class MatchOdds:
    """Odds for one match from any source.

    Attributes
    ----------
    player1 : str
        Player 1 name (winner side in source, or alphabetical first).
    player2 : str
        Player 2 name.
    p1_odds : float
        Decimal odds for player 1 to win (e.g. 1.50 = 50% implied + vig).
    p2_odds : float
        Decimal odds for player 2 to win.
    source : str
        Origin label: ``"betfair"`` or ``"oddsportal"``.
    market_id : str
        Bookmaker/exchange market identifier for deduplication.
    fetched_at : datetime
        UTC timestamp when odds were fetched.
    tournament : str
        Tournament name when available.
    """

    player1: str
    player2: str
    p1_odds: float
    p2_odds: float
    source: str
    market_id: str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tournament: str = ""


class BetfairFetcher:
    """Wraps betfairlightweight to fetch live ATP match-winner markets.

    Parameters
    ----------
    username, password, app_key : str
        Betfair credentials.
    cert_path, key_path : str
        Paths to SSL certificate and key files. Required by Betfair's
        non-interactive login endpoint.
    """

    _TENNIS_EVENT_TYPE = "2"   # Betfair event type ID for Tennis
    _MATCH_ODDS_MARKET = "Match Odds"

    def __init__(
        self,
        username: str,
        password: str,
        app_key: str,
        cert_path: str,
        key_path: str,
    ) -> None:
        self._username = username
        self._password = password
        self._app_key = app_key
        self._cert = (cert_path, key_path)
        self._client: object | None = None

    @classmethod
    def from_env(cls) -> "BetfairFetcher":
        """Construct from environment variables.

        Raises
        ------
        ValueError
            If any required environment variable is missing.
        """
        required = {
            "BETFAIR_USERNAME": os.environ.get("BETFAIR_USERNAME"),
            "BETFAIR_PASSWORD": os.environ.get("BETFAIR_PASSWORD"),
            "BETFAIR_APP_KEY":  os.environ.get("BETFAIR_APP_KEY"),
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing environment variables: {missing}")

        home = os.path.expanduser("~")
        cert_path = os.environ.get("BETFAIR_CERT_PATH",
                                   os.path.join(home, ".betfair", "client-2048.crt"))
        key_path  = os.environ.get("BETFAIR_KEY_PATH",
                                   os.path.join(home, ".betfair", "client-2048.key"))

        return cls(
            username=required["BETFAIR_USERNAME"],   # type: ignore[arg-type]
            password=required["BETFAIR_PASSWORD"],   # type: ignore[arg-type]
            app_key=required["BETFAIR_APP_KEY"],     # type: ignore[arg-type]
            cert_path=cert_path,
            key_path=key_path,
        )

    def _ensure_logged_in(self) -> None:
        """Lazy login — only authenticate on first API call."""
        if self._client is not None:
            return
        try:
            import betfairlightweight
        except ImportError as exc:
            raise ImportError(
                "betfairlightweight not installed. "
                "Run: poetry add betfairlightweight"
            ) from exc

        self._client = betfairlightweight.APIClient(
            username=self._username,
            password=self._password,
            app_key=self._app_key,
            certs=self._cert,
        )
        self._client.login()  # type: ignore[union-attr]
        logger.info("Betfair login successful.")

    def fetch_tennis_markets(
        self,
        competition_ids: Optional[list[str]] = None,
    ) -> list[MatchOdds]:
        """Fetch all live ATP Match Odds markets from Betfair.

        Parameters
        ----------
        competition_ids : list[str] | None
            Betfair competition IDs to filter by (e.g. ATP singles comps).
            None fetches all tennis Match Odds markets.

        Returns
        -------
        list[MatchOdds]
            One entry per available market. Empty list if API unavailable.
        """
        self._ensure_logged_in()
        client = self._client

        market_filter = {
            "eventTypeIds": [self._TENNIS_EVENT_TYPE],
            "marketTypeCodes": [self._MATCH_ODDS_MARKET.upper().replace(" ", "_")],
        }
        if competition_ids:
            market_filter["competitionIds"] = competition_ids

        try:
            catalogues = client.betting.list_market_catalogue(  # type: ignore[union-attr]
                filter=market_filter,
                market_projection=["COMPETITION", "EVENT", "RUNNER_DESCRIPTION"],
                max_results=200,
            )
        except Exception as exc:
            logger.error("Betfair list_market_catalogue failed: %s", exc)
            return []

        if not catalogues:
            logger.info("No Betfair tennis markets found.")
            return []

        market_ids = [c.market_id for c in catalogues]
        try:
            books = client.betting.list_market_book(  # type: ignore[union-attr]
                market_ids=market_ids,
                price_projection={"priceData": ["EX_BEST_OFFERS"]},
            )
        except Exception as exc:
            logger.error("Betfair list_market_book failed: %s", exc)
            return []

        results: list[MatchOdds] = []
        cat_by_id = {c.market_id: c for c in catalogues}
        now = datetime.now(timezone.utc)

        for book in books:
            cat = cat_by_id.get(book.market_id)
            if cat is None or len(book.runners) < 2:
                continue

            runners = book.runners
            names = [r.runner_name for r in runners] if hasattr(runners[0], "runner_name") else ["", ""]

            def _best_back(runner: object) -> float:
                try:
                    prices = runner.ex.available_to_back  # type: ignore[union-attr]
                    return float(prices[0].price) if prices else 0.0
                except Exception:
                    return 0.0

            p1_odds = _best_back(runners[0])
            p2_odds = _best_back(runners[1])

            if p1_odds <= 1.0 or p2_odds <= 1.0:
                continue  # no valid market

            tournament = cat.competition.name if hasattr(cat, "competition") and cat.competition else ""
            results.append(MatchOdds(
                player1=names[0] if len(names) > 0 else "Unknown",
                player2=names[1] if len(names) > 1 else "Unknown",
                p1_odds=p1_odds,
                p2_odds=p2_odds,
                source="betfair",
                market_id=book.market_id,
                fetched_at=now,
                tournament=tournament,
            ))

        logger.info("Betfair: fetched %d tennis markets.", len(results))
        return results

    def logout(self) -> None:
        if self._client is not None:
            try:
                self._client.logout()  # type: ignore[union-attr]
            except Exception:
                pass
            self._client = None


class OddsPortalFetcher:
    """Scrapes upcoming ATP match odds from OddsPortal.

    **Fragile**: OddsPortal layout changes break this. Use Betfair when
    available. Falls back gracefully (returns empty list) on parse failure.

    Parameters
    ----------
    session : requests.Session | None
        Optional pre-configured session (for testing / proxy injection).
    delay : float
        Seconds to wait between requests. Default 2.0 to avoid rate limits.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        delay: float = _SCRAPE_DELAY,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._delay = delay

    def fetch_upcoming_atp(self) -> list[MatchOdds]:
        """Scrape upcoming ATP match odds from OddsPortal's ATP page.

        Returns
        -------
        list[MatchOdds]
            Parsed match odds. Empty list on network/parse error.
        """
        url = _ODDSPORTAL_BASE + _ODDSPORTAL_ATP
        try:
            resp = self._session.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("OddsPortal fetch failed: %s", exc)
            return []

        time.sleep(self._delay)
        return self._parse_atp_page(resp.text)

    def _parse_atp_page(self, html: str) -> list[MatchOdds]:
        """Parse OddsPortal ATP page HTML into MatchOdds records.

        OddsPortal embeds odds in JSON within ``<script>`` tags labelled
        ``__NEXT_DATA__``. Extract and parse that JSON blob.
        """
        try:
            import json
            import re

            match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
            if not match:
                logger.warning("OddsPortal: __NEXT_DATA__ script tag not found.")
                return []

            data = json.loads(match.group(1))
            events = (
                data.get("props", {})
                    .get("pageProps", {})
                    .get("initialData", {})
                    .get("tournamentEvents", {})
                    .get("events", [])
            )
            if not events:
                logger.warning("OddsPortal: no events found in JSON payload.")
                return []

            results: list[MatchOdds] = []
            now = datetime.now(timezone.utc)
            for ev in events:
                home = ev.get("home-name", "")
                away = ev.get("away-name", "")
                odds_raw = ev.get("odds", {})
                # OddsPortal odds key varies; look for the first bookmaker
                if not odds_raw or not home or not away:
                    continue
                first_book = next(iter(odds_raw.values()), None)
                if first_book is None:
                    continue
                try:
                    p1_o = float(first_book[0])
                    p2_o = float(first_book[1])
                except (IndexError, TypeError, ValueError):
                    continue
                if p1_o <= 1.0 or p2_o <= 1.0:
                    continue
                results.append(MatchOdds(
                    player1=home,
                    player2=away,
                    p1_odds=p1_o,
                    p2_odds=p2_o,
                    source="oddsportal",
                    market_id=str(ev.get("id", "")),
                    fetched_at=now,
                    tournament=ev.get("tournament-name", ""),
                ))
            logger.info("OddsPortal: parsed %d matches.", len(results))
            return results

        except Exception as exc:
            logger.warning("OddsPortal parse error: %s", exc)
            return []
