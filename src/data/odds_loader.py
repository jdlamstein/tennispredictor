"""Download and parse historical odds from tennis-data.co.uk.

tennis-data.co.uk provides free Excel (.xlsx) files with ATP/WTA match
results and bookmaker odds (Bet365, Pinnacle, Betway, etc.) from 2000 onward.

URL pattern
-----------
ATP men's : http://www.tennis-data.co.uk/{year}/{year}.xlsx
WTA women's: http://www.tennis-data.co.uk/{year}w/{year}w.xlsx

Key columns in the downloaded files
------------------------------------
Date    : match date (DD/MM/YY or DD/MM/YYYY)
Winner  : winner full name
Loser   : loser full name
Surface : H (Hard), C (Clay), G (Grass), D (Carpet/Indoor)
B365W   : Bet365 winner odds (decimal)
B365L   : Bet365 loser odds (decimal)
PSW     : Pinnacle winner odds (decimal)  ← preferred for EV calculations
PSL     : Pinnacle loser odds (decimal)
MaxW    : market-maximum winner odds
MaxL    : market-maximum loser odds
AvgW    : market-average winner odds
AvgL    : market-average loser odds

Usage
-----
>>> from src.data.odds_loader import download_atp_odds, load_odds_dir
>>> download_atp_odds(years=range(2010, 2026), dest_dir="~/Data/tennis/tennis_data/odds")
>>> df = load_odds_dir("~/Data/tennis/tennis_data/odds")
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_ATP_URL = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"
_WTA_URL = "http://www.tennis-data.co.uk/{year}w/{year}w.xlsx"

# Column name mapping: tennis-data.co.uk → our standardised format
_COL_MAP = {
    "Date": "date",
    "Winner": "p1_name",       # winner is always player 1 in this dataset
    "Loser": "p2_name",
    "Surface": "surface",
    "PSW": "p1_odds",          # Pinnacle winner odds
    "PSL": "p2_odds",          # Pinnacle loser odds
    "B365W": "b365_p1_odds",
    "B365L": "b365_p2_odds",
    "MaxW": "max_p1_odds",
    "MaxL": "max_p2_odds",
    "AvgW": "avg_p1_odds",
    "AvgL": "avg_p2_odds",
}

# In tennis-data.co.uk files, column "Winner" always won the match
_WINNER_IS_P1 = 1


def download_atp_odds(
    years: range | list[int],
    dest_dir: str | Path,
    tour: str = "atp",
    delay_sec: float = 1.5,
    overwrite: bool = False,
) -> list[Path]:
    """Download tennis-data.co.uk Excel files for given years.

    Parameters
    ----------
    years : range or list[int]
        Calendar years to download (e.g., ``range(2010, 2026)``).
    dest_dir : str or Path
        Directory where files will be saved. Created if absent.
    tour : str
        ``"atp"`` (men's) or ``"wta"`` (women's).
    delay_sec : float
        Seconds to wait between requests to be polite to the server.
    overwrite : bool
        If False (default), skip files that already exist.

    Returns
    -------
    list[Path]
        Paths of successfully downloaded files.
    """
    dest = Path(dest_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    url_template = _ATP_URL if tour == "atp" else _WTA_URL
    downloaded: list[Path] = []

    for year in years:
        url = url_template.format(year=year)
        filename = dest / f"{tour}_{year}.xlsx"

        if filename.exists() and not overwrite:
            logger.info("Already exists — skipping: %s", filename)
            downloaded.append(filename)
            continue

        logger.info("Downloading %s → %s", url, filename)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            filename.write_bytes(resp.content)
            downloaded.append(filename)
            logger.info("Saved %s (%d bytes)", filename.name, len(resp.content))
        except requests.HTTPError as exc:
            logger.warning("HTTP %s for %s — skipping", exc.response.status_code, url)
        except requests.RequestException as exc:
            logger.error("Request failed for %s: %s", url, exc)

        time.sleep(delay_sec)

    return downloaded


def _parse_one(path: Path) -> pd.DataFrame:
    """Load a single tennis-data.co.uk Excel file and normalise columns.

    Parameters
    ----------
    path : Path
        Path to a downloaded .xlsx file.

    Returns
    -------
    pd.DataFrame
        Rows with standard column names. Only rows with valid Pinnacle odds
        (PSW and PSL both present and > 1.0) are kept.
    """
    import warnings
    # Try modern .xlsx first; fall back to legacy .xls binary format (2010-2012)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress openpyxl extension warnings
            raw = pd.read_excel(path, engine="openpyxl")
    except Exception:
        try:
            raw = pd.read_excel(path, engine="xlrd")
        except Exception as exc:
            logger.error("Failed to parse %s: %s", path, exc)
            return pd.DataFrame()

    # Keep only columns present in this file
    keep = {k: v for k, v in _COL_MAP.items() if k in raw.columns}
    df = raw[list(keep.keys())].rename(columns=keep).copy()

    # Standardise date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df["date_int"] = df["date"].dt.strftime("%Y%m%d").astype(int)

    # Require Pinnacle odds — essential for EV calculations
    if "p1_odds" in df.columns and "p2_odds" in df.columns:
        df = df.dropna(subset=["p1_odds", "p2_odds"])
        df = df[(df["p1_odds"] > 1.0) & (df["p2_odds"] > 1.0)]
    else:
        logger.warning("No Pinnacle odds (PSW/PSL) in %s — skipping", path.name)
        return pd.DataFrame()

    # Winner column always corresponds to the match winner (player 1 in our schema)
    df["actual_winner"] = _WINNER_IS_P1

    return df.reset_index(drop=True)


def load_odds_dir(
    odds_dir: str | Path,
    tour: str = "atp",
) -> pd.DataFrame:
    """Load all downloaded odds files from ``odds_dir`` into one DataFrame.

    Parameters
    ----------
    odds_dir : str or Path
        Directory containing files named ``atp_{year}.xlsx`` (or wta_).
    tour : str
        ``"atp"`` or ``"wta"`` — filters files by prefix.

    Returns
    -------
    pd.DataFrame
        Combined, chronologically sorted odds DataFrame.
        Empty DataFrame if no valid files are found.
    """
    odds_path = Path(odds_dir).expanduser()
    files = sorted(odds_path.glob(f"{tour}_*.xlsx"))

    if not files:
        logger.warning("No %s odds files found in %s", tour, odds_path)
        return pd.DataFrame()

    frames = [_parse_one(f) for f in files]
    frames = [f for f in frames if not f.empty]

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    logger.info(
        "Loaded %d odds records from %d files (%s)",
        len(combined), len(frames), odds_path,
    )
    return combined
