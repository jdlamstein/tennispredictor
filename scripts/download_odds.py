"""Download historical ATP odds from tennis-data.co.uk (free, .xlsx format).

Usage:
    poetry run python scripts/download_odds.py
"""
import sys
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.odds_loader import download_atp_odds

dest = os.path.expanduser("~/Data/tennis/tennis_data/odds")

# Remove any stale .csv files from the previous (failed) download attempt
stale = glob.glob(os.path.join(dest, "*.csv"))
if stale:
    print(f"Removing {len(stale)} stale .csv files from previous attempt...")
    for f in stale:
        os.remove(f)

years = range(2010, 2026)
print(f"Downloading ATP odds for {years.start}–{years.stop - 1} → {dest}")
downloaded = download_atp_odds(years=years, dest_dir=dest, delay_sec=1.5)
print(f"Done. {len(downloaded)} files saved.")
