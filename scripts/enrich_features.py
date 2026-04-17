"""Add surface-specific ELO and Glicko-2 features to atp_database.csv.

Reads the existing database, appends 14 new feature columns, and writes to
atp_database_enriched.csv (or a custom output path). The input file is never
modified; you can re-run safely.

New columns added
-----------------
Surface ELO (8):  player{1,2}_elo_{hard,clay,grass,carpet}
Glicko-2   (6):  player{1,2}_glicko, player{1,2}_rd, player{1,2}_sigma

Usage
-----
    poetry run python scripts/enrich_features.py
    poetry run python scripts/enrich_features.py --output /path/to/out.csv

Environment variables
---------------------
    ATP_ROOTDIR   Parent directory (default: ~/Data/tennis)
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.features.elo import add_surface_elo
from src.features.glicko import add_glicko2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ATP_ROOTDIR = os.path.expanduser(os.environ.get("ATP_ROOTDIR", "~/Data/tennis"))
_DEFAULT_INPUT = os.path.join(ATP_ROOTDIR, "tennis_data", "atp_database.csv")
_DEFAULT_OUTPUT = os.path.join(ATP_ROOTDIR, "tennis_data", "atp_database_enriched.csv")


def enrich(input_path: str, output_path: str) -> None:
    t_start = time.time()

    logger.info("Loading %s ...", input_path)
    raw = pd.read_csv(input_path, low_memory=False)
    logger.info("  %d rows × %d cols loaded in %.1fs", len(raw), len(raw.columns),
                time.time() - t_start)

    # Both feature functions require chronological ordering.
    raw = raw.sort_values("tourney_date").reset_index(drop=True)

    # Surface ELO — 8 new columns
    logger.info("Computing surface-specific ELO ...")
    t1 = time.time()
    enriched = add_surface_elo(raw)
    logger.info("  Done in %.1fs", time.time() - t1)

    # Glicko-2 — 6 new columns
    logger.info("Computing Glicko-2 ratings ...")
    t2 = time.time()
    enriched = add_glicko2(enriched)
    logger.info("  Done in %.1fs", time.time() - t2)

    new_cols = [c for c in enriched.columns if c not in raw.columns]
    logger.info("Added %d columns: %s", len(new_cols), new_cols)

    # Verify no column-count regression
    assert len(enriched.columns) == len(raw.columns) + len(new_cols), (
        "Column count mismatch after enrichment — check for accidental drops."
    )
    assert len(enriched) == len(raw), (
        "Row count changed during enrichment — data integrity violation."
    )

    logger.info("Saving to %s ...", output_path)
    enriched.to_csv(output_path, index=False)
    logger.info("Total time: %.1fs", time.time() - t_start)
    logger.info("Run backtest with:  ATP_DB=%s poetry run python scripts/backtest.py",
                output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich ATP database with surface ELO and Glicko-2 features."
    )
    parser.add_argument("--input", default=_DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    enrich(args.input, args.output)


if __name__ == "__main__":
    main()
