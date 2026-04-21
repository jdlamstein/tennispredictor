# Plan: Data Refresh & Result Settlement (Phase 7)

| Field   | Value |
|---------|-------|
| Created | 2026-04-20 |
| Updated | 2026-04-20 |
| Status  | draft |
| Commit  | 1ffc5e9 |
| Jira    | N/A |

## Context

Paper trading is live: 21 predictions logged for ATP Madrid Open (2026-04-20).

Two blockers prevent the system from being useful long-term:

1. **Stale player ratings** — `atp_database.csv` ends 2023-02-06. FeatureStore
   last_date=2021-12-27 (enriched DB cut even earlier). ELO/Glicko ratings are
   3+ years stale — predictions use outdated player strength estimates.

2. **Settlement untested** — `_from_sackmann` is implemented and downloads
   current-year Sackmann CSV, but has never been run against real pending bets.
   UTS fallback is a stub returning `[]`.

---

## Goals

- [x] **7A. Extend ATP database** — download 2023–2025 Sackmann CSVs, append to base DB, re-run enrichment. FeatureStore ratings become current.
- [ ] **7B. Verify settlement** — run `paper_trade.py settle` against the 21 logged Madrid predictions; confirm bets settle correctly when results come in.
- [x] **7C. Data refresh script** — `scripts/refresh_data.sh` created. Downloads current + prev year, rebuilds DB + enriched features, clears model cache.

---

## Phase 7A — Extend ATP Database

### Problem
`atp_database.csv` was built from Sackmann data through early Feb 2023.
Missing: most of 2023, all of 2024, and 2025 to date.

### Sackmann file format
Jeff Sackmann's tennis_atp repo (GitHub: JeffSackmann/tennis_atp) publishes:
```
atp_matches_{year}.csv   # one file per year
```
Columns include: `tourney_date`, `winner_name`, `loser_name`, `winner_id`,
`loser_id`, `surface`, `tourney_name`, `score`, stats, rankings, etc.

### Steps

1. **Download missing years:**
   ```bash
   cd ~/Data/tennis/tennis_data
   BASE=https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master
   for year in 2023 2024 2025; do
     curl -fO "$BASE/atp_matches_${year}.csv"
   done
   ```

2. **Build unified base DB** — `scripts/build_database.py` (already exists):
   re-run to include 2023–2025 files and produce updated `atp_database.csv`.

3. **Re-run enrichment:**
   ```bash
   ATP_ROOTDIR=~/Data/tennis poetry run python scripts/enrich_features.py
   ```
   Produces `atp_database_enriched.csv` with ELO/Glicko/form through 2025.

4. **Delete model cache** (forces retrain on new data):
   ```bash
   rm -f paper_model.joblib
   ```

5. **Re-run rolling backtest** — 2024 year should now have data:
   ```bash
   poetry run python scripts/rolling_backtest.py
   ```

### Expected outcome
- FeatureStore last_date ≈ 2025-04 (current season)
- 2024 rolling backtest year populated (was "skipped — no data")
- Player ELO ratings reflect last 2+ years of results

---

## Phase 7B — Verify Result Settlement

### Current state
`_from_sackmann` (in `src/data/results_fetcher.py`) fetches current year CSV:
```python
_SACKMANN_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
```
This is already implemented and should work.

### Verify
```bash
PAPER_DB=/tmp/paper_test.db ATP_DB=~/Data/tennis/tennis_data/atp_database_enriched.csv \
  poetry run python scripts/paper_trade.py settle --days 7
```

Check:
- Sackmann fetch succeeds (Madrid 2025 results should be there)
- Player name matching works (Sackmann uses "Djokovic N." format; FeatureStore surname matcher handles this)
- Settled bets have `outcome` and `pnl` populated in SQLite

### If Sackmann name format mismatches
Sackmann: `winner_name = "Djokovic N."` vs TheOddsAPI: `player1 = "Novak Djokovic"`.
`filter_for_pending` uses `_sn(name)` = last token → "Djokovic" matches either format.
If still failing: add logging to `_find_result` in `paper_trader.py` to show candidate names.

---

## Phase 7C — Weekly Data Refresh Script

### `scripts/refresh_data.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

BASE=https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master
DATA_DIR=${ATP_ROOTDIR:-~/Data/tennis}/tennis_data
CACHE=${MODEL_CACHE_PATH:-paper_model.joblib}

# Download current and previous year
for year in $(date +%Y) $(($(date +%Y)-1)); do
  curl -fsSL "$BASE/atp_matches_${year}.csv" -o "$DATA_DIR/atp_matches_${year}.csv"
done

# Rebuild base database + enriched features
poetry run python scripts/build_database.py
poetry run python scripts/enrich_features.py

# Invalidate model cache so next predict retrains on fresh data
rm -f "$CACHE"
echo "Data refresh complete. Cache cleared."
```

Run weekly via cron or manually before predict cycles.

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `~/Data/tennis/tennis_data/atp_matches_2023.csv` | Download from Sackmann |
| `~/Data/tennis/tennis_data/atp_matches_2024.csv` | Download from Sackmann |
| `~/Data/tennis/tennis_data/atp_matches_2025.csv` | Download from Sackmann |
| `~/Data/tennis/tennis_data/atp_database.csv` | Rebuild with new years |
| `~/Data/tennis/tennis_data/atp_database_enriched.csv` | Re-enrich |
| `scripts/refresh_data.sh` | New — weekly data refresh |

---

## Verification

```bash
# Check FeatureStore last date (should be ~2025-04)
PAPER_DB=/tmp/paper_test.db ATP_DB=~/Data/tennis/tennis_data/atp_database_enriched.csv \
  poetry run python scripts/paper_trade.py predict

# Rolling backtest — 2024 should now be populated
poetry run python scripts/rolling_backtest.py

# Settle Madrid predictions
PAPER_DB=/tmp/paper_test.db ATP_DB=~/Data/tennis/tennis_data/atp_database_enriched.csv \
  poetry run python scripts/paper_trade.py settle --days 7

# View P&L
PAPER_DB=/tmp/paper_test.db poetry run python scripts/paper_trade.py summary
```

## Success Criteria
- FeatureStore last_date ≥ 2025-01-01
- Rolling backtest 2024 year populated with bets + ROI
- At least 1 settled bet with correct outcome in paper DB
- `paper_trade.py summary` shows non-zero P&L
