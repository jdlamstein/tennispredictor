#!/usr/bin/env bash
# refresh_data.sh — pull latest Sackmann data, rebuild DB + enriched features, clear model cache.
# Run weekly before predict cycles to keep player ratings current.
#
# Usage:
#   ATP_ROOTDIR=~/Data/tennis bash scripts/refresh_data.sh
#
# Environment variables:
#   ATP_ROOTDIR        Parent directory (default: ~/Data/tennis)
#   MODEL_CACHE_PATH   Path to joblib cache to invalidate (default: paper_model.joblib)

set -euo pipefail

ATP_ROOTDIR=${ATP_ROOTDIR:-~/Data/tennis}
DATA_DIR=$(eval echo "${ATP_ROOTDIR}/tennis_data")
CACHE=${MODEL_CACHE_PATH:-paper_model.joblib}
BASE=https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master

CURRENT_YEAR=$(date +%Y)
PREV_YEAR=$((CURRENT_YEAR - 1))

echo "=== ATP Data Refresh ==="
echo "Data dir : ${DATA_DIR}"
echo "Cache    : ${CACHE}"
echo ""

# Download current and previous year (current may be incomplete mid-season)
for year in "${PREV_YEAR}" "${CURRENT_YEAR}"; do
    url="${BASE}/atp_matches_${year}.csv"
    dest="${DATA_DIR}/atp_matches_${year}.csv"
    echo "Fetching ${url} ..."
    curl -fsSL "${url}" -o "${dest}"
    echo "  Saved ${dest}"
done

# Rebuild base database from all Sackmann CSVs
echo ""
echo "Building atp_database.csv ..."
poetry run python scripts/build_database.py --data-dir "${DATA_DIR}"

# Re-run enrichment (surface ELO + Glicko-2)
echo ""
echo "Enriching features ..."
ATP_ROOTDIR="${ATP_ROOTDIR}" poetry run python scripts/enrich_features.py

# Invalidate model cache — forces retrain on fresh data next predict run
if [ -f "${CACHE}" ]; then
    rm -f "${CACHE}"
    echo ""
    echo "Model cache cleared: ${CACHE}"
fi

echo ""
echo "Refresh complete. Run 'poetry run python scripts/paper_trade.py predict' to retrain + predict."
