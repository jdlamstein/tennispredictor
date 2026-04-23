# Tennis Predictor

With [data generously maintained by Jeff Sackmann](https://github.com/JeffSackmann/tennis_atp), I built a
machine learning system to predict ATP tennis match winners and use those predictions to beat online
betting houses such as Pinnacle Sports and Bet365.

Bookmakers set odds slightly above 100% (typically 102–104%), giving them an edge regardless of outcome.
A nice paper by [Kaunitz, Zhong, and Kreiner](https://arxiv.org/abs/1710.02824) exploited situations
where betting odds deviated from the expected market average — their strategy was profitable until
bookmakers identified and limited them.

Player skill representation is critical. Tennis ranking points decay over 52 weeks and are
tournament-weighted; ELO (adapted from chess) provides more stable relative ratings. Glicko-2 extends
ELO with a rating deviation and volatility term, capturing consistency alongside raw skill.

Bet sizing follows [Kelly Criterion](https://www.doc.ic.ac.uk/teaching/distinguished-projects/2015/m.sipko.pdf):
stake as a fraction of bankroll proportional to edge over the bookmaker's implied probability.
Quarter-Kelly (25% of full Kelly) reduces variance while preserving long-run EV.

---

## Architecture

```
Sackmann CSVs ──► build_database.py ──► atp_database.csv
                                              │
                                   enrich_features.py
                                   (surface ELO, Glicko-2,
                                    serve-stat EMAs, +28 cols)
                                              │
                                   atp_database_enriched.csv
                                              │
                          ┌───────────────────┼───────────────────┐
                          │                   │                   │
                    FeatureStore        rolling_backtest     paper_trade
                   (54 features)         (walk-forward         (live odds
                    XGBoost +             validation)        → SQLite DB)
                    calibration
```

---

## Setup

### Prerequisites

- Python 3.11+ with [Poetry](https://python-poetry.org/)
- Sackmann tennis_atp repo: `git clone https://github.com/JeffSackmann/tennis_atp ~/Data/tennis/tennis_atp`
- [The Odds API](https://the-odds-api.com/) key (free tier: 500 requests/month) for live paper trading

### Install

```bash
poetry install
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ATP_ROOTDIR` | `~/Data/tennis` | Parent directory for all data files |
| `ATP_DB` | `$ATP_ROOTDIR/tennis_data/atp_database_enriched.csv` | Enriched CSV used by backtester and FeatureStore |
| `PAPER_DB` | `$(pwd)/paper_trades.db` | SQLite database for paper-trade predictions (use absolute path) |
| `ODDS_API_KEY` | — | The Odds API key |
| `HOLDOUT_YEAR` | `2022` | First holdout year; model trained on all data before this year |
| `MIN_EDGE` | `0.05` | Minimum model edge over implied probability to place a bet |
| `MAX_ODDS` | `3.0` | Maximum odds (no heavy underdog bets) |
| `KELLY` | `0.25` | Kelly fraction (quarter-Kelly) |

---

## Data Pipeline

### 1. Build base database

Consolidates all Sackmann ATP CSVs into a single chronologically sorted file:

```bash
poetry run python scripts/build_database.py
# Output: ~/Data/tennis/tennis_data/atp_database.csv
```

### 2. Enrich with rating and serve-stat features

Adds 28 new columns — surface-specific ELO (8), Glicko-2 (6), serve-stat EMAs (14) — all
computed as pre-match values with no look-ahead bias:

```bash
poetry run python scripts/enrich_features.py
# Input:  atp_database.csv
# Output: atp_database_enriched.csv  (~916K rows, 28 new columns, ~2.5 min)
```

Re-run this after pulling updated Sackmann data.

---

## Backtesting

### Rolling backtest (walk-forward)

Trains one model per holdout year and evaluates on the next year — no future data leaks:

```bash
MIN_EDGE=0.05 MAX_ODDS=3.0 poetry run python scripts/rolling_backtest.py
```

### Single-period backtest

```bash
MIN_EDGE=0.05 MAX_ODDS=3.0 \
  ATP_DB=~/Data/tennis/tennis_data/atp_database_enriched.csv \
  poetry run python scripts/backtest.py
```

---

## Paper Trading (Live)

### Predict — fetch current odds and log predictions

```bash
PAPER_DB=$(pwd)/paper_trades.db \
  ATP_DB=~/Data/tennis/tennis_data/atp_database_enriched.csv \
  ODDS_API_KEY=<your-key> \
  poetry run python scripts/paper_trade.py predict
```

### Settle — match completed results to predictions

```bash
PAPER_DB=$(pwd)/paper_trades.db \
  poetry run python scripts/paper_trade.py settle --days 2
```

> **Note:** Results are sourced from Sackmann's year-end CSVs (currently available through 2024).
> Settlement of 2025/2026 predictions will work once those files are published, or when a live
> results source is integrated.

### Summary — P&L report

```bash
PAPER_DB=$(pwd)/paper_trades.db \
  poetry run python scripts/paper_trade.py summary
```

---

## Tests

```bash
poetry run pytest -x -q   # 225 tests
```

---

## Results

### Model

- **Algorithm:** XGBoost with Platt-scaling calibration
- **Features:** 54 (surface-specific ELO diff × 4 surfaces, Glicko-2 diff, H2H record,
  serve-stat EMAs, days rest, win/loss streaks, surface encoding, time encoding)
- **Accuracy at high confidence** (model_prob > 0.65): 74.8%

### Backtest (2016–2021, Phase 8 filters)

Bet filters: min edge = 5% above implied probability, max odds = 3.0, quarter-Kelly sizing.

| Metric | Value |
|--------|-------|
| Mean ROI | -4.4% |
| Profitable years | 2 / 6 |
| Bet win rate | ~51% |
| Break-even win rate | ~53% (given ~4% average bookmaker vig) |

**Filter effect:** Edge filter reduces bet volume by ~60% and raises win rate from 43% → 51%.
The remaining gap to break-even (~2pp) is the active research target.

### Active paper trading

- 32 predictions logged for Madrid Open 2026 (April 2026)
- Settlement pending Sackmann 2026 data publication

---

## Legacy Code

The `preprocessing/`, `models/`, and `main/` directories contain the original Phase 0-2 code
(PyTorch Lightning MLP, scikit-learn classifiers). These achieved 90–93% test accuracy but did
not implement betting strategy. They are preserved for reference; the active pipeline lives in
`src/` and `scripts/`.
