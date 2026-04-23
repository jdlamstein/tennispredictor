## Project Overview

ML system to predict ATP tennis match winners using historical match data and engineered player
features. Goal: beat online bookmakers (Pinnacle, Bet365) by finding matches where the model's
probability exceeds the bookmaker's implied probability by a meaningful margin.

The full pipeline is implemented: odds fetching → model prediction → Kelly sizing → paper trading
→ result settlement → P&L tracking.

---

## Environment Setup

```bash
poetry install
```

**Required environment variables:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `ATP_ROOTDIR` | `~/Data/tennis` | Parent directory for all data |
| `ATP_DB` | `$ATP_ROOTDIR/tennis_data/atp_database_enriched.csv` | Enriched feature CSV |
| `PAPER_DB` | `$(pwd)/paper_trades.db` | SQLite paper-trade DB (use absolute path) |
| `ODDS_API_KEY` | — | The Odds API key |
| `HOLDOUT_YEAR` | `2022` | First holdout year for model training |
| `MIN_EDGE` | `0.05` | Min model_prob − 1/odds to place bet |
| `MAX_ODDS` | `3.0` | Max odds cap (no heavy underdog bets) |
| `KELLY` | `0.25` | Kelly fraction |

---

## Current Commands (active pipeline)

**Build and enrich database (run once; re-run after Sackmann updates):**
```bash
poetry run python scripts/build_database.py
poetry run python scripts/enrich_features.py
```

**Rolling backtest (walk-forward, one holdout year at a time):**
```bash
MIN_EDGE=0.05 MAX_ODDS=3.0 poetry run python scripts/rolling_backtest.py
```

**Single-period backtest:**
```bash
MIN_EDGE=0.05 MAX_ODDS=3.0 ATP_DB=~/Data/tennis/tennis_data/atp_database_enriched.csv \
  poetry run python scripts/backtest.py
```

**Paper trading:**
```bash
# Predict: fetch live odds and log predictions
PAPER_DB=$(pwd)/paper_trades.db ODDS_API_KEY=<key> \
  poetry run python scripts/paper_trade.py predict

# Settle: match Sackmann results to pending predictions
PAPER_DB=$(pwd)/paper_trades.db \
  poetry run python scripts/paper_trade.py settle --days 2

# Summary: P&L report
PAPER_DB=$(pwd)/paper_trades.db \
  poetry run python scripts/paper_trade.py summary
```

**Tests:**
```bash
poetry run pytest -x -q   # 225 tests
```

---

## Architecture

### Data Flow

```
Sackmann CSVs (tennis_atp/*.csv)
    │
    ▼
scripts/build_database.py
    → atp_database.csv  (~916K rows, raw match data)
    │
    ▼
scripts/enrich_features.py
    → atp_database_enriched.csv  (+28 columns: surface ELO ×8, Glicko-2 ×6, serve EMAs ×14)
    │
    ├──► scripts/backtest.py / rolling_backtest.py
    │        flat-CSV path: _prepare_features() → StandardScaler → XGBoost
    │
    └──► scripts/paper_trade.py
             FeatureStore path: build_training_matrix() → StandardScaler → XGBoost
             → calibration → The Odds API odds → MatchOdds → Kelly sizing → SQLite
```

### Key Source Files

**`src/features/`**

- `feature_store.py` — FeatureStore + PlayerState. Chronological state replay over match history.
  - `_feats_from_states()` — **54-element feature vector** — single source of truth for both training and inference.
  - `PlayerState` fields: ELO (4 surfaces), Glicko-2 (rating/RD/sigma), H2H, win/loss streaks, serve-stat EMAs, days_rest.
  - `build_training_matrix()` — produces (X, y) for model training.
  - `build()` — produces live PlayerState objects for inference.
  - Both paths update identical state fields after each match (training/inference parity invariant).

- `elo.py` — Surface-specific ELO. `add_surface_elo(df)` writes 8 pre-match columns.

- `glicko.py` — Glicko-2 ratings. `add_glicko2(df)` writes 6 pre-match columns.
  Period-based updates; `_flush_glicko()` must be called before advancing periods.

- `serve_stats.py` — Serve-stat EMAs for the flat-CSV path. `add_serve_stats(df)` writes 14 pre-match columns.

**`src/evaluation/`**

- `backtester.py` — `BacktestConfig` + `Backtester.run()`.
  Config: `min_edge`, `max_odds`, `kelly_fraction`, `max_kelly`, `min_ev`.
  Edge filter: `edge = model_prob - 1/odds` (not EV) must exceed `min_edge`.

- `calibration.py` — Platt-scaling wrapper around XGBoost.
  `transform()` returns `[p1_win_prob, p2_win_prob]` — `probs[:, 0]` is always P(player1 wins).

**`src/betting/`**

- `paper_trader.py` — `PaperTrader`, `PaperTraderConfig`, `MatchOdds`.
  SQLite-backed `predictions` table. `run_result_cycle()` matches Sackmann results by surname.

**`src/data/`**

- `odds_api.py` — The Odds API client. `fetch_atp_odds()` → list of `MatchOdds`.

- `results_fetcher.py` — Sackmann CSV result fetching.
  `_from_sackmann()` fetches all years from `cutoff.year` to `current_year` (handles year-boundary settle).

**`scripts/`**

| Script | Purpose |
|--------|---------|
| `build_database.py` | Consolidates Sackmann CSVs → `atp_database.csv` |
| `enrich_features.py` | Adds surface ELO + Glicko-2 + serve-stat EMAs → `atp_database_enriched.csv` |
| `backtest.py` | Single-period backtest (flat-CSV path) |
| `rolling_backtest.py` | Walk-forward backtest (one holdout year at a time) |
| `paper_trade.py` | CLI: `predict \| settle \| summary \| start` |

---

## Feature Space (54 features)

Player1/player2 symmetric pairs of:
- Surface-specific ELO (4 surfaces × 2 players = 8)
- Glicko-2: rating, RD, sigma (3 × 2 = 6)
- H2H win rate + match count (2 × 2 = 4)
- Win streak, loss streak (2 × 2 = 4)
- Days rest capped at 30 (1 × 2 = 2)
- Serve-stat EMAs: first_serve_pct, first_serve_win_pct, second_serve_win_pct, bp_save_pct, ace_rate, df_rate, serve_games (7 × 2 = 14)

Plus: surface encoding (1), year (1), month sin/cos (2) = 4 shared features.

Total: 8 + 6 + 4 + 4 + 2 + 14 + 4 = **52** player features + **2** (ELO diff, Glicko diff used implicitly). Final vector = 54.

---

## Critical Invariants

### Label / Column Convention
- `y = game_winner - 1` → `y=0` means player1 wins, `y=1` means player2 wins
- `probs[:, 0]` = P(class 0) = **P(player1 wins)** — never use `probs[:, 1]` for p1
- `actual_winner ∈ {1, 2}` in raw data

### Kelly / EV / Edge Math
- **Kelly:** `f = (p*b - q) / b` where `b = odds - 1`, `q = 1 - p`
- **EV:** `p*(odds-1) - (1-p)` = `p*b - q`
- **Edge:** `model_prob - 1/odds` (implied prob from decimal odds); used for `min_edge` filter

### Look-Ahead Bias Prevention
- FeatureStore: PRE-match state written to output **first**, then state updated with result
- Serve-stat EMAs: same capture-then-update order in both `serve_stats.py` and `feature_store.py`
- Surface ELO and Glicko-2 in enriched CSV are pre-match values
- Raw serve columns (`w_ace`, `w_svpt`, etc.) dropped by `_DROP_PATTERNS` in `backtest.py`
  — EMA columns are exempt: `"_ema_" not in c` guard prevents false drops

### Training / Inference Consistency
- `build()` and `build_training_matrix()` must update **identical** PlayerState fields after each match
- `_feats_from_states()` is the single source for the feature vector — both paths call it
- Scaler fit on training set only; transform only on test/inference
- Feature count: **54** — any change requires updating the docstring and `test_feature_store.py`

### Surface Map
- `{0: "hard", 1: "clay", 2: "grass", 3: "carpet"}` — consistent across `feature_store.py`, `elo.py`, `build_database.py`

---

## Settlement Data Gap

Sackmann publishes **year-end CSVs**. Latest available: **2024**.
Settlement of 2025/2026 predictions requires a live results source (currently unimplemented).
`results_fetcher.py` already supports multi-year lookback — it will work automatically once
Sackmann publishes 2025+ files.

UTS API fallback (`_from_uts`) is stubbed — returns empty list with a warning.

---

## algo-review Skill

Run `/algo-review` in Claude Code to audit the betting pipeline for:
- Label/column convention violations
- Look-ahead bias
- Kelly/EV/edge math errors
- Training/inference consistency
- EMA capture order

Skill file: `.claude/skills/algo-review/SKILL.md`
