# Plan: Phase 8 — Prediction Improvement & Profitable Betting

| Field   | Value |
|---------|-------|
| Created | 2026-04-21 |
| Updated | 2026-04-21 |
| Status  | completed |
| Commit  | 7fe598a |
| Jira    | N/A |

## Context

Current system bets any +EV opportunity with 2% min EV threshold and no odds cap.
Backtest diagnostics showed root cause: Kelly over-bets underdogs at high odds (avg 3.73)
where model says ~50.5% win but actual rate is 43%. Bookmaker vig (~4.5%) > model edge.

Model accuracy at high confidence: 74.8% — good. Problem is BET SELECTION, not prediction.

Second problem: feature vector has zero serve statistics despite them being:
- Extractable from Sackmann CSVs (already in `atp_database.csv`)
- Among the strongest predictors in literature (RF paper: "serve strength is key predictor")
- Currently DROPPED by `_DROP_PATTERNS` in `backtest.py` (raw match-level, correct to drop)
- Need to be computed as rolling player averages in `feature_store.py` instead

Quarter-Kelly (0.25) and 5% max bankroll cap already exist — no change needed.

---

## Goals

- [x] **8A. Tighter bet filters** — raise min edge, add max odds cap
- [x] **8B. Serve stat features** — rolling averages in PlayerState → feature vector (40 → 54 features)
- [x] **8C. Fatigue feature** — days_since_last_match (more granular than weeks_inactive)
- [ ] **8D. Validate** — rolling backtest shows improved ROI and win rate

---

## Phase 8A — Tighter Bet Filters

### Problem
`min_ev = 0.02` (2%) is too loose. Bets on underdogs with tiny edges over high-odds lines.
No odds cap means system bets on avg odds 3.73 (market implies 27% win).

### Changes

**`src/betting/paper_trader.py` — add to `PaperTraderConfig`:**
```python
min_edge: float = 0.05    # model_prob - (1/odds) must exceed this
max_odds: float = 3.0     # don't bet heavy underdogs
```

Bet placement filter (currently ~lines 289-318):
```python
implied_prob = 1.0 / odds
edge = model_prob - implied_prob
if edge < config.min_edge or odds > config.max_odds:
    skip
```

**`src/evaluation/backtester.py` — add to `BacktestConfig`:**
```python
min_edge: float = 0.05
max_odds: float = 3.0
```

**`scripts/backtest.py` — env vars:**
```python
MIN_EDGE = float(os.environ.get("MIN_EDGE", "0.05"))
MAX_ODDS = float(os.environ.get("MAX_ODDS", "3.0"))
```

---

## Phase 8B — Serve Stat Rolling Features

### Problem
40-feature vector has zero serve stats. Sackmann has per-match serve data; need
rolling (EMA) player averages computed during state update in feature_store.py.

### New `PlayerState` fields (`src/features/feature_store.py`)

```python
# Serve stats — EMA (alpha=0.15 ≈ 7-match half-life)
first_serve_pct: float = 0.60       # 1stIn / svpt
first_serve_win_pct: float = 0.72   # 1stWon / 1stIn
second_serve_win_pct: float = 0.50  # 2ndWon / (svpt - 1stIn)
bp_save_pct: float = 0.65           # bpSaved / bpFaced
ace_rate: float = 0.07              # ace / svpt
df_rate: float = 0.04               # df / svpt
serve_games: int = 0                # matches with valid serve data
```

### State update (after each match, both `build()` and `build_training_matrix()`)

```python
svpt = row.get("player_svpt", 0) or 0
if svpt > 0:
    alpha = 0.15
    first_in = row.get("player_1stIn", 0) or 0
    first_won = row.get("player_1stWon", 0) or 0
    second_won = row.get("player_2ndWon", 0) or 0
    bp_saved = row.get("player_bpSaved", 0) or 0
    bp_faced = row.get("player_bpFaced", 0) or 0
    ace = row.get("player_ace", 0) or 0
    df = row.get("player_df", 0) or 0
    second_in = svpt - first_in

    ps.first_serve_pct = alpha * (first_in / svpt) + (1-alpha) * ps.first_serve_pct
    ps.first_serve_win_pct = alpha * (first_won / first_in if first_in > 0 else ps.first_serve_win_pct) + (1-alpha) * ps.first_serve_win_pct
    ps.second_serve_win_pct = alpha * (second_won / second_in if second_in > 0 else ps.second_serve_win_pct) + (1-alpha) * ps.second_serve_win_pct
    ps.bp_save_pct = alpha * (bp_saved / bp_faced if bp_faced > 0 else ps.bp_save_pct) + (1-alpha) * ps.bp_save_pct
    ps.ace_rate = alpha * (ace / svpt) + (1-alpha) * ps.ace_rate
    ps.df_rate = alpha * (df / svpt) + (1-alpha) * ps.df_rate
    ps.serve_games += 1
```

Column names in enriched CSV: `player1_ace`, `player1_svpt`, `player1_1stIn`, etc.
When iterating, prefix varies by winner/loser perspective — handle via the existing
winner/loser → player1/player2 remapping already in place.

### New features appended to `_feats_from_states()` (positions 41-54)

```python
ps1.first_serve_pct, ps1.first_serve_win_pct, ps1.second_serve_win_pct,
ps1.bp_save_pct, ps1.ace_rate, ps1.df_rate, float(ps1.serve_games),
ps2.first_serve_pct, ps2.first_serve_win_pct, ps2.second_serve_win_pct,
ps2.bp_save_pct, ps2.ace_rate, ps2.df_rate, float(ps2.serve_games),
```

Feature count: 40 → 54.

---

## Phase 8C — Fatigue Feature

`weeks_inactive` = days/7 loses intra-week precision. Replace with `days_rest` (capped at 30).

In `_feats_from_states()`, replace `ps1.weeks_inactive` / `ps2.weeks_inactive`:
```python
p1_days_rest = min((match_date - ps1.last_match_date).days, 30) if ps1.last_match_date else 30
p2_days_rest = min((match_date - ps2.last_match_date).days, 30) if ps2.last_match_date else 30
```

Feature count stays 54 (replacing existing slots, not adding).
Remove `weeks_inactive` from `PlayerState` if no longer used (or keep for compatibility).

---

## Files to Modify

| File | Change |
|------|--------|
| `src/features/feature_store.py` | Add serve stat fields + EMA update in build() + build_training_matrix(); add 14 features to _feats_from_states(); replace weeks_inactive with days_rest |
| `src/evaluation/backtester.py` | Add min_edge + max_odds to BacktestConfig; apply filters in run() |
| `src/betting/paper_trader.py` | Add min_edge + max_odds to PaperTraderConfig; apply in bet placement |
| `scripts/backtest.py` | Wire MIN_EDGE + MAX_ODDS env vars |
| `scripts/paper_trade.py` | Wire min_edge + max_odds from env |
| `tests/test_mocks.py` | Update feature count assertions (40 → 54) |

---

## Execution Order

1. Phase 8A — bet filters (fast, config + logic changes)
2. Phase 8B — serve stats (feature_store.py — biggest change)
3. Phase 8C — days_rest (small replacement)
4. Delete `paper_model.joblib` (feature count changed)
5. Run rolling backtest — compare ROI
6. Update tests

---

## Verification

```bash
# Rolling backtest (compare to prior)
poetry run python scripts/rolling_backtest.py

# Full backtest with new filters
MIN_EDGE=0.05 MAX_ODDS=3.0 poetry run python scripts/backtest.py

# All tests
poetry run pytest -x -q

# End-to-end predict
ODDS_API_KEY=4b981035fb7f1c35cec84248ee08c39e \
  poetry run python scripts/paper_trade.py predict
```

## Success Criteria
- Rolling backtest ROI improves vs. prior run
- Bet win rate ≥ 55% (was ~43%)
- Fewer total bets placed (edge filter working)
- All tests pass
- `paper_trade.py predict` works with new 54-feature vector
