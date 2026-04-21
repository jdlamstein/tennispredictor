# Plan: Code Review, Cleanup & Bug Fixes

| Field   | Value |
|---------|-------|
| Created | 2026-04-21 |
| Updated | 2026-04-21 |
| Status  | completed |
| Commit  | 66e7ada |
| Jira    | N/A |

## Context

Three-agent audit of the full pipeline surfaced 2 critical bugs, several high-severity issues, dead code,
and test coverage gaps. The most impactful bug — inverted probability columns in `TemperatureScaling` —
directly explains the observed 33-42% XGBoost win rate: calibration corrupts probabilities before
they're used to size bets.

---

## Bugs Found — Ordered by Severity

### CRITICAL

**C1. `calibration.py:115-116, 150-151` — TemperatureScaling column inversion**

```python
# WRONG (current)
log_p1 = np.log(np.clip(probs[:, 1], 1e-12, 1.0))  # probs[:,1] is p2_win
log_p2 = np.log(np.clip(probs[:, 0], 1e-12, 1.0))  # probs[:,0] is p1_win

# CORRECT
log_p1 = np.log(np.clip(probs[:, 0], 1e-12, 1.0))  # probs[:,0] = P(player1 wins)
log_p2 = np.log(np.clip(probs[:, 1], 1e-12, 1.0))  # probs[:,1] = P(player2 wins)
```

Class encoding: `y = game_winner - 1` → `y=0` means player1 wins → sklearn `probs[:,0]` = P(player1 wins).
Same inversion exists in `transform()` method at lines 150-151. Fix both.

**C2. `test_mocks.py` — calibration tests validate wrong behavior**

Tests pass today because they test the inverted code. After C1 fix, these tests will fail.
Must update `test_mocks.py` to use correct column indices (0 for p1_win, 1 for p2_win).

---

### HIGH

**H1. `paper_trader.py:430-433` — `total_staked` never updated in daily summary**

`ts` is initialized from existing row but new stakes from the current cycle are never added.
Result: `daily_summary.total_staked` undercounts. Fix: accumulate new stakes into `ts`.

**H2. `feature_store.py:430, 443` — duplicate `year` column**

`_feats_from_states()` returns `float(year)` at both position 27 and position 40 (`year_col`).
These are identical. Wastes one of 40 feature slots. Remove the duplicate (drop `year_col`).
Must verify feature count remains 40 after removal — if intentional, replace with a different feature.

**H3. `recent_matches` logic — sequential checker, not rolling window**

Same bug in both `build_database.py:253-256` and `feature_store.py:323-327`:
```python
if ps.last_match_date and ps.last_match_date >= two_weeks_ago:
    ps.recent_matches += 1
else:
    ps.recent_matches = 1  # wrong — resets to 1 every time
```
This increments only if the PREVIOUS match was ≤ 14 days ago; cannot count 3+ matches in a window.
**Must fix both files identically** — training and inference must use same logic.
Fix: maintain a deque of match dates; count how many fall within 14 days.

---

### MEDIUM

**M1. `paper_trade.py:165` — dead import + unused config**

`blend_with_market` imported but never called. `MARKET_ALPHA` env var (line 62) never used.
Remove both. If blending is desired, wire it in deliberately.

**M2. Inconsistent `MARKET_ALPHA` defaults across scripts**

`scripts/backtest.py:64` → `MARKET_ALPHA = 1.0` (pure model)
`scripts/paper_trade.py:62` → `MARKET_ALPHA = 0.3`
Same parameter, different defaults. Pick one and document why.

**M3. `results_fetcher.py:139-153` — `_from_uts()` stub**

Returns `[]` with a warning. Settlement silently returns zero results if Sackmann fetch fails.
Acceptable short-term; document explicitly that settlement requires Sackmann availability.

**M4. `paper_trader.py:348-352` — feature_builder exceptions silently disable model**

Any exception from `feature_builder(m)` → returns `None` → falls back to market-only mode.
Silent data loss. At minimum, log the exception type + traceback so failures are visible.

**M5. `backtester.py:134` — `actual_winner` not validated**

`actual_winner` is cast to int but never checked against `{1, 2}`. Out-of-range values pass silently.
Add: `assert actual_winner in (1, 2), f"Unexpected actual_winner={actual_winner}"`.

**M6. `paper_trader.py:399-408` — name matching fragile**

`_sn(name)` splits on space and takes last token. Breaks on double-spaces or unusual formats.
Add a `.strip()` guard.

---

### LOW

**L1. `calibration.py:93-131` — no input shape validation in `fit()`**

No check that `probs.shape[0] == len(outcomes)`. Add assertion at entry point.

**L2. `odds_fetcher.py:192` — Betfair market filter `.upper()` may break API**

`"Match Odds".replace(" ", "_").upper()` → `"MATCH_ODDS"`. Verify Betfair expects this exact format.

---

## Test Coverage Gaps

| Module | Gap |
|--------|-----|
| `src/models/classifiers.py` | No dedicated tests |
| `src/models/mlp.py` | No dedicated tests |
| `src/evaluation/metrics.py` | No dedicated tests |
| `src/data/odds_loader.py` | Only indirectly tested via mocks |

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/evaluation/calibration.py` | Fix probs column inversion (C1) + add shape validation (L1) |
| `tests/test_mocks.py` | Fix calibration tests to use correct columns (C2) |
| `src/betting/paper_trader.py` | Fix total_staked (H1), improve feature_builder error logging (M4), name match guard (M6) |
| `src/features/feature_store.py` | Remove duplicate year_col (H2), fix recent_matches (H3) |
| `scripts/build_database.py` | Fix recent_matches (H3 — must match feature_store.py) |
| `scripts/paper_trade.py` | Remove dead blend_with_market import + MARKET_ALPHA (M1) |
| `src/evaluation/backtester.py` | Add actual_winner validation (M5) |
| `tests/` | Add coverage for classifiers, metrics, odds_loader |

---

## Execution Order

1. Fix C1 (calibration inversion) — unblocks correct XGBoost predictions
2. Fix C2 (update calibration tests to match corrected behavior)
3. Fix H1 (daily_summary total_staked)
4. Verify H2 (duplicate year) — read `_feats_from_states` before deleting
5. Fix H3 (recent_matches) — update both files identically
6. Fix M1–M6 in one pass
7. Add missing test coverage
8. Run full test suite; confirm win rate ≥ 50% in backtest

---

## Verification

```bash
# All tests pass
poetry run pytest -x -q

# Calibration fix — backtest win rate should be 50–60% not 33-42%
poetry run python scripts/backtest.py

# Rolling backtest ROI should be unchanged or improved
poetry run python scripts/rolling_backtest.py

# Paper trade cycle works end-to-end
ODDS_API_KEY=4b981035fb7f1c35cec84248ee08c39e \
  poetry run python scripts/paper_trade.py predict
```
