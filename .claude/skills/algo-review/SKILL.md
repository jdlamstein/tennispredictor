---
name: algo-review
description: >
  Deep logic audit of the tennis betting algorithm. Checks mathematical
  correctness, look-ahead bias, training/inference consistency, and betting
  invariants. Use when the user says "review the algorithm", "check the logic",
  "algo review", "/algo-review", or asks to verify the betting math is sound.
argument-hint: "[file or component to review]"
---

You are auditing the tennis prediction + betting pipeline for logical correctness.
This is not a style review — focus exclusively on mathematical errors, data leakage,
and invariant violations that would corrupt predictions or P&L.

## Project Invariants

These are the ground-truth rules for this codebase. Violations are bugs, not preferences.

### Label & Column Convention
- `y = game_winner - 1` → `y=0` means **player1 wins**, `y=1` means player2 wins
- `probs[:, 0]` = P(class 0) = **P(player1 wins)** — never use `probs[:, 1]` for p1
- `actual_winner ∈ {1, 2}` in raw data; `y ∈ {0, 1}` in model training
- Violation pattern: any code that indexes `probs[:, 1]` as "p1 wins" or `probs[:, 0]` as "p2 wins"

### Math Formulas
- **Kelly**: `f = (p*b - q) / b` where `b = odds - 1`, `q = 1 - p`; result is fraction of bankroll
- **EV**: `EV = p*(odds-1) - (1-p)` = `p*b - q`; positive = bet has value
- **Edge**: `edge = model_prob - 1/odds` (implied prob from decimal odds); used for min_edge filter
- **Devig**: `p1_true = (1/p1_odds) / (1/p1_odds + 1/p2_odds)`; removes bookmaker margin
- **ELO update**: K-factor changes with experience; winner ELO must increase, loser decrease
- Violation pattern: dividing instead of subtracting in Kelly, using raw odds as probability, devig applied to already-devigged probs

### Look-Ahead Bias (Critical)
No feature visible at inference time may encode information from after the match starts.

- **Feature capture order**: PRE-match state → capture features → THEN update state with result
- **Raw serve stats** (`w_ace`, `w_svpt`, `w_1stIn`, `w_1stWon`, `w_2ndWon`, `w_SvGms`, `w_bpSaved`, `w_bpFaced`) are POST-match — must be dropped from feature matrix or converted to rolling EMA
- **Raw ELO** (`player1_elo`, `player2_elo` in base CSV) is stored POST-match — must drop; use surface-specific ELOs from enriched CSV which are PRE-match
- **Rankings** (`player1_rank`) reflect known-at-match-time data — currently dropped as precaution
- **EMA update ordering**: write current (pre-match) EMA value to output array FIRST, then update state
- Violation pattern: state update before feature capture; using match-level stats directly as features

### Training / Inference Consistency
The model's training features must exactly match inference features, or predictions are garbage.

- `FeatureStore.build()` and `FeatureStore.build_training_matrix()` must update **identical state fields** after each match — any new field added to one must be in both
- `_feats_from_states()` is the single source of truth for the feature vector — both paths call it
- Scaler must be **fit** on training set only; **transform** only at inference
- Feature count in `_feats_from_states()` must match the comment in `build_training_matrix` docstring (currently 54)
- `_DROP_PATTERNS` in `backtest.py` must not accidentally drop EMA columns — check `_ema_` guard
- Violation pattern: state field updated in `build()` but not `build_training_matrix()`; scaler fit on test data; feature count mismatch between train and predict

### Betting Logic
- `min_edge` filter: skip if `model_prob - 1/odds < min_edge` — computed from implied probability, not EV
- `max_odds` filter: skip if `odds > max_odds` — prevents betting heavy underdogs
- Kelly stake must be `>= 0` before placing; negative Kelly = no edge, skip
- Quarter-Kelly: `kelly_fraction = 0.25` is the default; full Kelly risks ruin
- `max_kelly` cap applied after Kelly calculation: `stake = min(kelly_stake, bankroll * max_kelly)`
- Violation pattern: using raw EV as edge proxy; `>=` instead of `>` on max_odds; applying cap before Kelly calculation

### Glicko-2 Period Handling
- Period buffer must be **flushed before** advancing to new period, not after
- `build()` must pass `holdout_year` so future data doesn't leak into live feature store
- Violation pattern: `_flush_glicko()` called at wrong boundary; missing holdout_year

### Surface Map
- `{0: "hard", 1: "clay", 2: "grass", 3: "carpet"}` — must be consistent across all files
- Files that use this: `feature_store.py`, `elo.py`, `build_database.py`
- Violation pattern: surface integers used directly as features without map lookup

---

## Audit Checklist

When reviewing code, systematically check:

1. **Probability column**: every `probs[:, N]` — is N correct for the intended player?
2. **Feature capture order**: in any state-replay loop, does feature capture precede state update?
3. **Dropped columns**: are post-match columns (`_ace`, `_svpt`, etc.) absent from feature matrix?
4. **EMA ordering**: is pre-match value written before EMA update?
5. **Kelly/EV/Edge math**: expand the formula mentally, check sign and denominator
6. **Both build paths**: if state logic changed, is it in both `build()` and `build_training_matrix()`?
7. **Scaler**: is `fit_transform` only on training set, `transform` only on test/inference?
8. **Filter logic**: is `min_edge` computed as `model_prob - 1/odds` (not EV)?
9. **Calibration return**: does `transform()` return `[cal_p1, cal_p2]` not `[cal_p2, cal_p1]`?
10. **Devig**: called on raw odds, not on already-devigged probabilities?

---

## Output Format

Report only findings. No "looks good" for passing checks — silence = pass.

```
[CRITICAL] <file>:<line> — <invariant violated>. <what the code does> vs <what it should do>.
[HIGH]     <file>:<line> — <problem>. <fix>.
[MEDIUM]   <file>:<line> — <risk>. <mitigation>.
```

If no issues: "No invariant violations found in reviewed scope."

Group by severity. For CRITICAL findings, show the offending code snippet and the corrected version.

Do not flag style, naming, comments, or performance unless the user explicitly asks.
If the scope is unclear (user typed `/algo-review` with no args), review:
- `src/evaluation/calibration.py`
- `src/features/feature_store.py`
- `src/betting/paper_trader.py`
- `src/evaluation/backtester.py`
- `src/features/serve_stats.py`
