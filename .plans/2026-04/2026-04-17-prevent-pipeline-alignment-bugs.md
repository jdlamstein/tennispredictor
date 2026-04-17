# Plan: Prevent Prediction Pipeline Alignment Bugs

| Field   | Value |
|---------|-------|
| Created | 2026-04-17 |
| Updated | 2026-04-17 |
| Status  | completed |
| Commit  | c9969d3 |
| Jira    | N/A |

## Summary

Five bugs in `scripts/backtest.py` caused a model with 92–96% holdout accuracy to produce
a 49% win rate in backtesting — worse than random. All bugs lived at **boundaries between
pipeline components**, invisible to unit tests. This plan documents how to prevent recurrence.

## Context

Root causes:
1. **Bug 1** — `game_winner` dropped before label extraction → `KeyError`
2. **Bug 2** — Case B probability flip → betting at loser's odds with winner's probability
3. **Bug 3** — `probs[:, 1]` used instead of `probs[:, 0]` → all predictions inverted
4. **Bug 4** — No name-collision filter → 11% of rows had wrong actual_winner
5. **Bug 5** — Row-order mismatch → `probs[i]` assigned to wrong match (critical)

After fixes: Naive Bayes 90.8% win rate, XGBoost 95.2% win rate.

## Goals

- [x] Write 14 regression tests that catch each bug before it reaches production
- [x] Fix 3 wrong `predict_proba` docstrings in base_predictor, classifiers, mlp
- [x] Add ordering-contract docstrings to `_prepare_features()` and `match_odds()`
- [x] Add 5 runtime assertions at critical pipeline handoff points
- [x] Document ML pipeline alignment principles in `.agent-context/`

## Files Changed

| File | Change |
|------|--------|
| `tests/test_backtest_pipeline.py` | New — 14 regression tests across 4 test classes |
| `scripts/backtest.py` | Docstrings, assertions, empty-DataFrame guard in dedup |
| `src/models/base_predictor.py` | Fixed wrong `predict_proba` docstring |
| `src/models/classifiers.py` | Fixed wrong `predict_proba` docstring |
| `src/models/mlp.py` | Fixed wrong `predict_proba` docstring |
| `.agent-context/ml-pipeline-alignment-principles.md` | New — design principles |

## Notes

Full test suite: 58 tests, all passing.
Design principles documented in `.agent-context/ml-pipeline-alignment-principles.md`.
See bug catalogue in `tests/test_backtest_pipeline.py` module docstring for per-bug test mapping.
