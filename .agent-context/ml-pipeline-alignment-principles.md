# ML Pipeline Alignment Principles

Hard-won lessons from five bugs that reduced a 93% accurate model to 49% win-rate predictions.
Each principle maps to a concrete failure that occurred in `scripts/backtest.py`.

---

## 1. The DataFrame Alignment Contract

Any function that returns a numpy array derived from a DataFrame must document the row ordering explicitly. Callers that pair the returned array with other per-match data **must use the same frozen (sorted + reset) DataFrame**, not the original.

**Violated by Bug 5** (`_prepare_features` sorted internally; `match_odds` received the original unsorted DataFrame, so `probs[i]` was assigned to the wrong match — silent, random-walk predictions).

**Rules:**
- Document sort order in the function's docstring.
- Name the frozen copy explicitly: `test_sorted = test_raw.sort_values(...).reset_index(drop=True)`.
- Add an assertion at the consumer: `assert probs.shape[0] == len(test_df)`.
- Add a sort-order assertion if the column is available: `assert (dates[:-1] <= dates[1:]).all()`.

---

## 2. sklearn `predict_proba` Column Convention

`predict_proba` returns shape `(n, 2)` where **column k = P(class k)**. With label encoding `y = game_winner - 1`, class 0 = player1 wins → `probs[:, 0]` = P(player1 wins).

**Violated by Bug 3** (`probs[:, 1]` was used as P(player1 wins), inverting every prediction).

**Diagnostic signature**: Model accuracy is unaffected (argmax is symmetric under column swap), but EV and Kelly sizing are completely wrong. A 93% accurate model becomes a 49% bettor.

**Rules:**
- Always use `probs[:, 0]` for P(player1 wins) in downstream EV/Kelly calculations.
- Document in every `predict_proba` docstring: `Column 0 → P(class 0) = P(player1 wins)`.
- Write a test that verifies column 0 is high when input features match a clear class-0 example.

---

## 3. Label Encoding Contract

The label column (`game_winner`) must be extracted **before** any drop loop. It must appear in exactly one place: the extraction statement.

**Violated by Bug 1** (`game_winner` was in `_DROP_EXACT` and then `df.pop("game_winner")` was called after the loop, raising `KeyError`).

**Rules:**
- Extract `y` first, before any column-drop code: `y = df["game_winner"].values - 1`.
- Keep the label column out of all drop lists (use `df.drop(columns=["game_winner"], errors="ignore")` as a separate cleanup step).
- Assert that the label is absent from the feature matrix before returning: `assert "game_winner" not in feat_cols`.

---

## 4. Probability Invariant in Swapped-Name Joins

After `match_odds()`, every row satisfies:

```
p1_win_prob = P(player1 wins)
p1_odds     = Pinnacle odds for player1 to win
```

This invariant holds **regardless of Case A or Case B**. Do **not** flip `p1_win_prob` in Case B.

**Violated by Bug 2** (`p1_win_prob = 1 - p1_win_prob` was applied in Case B, causing the backtester to price bets with winner's probability against loser's odds — systematic loss).

**How Case B works:**
- Case B: ATP player1 surname matches the odds file's *loser* (names are swapped in the odds data).
- The model's `p1_win_prob` correctly reflects P(player1 wins). Player1 is the underdog, so this value is low.
- `p1_odds` = PSL (loser's odds, higher value). The EV formula handles this correctly without flipping.

---

## 5. Name-Matching Joins Are Silent Degraders

Fuzzy surname joins on `(year, p1_surname, p2_surname)` produce false-positive matches when two different matches in the same year happen to share a player's surname.

**Violated by Bug 4** (11% of Case A rows had `game_winner = 2`, meaning the wrong match was pulled in because player1's surname matched the odds winner but the match outcome disagreed).

**Rules:**
- Every fuzzy join must include a post-join ground-truth consistency check.
- Case A (ATP p1 = odds winner): filter `ma = ma[ma["game_winner"] == 1]`.
- Case B (ATP p1 = odds loser): filter `mb = mb[mb["game_winner"] == 2]`.
- Deduplicate on sorted surnames to prevent betting both orderings of the same physical match.
- Log the match rate and collision rate on every run to detect drift.

---

## 6. Integration Tests Are Required at Handoff Points

Unit tests of individual components don't catch boundary bugs. The five bugs above all lived at **interfaces between functions**, not inside any single function.

**Rule:** For every `(function A produces X, function B consumes X)` pair, write a test that:
1. Calls A to produce X.
2. Passes X to B.
3. Asserts the semantics of the output (not just that it runs).

Example from Bug 5:
```python
# Don't just test that match_odds runs — test that the probability lands on the right player.
result = match_odds(test_sorted, odds, probs)
smith_rows = result[result["p1_sn"] == "smith"]
assert smith_rows.iloc[0]["p1_win_prob"] == pytest.approx(0.90)
```

---

## 7. Test Data Name Format Contract

ATP database and odds files use different name formats. Tests using both must match this convention:

| Source | Format | Surname extraction |
|--------|--------|--------------------|
| ATP database (`player1_name`) | `"First Surname"` | Last token: `str.split().str[-1]` |
| tennis-data.co.uk (`p1_name`) | `"Surname I."` | First token: `str.split().str[0]` |

**Common test mistake**: Using `"Smith J"` for an ATP name — last token = `"J"`, not `"Smith"`. Use `"John Smith"` instead.

---

## Summary Table

| Bug | Root cause | Prevention |
|-----|------------|------------|
| 1 | Label extracted after drop | Extract `y` before all drops; assert it's absent from features |
| 2 | Case B prob flip | Never flip `p1_win_prob`; document the invariant |
| 3 | Wrong probability column | Always use `probs[:, 0]`; test column convention directly |
| 4 | Name-collision mismatches | Post-join `game_winner` consistency filter; log collision rate |
| 5 | Row-order mismatch (critical) | Explicit sorted copy; shape/order assertions at consumer |
