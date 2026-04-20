# Plan: Profitable Tennis Betting Strategy

| Field   | Value |
|---------|-------|
| Created | 2026-04-15 |
| Updated | 2026-04-18 |
| Status  | in-progress |
| Commit  | 7e1b27c |
| Jira    | N/A |

> **Paid alternatives deferred**: See `.agent-context/possibilities.md` for paid tools
> (The Odds API, Betfair Historical Data, OddsJam) to consider if free sources prove insufficient.
>
> **Package manager**: This project uses **Poetry** (`pyproject.toml` + `poetry.lock`).

## Summary

Turn the existing ATP match prediction model (~93% accuracy, PyTorch MLP + sklearn
classifiers) into a profitable sports betting operation. The core insight is that raw
accuracy is not sufficient — profitability requires calibrated probabilities, EV
calculations against bookmaker odds, proper bet sizing (Kelly Criterion), backtesting
against historical lines, and a live paper-trading system to validate alpha before
risking real capital.

## Context

The model already achieves 93.1% test accuracy using 36 features (ELO ratings, recent
form, head-to-head, demographics, temporal encoding). Key gaps:

- No odds integration — no way to know if predictions beat the market
- No probability calibration — raw logits may be overconfident
- No Kelly Criterion or bankroll management
- No backtesting against actual betting lines
- No live prediction / paper-trading infrastructure
- No standardized benchmark framework to compare models

## Goals

- [ ] Identify the best betting platform(s) for tennis
- [ ] Build a dry-run live prediction system (paper trading)
- [ ] Implement backtesting against historical betting odds
- [ ] Expand the dataset with higher-signal features
- [ ] Research and evaluate algorithm improvements
- [ ] Build a standardized benchmark for comparing models
- [ ] Establish a consistent code style for the project

---

## 1. Best Platform to Place Tennis Bets

### Candidates

| Platform | Limits | Vig | Account Longevity | API | Availability |
|----------|--------|-----|-------------------|-----|--------------|
| **Pinnacle** | $5K–$50K/match | ~2–3% | ✅ Welcomes winners | ✅ Yes | Ex-US/UK/AU |
| **Betfair Exchange** | Market-dependent | 2–5% commission | ⚠️ Market-by-market | ✅ Yes | Most countries |
| **Bet365** | Low after restrictions | 5–8% | ❌ Restricts winners | ❌ No | Most countries |
| **William Hill** | Low after restrictions | 6–10% | ❌ Restricts winners | ❌ No | US + Europe |
| **DraftKings/FanDuel** | Moderate | 7–10% | ❌ Restricts winners | ❌ No | US only |
| **Betway** | Moderate | 5–7% | ❌ Restricts winners | ❌ No | Europe/AU |

### Recommendation: Pinnacle Primary, Betfair Exchange Secondary

**Pinnacle**
- Pros: Best odds in the world, no account restrictions, API for automation, full tennis
  coverage including Challengers/ITF, transparent odds history.
- Cons: Unavailable in US, UK, Australia. No promotions. Requires currency conversion.
- Sources: pinnacleapi.com, pinnaclesports.com/en/betting-resources

**Betfair Exchange**
- Pros: Peer-to-peer pricing often beats Pinnacle on liquid markets; can *lay* outcomes
  (act as bookmaker); full historical data API; opening-line timestamps available.
- Cons: 2–5% commission on net winnings per market; liquidity dries up on Challengers;
  complexity of matched betting; UK Gambling Commission oversight.
- Sources: developer.betfair.com, betfair.com/exchange

**For US-based operation**: Pinnacle via licensed offshore account. Supplement with
DraftKings/FanDuel for bonus-harvesting only (not profitable long-term for sharp play).

### Line Shopping Strategy (Open-Source / Free)
Never bet at only one book. Automate odds comparison using free tools:
- **OddsPortal scraper** (github.com/mdeber/oddsportal-scraper or similar) — scrapes
  historical and upcoming odds from oddsportal.com; free but fragile.
- **Betfair API** — free with account registration; returns live market prices.
- **tennis-data.co.uk** — free CSV downloads with B365 + Pinnacle odds; reliable.
- Paid alternatives (if free sources prove insufficient): see `.agent-context/possibilities.md`.

---

## 2. Dry-Run Live Predictions

### Method A: Automated Paper-Trading Database (Recommended)

Build a lightweight service that runs before each ATP match day:

```
1. Fetch upcoming matches from tennisabstract.com or ATP site scraper
2. Build feature vectors via generate_deploy.py pipeline
3. Run model inference → p_win for each player
4. Fetch current Betfair odds via free Betfair API (betfairlightweight library)
   OR scrape OddsPortal for Pinnacle odds
5. Compute EV: EV = p_model × (odds − 1) − (1 − p_model)
6. Log prediction + EV + Kelly stake to SQLite before match starts
7. After match: fetch result, compute P&L, update running metrics
```

Stack: Python + **APScheduler** (open-source) + **SQLite** + **pandas** + **betfairlightweight**
(open-source Betfair API wrapper) + OddsPortal scraper.

- Pros: Fully automated; free; parallel-tests multiple models; timestamps predictions.
- Cons: Betfair requires a funded account for API access; OddsPortal scraping is fragile;
  prices may not match exact execution.

**Key open-source libraries**:
- `betfairlightweight` — Python Betfair API wrapper (github.com/liampauling/betfairlightweight)
- `APScheduler` — task scheduling (pip install apscheduler)
- `sqlite3` — built into Python stdlib

### Method B: Manual Tracking (Quick Start, Zero Cost)

Fetch daily Betfair prices via the free API or record Pinnacle odds from OddsPortal
manually, log to a CSV file.

- Pros: Zero cost; fastest to start; useful for sanity-checking before automating.
- Cons: Manual; error-prone; doesn't scale beyond a few matches per week.

### Method C: Betfair Virtual Betting via API

`betfairlightweight` supports placing bets in a simulated environment without real money
(using Betfair's test API endpoint). Realistic market simulation.

- Pros: No money at risk; realistic liquidity constraints; free.
- Cons: Requires Betfair account registration; UK/EU preferred.

### Key Metric: Closing Line Value (CLV)

The best early dry-run metric is CLV: did our predicted probability beat the closing
market odds? If `p_model > p_closing_line` consistently, the model has genuine alpha
regardless of short-term variance.

Source: Joseph Peta, "Trading Bases" (2013); blog.mikemaloney.com on CLV theory.

---

## 3. Backtesting Against Known Matches

### Critical Principle: No Look-Ahead Bias

The existing train/val/test split is chronological (60/20/20) — correct. Backtesting
must be strictly chronological: model only trained on data before each prediction.

### Data Source: tennis-data.co.uk (Free, Recommended)

- URL: tennis-data.co.uk/data.php
- Coverage: ATP + WTA, 2000–present
- Includes: B365, Pinnacle, Betway, LB odds for each match
- Format: CSV, one file per year per tour
- Already referenced in codebase (`generate_deploy.py`); needs actual integration.

### Backtesting Pipeline Design

```python
# Pseudocode
for each match in historical_data (chronological order):
    p_model = model.predict_proba(features_before_match)
    odds_available = odds_data[match_id]
    p_implied = 1.0 / odds_available.pinnacle_odds  # de-vig if needed
    ev = p_model * (odds_available - 1) - (1 - p_model)
    if ev > threshold:
        stake = kelly_fraction * bankroll * ev / (odds_available - 1)
        result = match.actual_winner
        bankroll += stake * (odds_available - 1) if correct else -stake
```

### Metrics to Report for Every Backtest

| Metric | Description | Target |
|--------|-------------|--------|
| **ROI** | (Profit / Total staked) × 100 | > 5% long-term |
| **Yield** | ROI per bet | > 3% to be sustainable |
| **Brier Score** | Probability calibration | < 0.2 |
| **Log Loss** | Sharpness of probability estimates | Minimize |
| **CLV** | Beat closing line | > 0 consistently |
| **Sharpe Ratio** | Risk-adjusted returns | > 1.0 |
| **Max Drawdown** | Worst peak-to-trough | < 30% |
| **Win Rate** | % bets won | Context-dependent |

### De-Vigging Odds

```python
implied_p1 = 1 / odds_p1
implied_p2 = 1 / odds_p2
total = implied_p1 + implied_p2          # > 1.0 due to vig
fair_p1 = implied_p1 / total             # removes vig
```

### Bet Sizing: Kelly Criterion

Full Kelly: `f* = (p × b − q) / b` where `b = decimal_odds − 1`, `p = p_model`, `q = 1−p`

Fractional Kelly (recommended): use `f*/4` to `f*/2` — accounts for model uncertainty.

Source: Kelly (1956), "A New Interpretation of Information Rate"; Thorp (2008),
"The Kelly Criterion in Blackjack Sports Betting and the Stock Market."

---

## 4. Dataset Expansion

### Current: ~172K ATP matches, 36 features

### Expansion Options (Priority Order)

**1. Betting Market Odds as Features (Highest Priority)**
- Source: tennis-data.co.uk (free, historical back to 2000) — also enables backtesting
- Add pre-match Pinnacle/B365 log-odds as a feature (markets encode private information).
- Pros: Large information gain; free; solves backtesting data need simultaneously.
- Cons: Risk of circularity; need careful feature engineering (log-odds transform).

**2. Surface-Specific ELO Ratings**
- Split ELO by surface: hard, clay, grass, carpet → 8 new features.
- Source: Kovalchik (2016), "Searching for the GOAT of tennis win prediction."
- Pros: Documented accuracy improvement in literature (~1–2%).
- Cons: Sparse data for players who rarely play certain surfaces.

**3. Glicko-2 Ratings (Finish Existing Stub)**
- Already started in `preprocessing/pipeline.py` (Glicko class stub).
- Glicko-2 adds rating deviation and volatility — better for sparse-data players.
- Pros: Better uncertainty quantification for lower-tier players.
- Cons: Marginal gains over ELO for top-100 ATP players.
- Source: Glickman (2012), "Example of the Glicko-2 system."

**4. Ultimate Tennis Statistics API**
- URL: ultimatetennisstatistics.com — free REST API
- Stats: first serve %, break point conversion %, rally length by surface.
- Pros: Directly captures playing style; free.
- Cons: Only available for recent ATP matches; sparse for lower-ranked players.

**5. WTA Data**
- Jeff Sackmann also maintains `tennis_wta` repository (~80K+ matches).
- Pros: Doubles training data.
- Cons: ATP and WTA are distinct sports; use domain-adapted or separate models.

**6. Fatigue / Travel Data**
- Days since last match, tournament location distance, timezone changes.
- Source: Match date + venue via geopy geocoding.
- Pros: Captures scheduling fatigue effects.
- Cons: Noisy; may already be captured by `weeks_inactive` feature.

**7. ATP Challenger Expansion**
- Already partially included; expand systematically.
- Pros: Less efficient markets = more potential alpha.
- Cons: Less reliable statistics for lower-ranked players.

**8. Injury/Withdrawal Data**
- Source: tennisabstract.com injury database or ATP press releases.
- Pros: Captures key information markets know but statistics don't show.
- Cons: Very difficult to acquire systematically; sparse; potentially unreliable.

---

## 5. Algorithm Research & Improvement

### Current Best: 93.1% MLP ≈ 93.0% Naive Bayes

Note: 93% accuracy on a dataset where favorites win ~65% means the model already learns
well. The critical question is whether it has **calibrated probabilities** — raw logits
may be overconfident (model says 95% but wins only 80%). Calibration is prerequisite to
valid EV calculations.

### Probability Calibration (Do First)

- **Temperature Scaling**: divide logits by learned scalar T (simplest, works well for NNs).
- **Platt Scaling**: fit logistic regression on top of raw logits.
- **Isotonic Regression**: non-parametric monotone calibration.
- **Measure**: Reliability diagrams + Expected Calibration Error (ECE).
- Source: Guo et al. (2017), "On Calibration of Modern Neural Networks" — ICML.

### Model Improvements

**A. Gradient Boosting (XGBoost / LightGBM)**
- Often outperforms deep learning on tabular data with <1M rows.
- Native feature importance; handles missing values without -10 imputation hack.
- Pros: Faster training, interpretable, strong baseline.
- Cons: Less flexible; needs adapter to current pipeline.
- Source: Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System" — KDD.

**B. Market-Informed Ensemble**
- Blend model probability with bookmaker implied probability:
  `p_final = α × p_model + (1−α) × p_market`
- Tune α on validation set. Markets encode private information (injury, lineup).
- Source: Constantinou & Fenton (2013), sports analytics ensemble methods.
- Pros: Often the best approach; simple to implement.
- Cons: Reduces independence of model from market.

**C. Recurrent Architecture for Match Sequence**
- LSTM/Transformer over a player's last N matches (ordered sequence).
- Current model uses aggregate statistics — loses temporal ordering within a form cycle.
- Pros: Richer form representation.
- Cons: Complex; limited data per player for RNN; marginal gain unclear.

**D. Bayesian Neural Network**
- MC Dropout at test time for epistemic uncertainty.
- Better bet sizing by incorporating model uncertainty into Kelly formula.
- Source: Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation."
- Pros: Principled uncertainty; reduces overbetting.
- Cons: Slower inference; complexity.

### Market Efficiency Research

- Grand Slam markets: ~52% profitable bet rate for favorites (efficient).
- ATP 250/500 and Challengers: less bookmaker attention = more edge.
- Opening vs. closing line movement: sharp money moves lines; beating closing line is
  strong evidence of genuine edge.
- Source: Klaassen & Magnus (2003), "Forecasting the Winner of a Tennis Match";
  Forrest & McHale (2006), "Is spread betting on tennis efficient?"

### Recommended Algorithm Roadmap

| Phase | Action | Expected Impact |
|-------|--------|-----------------|
| 1 | Calibrate existing MLP (temperature scaling) | Enables valid EV calculation |
| 2 | Add XGBoost/LightGBM as alternative | Compare on CLV metric |
| 3 | Surface-specific ELO features | +1–2% accuracy documented |
| 4 | Ensemble model + market | Likely best long-term approach |
| 5 | Glicko-2 completion | Marginal gain |
| 6 | LSTM/Transformer for sequence | Experimental |

---

## 6. Benchmarking & Algorithm Switching

### Framework: MLflow (Recommended)

```bash
pip install mlflow
mlflow server --host 127.0.0.1 --port 5000
```

- Tracks: hyperparameters, metrics, model artifacts per experiment run.
- Model registry: tag best model as "Production," swap with one command.
- Comparison UI: side-by-side metric comparison across runs.
- Pros: Free, open-source, integrates with PyTorch/sklearn/XGBoost; complements wandb.
- Cons: Adds infrastructure dependency.

### Standardized Predictor Interface

```python
# src/models/base_predictor.py
from abc import ABC, abstractmethod
import numpy as np

class BasePredictor(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_samples, 2): [p_player2_wins, p_player1_wins]"""
        ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BasePredictor": ...
```

All existing models wrapped to implement this interface. New models implement natively.

### Standardized Evaluation Suite

```python
# src/evaluation/benchmark.py
def evaluate_predictor(
    predictor: BasePredictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    odds_test: np.ndarray,      # Pinnacle odds
) -> dict:
    return {
        "accuracy": ..., "log_loss": ..., "brier_score": ..., "ece": ...,
        "roi": ..., "yield": ..., "sharpe": ..., "max_drawdown": ..., "clv": ...,
    }
```

### Config-Driven Model Selection

```yaml
# configs/experiment.yaml
model:
  type: mlp          # or: xgboost, naive_bayes, ensemble
  checkpoint: latest
  kelly_fraction: 0.25
  min_ev_threshold: 0.02
```

Swap model: `python main/run.py model.type=xgboost`

### Regression Test Suite

Before declaring a new model "better," it must pass:
1. `accuracy >= baseline_accuracy - 1%` (no regression)
2. `roi >= 0%` on holdout odds backtest
3. `brier_score <= baseline_brier`
4. CLV positive on rolling 90-day window

### Testing Philosophy: Real Data, Purposeful Tests

- **No mocks**: All tests run against real match records from the actual ATP dataset
  (a small fixed slice, e.g., 200 matches). We were burned on mocked tests that passed
  but failed on real data (wrong ELO ordering, NaN propagation bugs).
- **Sparse and purposeful**: One test per meaningful invariant, not one per function.
  Test behavior, not implementation. Skip tests for trivial getters/setters.
- **What to test**:
  - ELO ratings are monotonically updated after each match
  - Kelly stake is always ≥ 0 and ≤ bankroll
  - Backtester P&L arithmetic matches manual spot-check on 10 known matches
  - Calibration reduces ECE vs. uncalibrated baseline on a holdout slice
  - Normalization: deploy features use train mean/std, not test set statistics
- **What NOT to test**:
  - That pandas DataFrames have expected column names (fragile)
  - That wandb or MLflow log the right keys
  - Individual getters, trivial computations

---

## 7. Code Development Style

### Current Gaps

- Hardcoded paths in multiple scripts
- Feature engineering classes in one large file (`pipeline.py`)
- No type hints
- No unit tests
- `-10` as NaN sentinel (fragile)

### Recommended Project Structure

```
tennispredictor/
├── src/
│   ├── data/
│   │   ├── loader.py           # Load raw CSVs (one class per source)
│   │   ├── cleaner.py          # ATP class → pure functions + thin class
│   │   └── odds_loader.py      # tennis-data.co.uk integration (NEW)
│   ├── features/
│   │   ├── elo.py              # Elo class (extracted from pipeline.py)
│   │   ├── glicko.py           # Glicko2 class (extracted + completed)
│   │   ├── recent_form.py      # RecentMatches class (extracted)
│   │   ├── temporal.py         # TimePeriod class (extracted)
│   │   └── pipeline.py         # Thin orchestrator only
│   ├── models/
│   │   ├── base_predictor.py   # ABC interface (NEW)
│   │   ├── mlp.py              # PyTorch Lightning wrapper (refactored)
│   │   ├── classifiers.py      # sklearn wrappers (refactored)
│   │   └── xgboost_model.py    # XGBoost wrapper (NEW)
│   ├── evaluation/
│   │   ├── metrics.py          # All metric functions (NEW)
│   │   ├── backtester.py       # Backtesting engine (NEW)
│   │   └── calibration.py      # Temperature scaling etc. (NEW)
│   ├── betting/
│   │   ├── kelly.py            # Kelly Criterion sizing (NEW)
│   │   ├── paper_trader.py     # Dry-run tracking (NEW)
│   │   └── odds_fetcher.py     # Betfair API + OddsPortal scraper (free; NEW)
│   └── pipeline/
│       ├── train_pipeline.py   # End-to-end training
│       └── predict_pipeline.py # End-to-end inference
├── configs/
│   └── experiment.yaml
├── tests/
│   ├── conftest.py             # Shared real-data fixtures (small ATP CSV slice)
│   ├── test_features.py        # ELO/Glicko invariants, streak logic
│   ├── test_backtester.py      # P&L arithmetic on 10 known matches
│   └── test_kelly.py           # Stake bounds, bankroll conservation
├── .plans/
└── requirements.txt
```

### Style Rules

| Rule | Detail |
|------|--------|
| Type hints | All functions; use `np.ndarray`, `pd.DataFrame` explicitly |
| Immutability | Never mutate DataFrames in-place; always return new copies |
| Functions ≤ 40 lines | Extract helpers if needed |
| Files ≤ 400 lines | `pipeline.py` is a target for splitting |
| No global state | Config via constructor args or injected config object |
| Constants in `param_tennis.py` | Migrate to YAML configs over time |
| Error handling | Raise `ValueError` with descriptive messages for bad inputs |
| Docstrings | NumPy format on all public functions |
| Tests | pytest; sparse and purposeful; real ATP data slices (no mocks); cover invariants not implementation |
| Logging | `logging` module (not `print`); structured log levels |
| NaN handling | Use `np.nan` + `pd.isna()`; remove `-10` sentinel |

### Naming Conventions

```python
def compute_kelly_stake(p: float, odds: float, fraction: float = 0.25) -> float: ...
class EloRatingEngine: ...
MIN_MATCHES_FOR_ELO = 10
```

- Pros: High cohesion, easy to find code, simple unit testing of individual components,
  each file is reviewable in one sitting.
- Cons: More files; initial refactor effort; some abstraction overhead.

---

## Approach

Four implementation phases, each building on the previous:

### Phase 0 — Foundations
1. Migrate this plan to `.plans/2026-04/` (done)
2. Migrate package management from `requirements.txt` to **Poetry** (`pyproject.toml`)
3. Download tennis-data.co.uk historical odds CSVs (2000–2025) — free
4. Add `BasePredictor` ABC; wrap existing MLP + classifiers
5. Add MLflow alongside existing wandb

### Phase 1 — Calibration + Backtesting
1. Implement temperature scaling calibration
2. Build `src/evaluation/backtester.py` with tennis-data.co.uk odds
3. Compute full metric suite on historical data
4. Implement Kelly Criterion (`src/betting/kelly.py`)

### Phase 2 — Data Expansion
1. Surface-specific ELO (split by hard/clay/grass)
2. Complete Glicko-2 implementation
3. Pull Ultimate Tennis Statistics serve/rally features
4. Add Pinnacle log-odds as input feature

### Phase 3 — Algorithms
1. Add XGBoost/LightGBM predictor
2. Build market-informed ensemble (model + odds blend)
3. Run full benchmark suite comparing all models

### Phase 4 — Live System
1. APScheduler paper-trading daemon
2. The Odds API integration for live prices
3. Automated daily prediction + result logging
4. Dashboard: CLV, running P&L, ROI

---

## Tasks

- [x] Phase 0: Migrate `requirements.txt` to Poetry (`pyproject.toml` + `poetry.lock`)
- [x] Phase 0: Create `tests/conftest.py` with a real 200-row ATP data fixture
- [x] Phase 0: Create `src/models/base_predictor.py` ABC
- [x] Phase 0: Wrap MLP and classifiers to `BasePredictor`
- [x] Phase 0: Download tennis-data.co.uk odds CSVs (free) — atp_2010–2025.xlsx present, odds_loader.py + download_odds.py implemented
- [ ] Phase 0: Set up MLflow server (DEFERRED — add when running hyperparameter grid searches)
- [x] Phase 1: Implement temperature scaling in `src/evaluation/calibration.py`
- [x] Phase 1: Implement `src/evaluation/backtester.py`
- [x] Phase 1: Implement `src/evaluation/metrics.py` (ROI, Brier, CLV, Sharpe, drawdown)
- [x] Phase 1: Implement `src/betting/kelly.py`
- [x] Phase 1: Implement `src/data/odds_loader.py` (download + parse tennis-data.co.uk)
- [x] Phase 2: Surface-specific ELO in `src/features/elo.py`
- [x] Phase 2: Complete `src/features/glicko.py`
- [x] Phase 2: `scripts/enrich_features.py` to build `atp_database_enriched.csv` (+14 cols, 48 features)
- [x] Phase 2: `src/data/odds_loader.py` for tennis-data.co.uk
- [x] Phase 3: `src/models/xgboost_model.py`
- [x] Phase 3: Market-informed ensemble (`blend_with_market`, MARKET_ALPHA env var; optimal α=0.3)
- [x] Phase 4: `src/betting/paper_trader.py` with APScheduler
- [x] Phase 4: `src/betting/odds_fetcher.py` (Betfair API via `betfairlightweight` + OddsPortal scraper)
- [x] Phase 4: `src/data/results_fetcher.py` (Sackmann GitHub → daily settlement)
- [x] Phase 4: `scripts/paper_trade.py` CLI (start | predict | settle | summary)
- [x] Phase 4: `src/features/feature_store.py` — live ELO/Glicko-2 replay, name resolution, feature vectors
- [x] Phase 4: Wire FeatureStore into `paper_trade.py` (model-based predictions when ATP_DB available)
- [x] Phase 4: `tests/test_feature_store.py` — 32 tests for PlayerState, ELO updates, _resolve, make_features
- [x] Phase 5: CLV via MaxW/MaxL closing odds in `match_odds()` — `p1_closing_odds`/`p2_closing_odds` propagated for Case A/B
- [x] Phase 5: Model + scaler persistence — joblib cache in `paper_trade.py` (`MODEL_CACHE_PATH` env var)
- [x] Phase 5: `scripts/rolling_backtest.py` — walk-forward 2019–2024 with per-year table + aggregate
- [x] Phase 5: `scripts/sensitivity_analysis.py` — Kelly × MIN_EV × MARKET_ALPHA grid search
- [x] Phase 6: `scripts/plot_backtest.py` — equity curve + drawdown (matplotlib); `--rolling` flag for bar chart
- [x] Phase 5/6: `tests/test_backtester.py` CLV tests (5 new), `tests/test_rolling_backtest.py` (7 new), `tests/test_paper_trade_persistence.py` (7 new)

## Files Changed

New files to create:
- `src/models/base_predictor.py`
- `src/evaluation/metrics.py`
- `src/evaluation/backtester.py`
- `src/evaluation/calibration.py`
- `src/betting/kelly.py`
- `src/betting/paper_trader.py`
- `src/betting/odds_fetcher.py`
- `src/data/odds_loader.py`
- `src/models/xgboost_model.py`
- `configs/experiment.yaml`
- `tests/conftest.py` (shared real-data fixture)
- `tests/test_features.py`
- `tests/test_backtester.py`
- `tests/test_kelly.py`

Files to refactor:
- `preprocessing/pipeline.py` → split into `src/features/elo.py`, `glicko.py`,
  `recent_form.py`, `temporal.py`
- `main/classifier.py` → wrapped by `src/models/classifiers.py`
- `main/train.py` → wrapped by `src/models/mlp.py`

## Open Questions

- What jurisdiction is the user in? (Determines Pinnacle vs. Betfair as primary.)
- Real money or purely academic? (Determines urgency of live paper-trading system.)
- Budget for The Odds API subscription (~$50/month)?
- Should WTA data be included or ATP-only focus?

## Notes

Additional context, learnings, or references.

## Key Sources

- Jeff Sackmann tennis_atp dataset: github.com/JeffSackmann/tennis_atp
- tennis-data.co.uk historical odds: tennis-data.co.uk/data.php
- Ultimate Tennis Statistics API: ultimatetennisstatistics.com
- Betfair Exchange API: developer.betfair.com
- betfairlightweight (open-source): github.com/liampauling/betfairlightweight
- OddsPortal: oddsportal.com (free scraping)
- Pinnacle API docs: pinnacleapi.com
- Kelly (1956): "A New Interpretation of Information Rate" — Bell System Technical Journal
- Thorp (2008): "The Kelly Criterion in Blackjack Sports Betting and the Stock Market"
- Guo et al. (2017): "On Calibration of Modern Neural Networks" — ICML 2017
- Kovalchik (2016): "Searching for the GOAT of tennis win prediction" — Journal of Quantitative Analysis in Sports
- Klaassen & Magnus (2003): "Forecasting the Winner of a Tennis Match" — J. Royal Statistical Society
- Forrest & McHale (2006): "Is spread betting on tennis efficient?"
- Glickman (2012): "Example of the Glicko-2 system" — glicko.net/glicko/glicko2.pdf
- Constantinou & Fenton (2013): "Solving the problem of inadequate scoring rules" — J. Quantitative Analysis in Sports
- Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System" — KDD 2016
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
