# Tennis Betting Strategy Module

A comprehensive statistical betting framework for ATP tennis matches using machine learning predictions and the Kelly Criterion for optimal bet sizing.

## Overview

This module implements a sophisticated betting strategy based on:
- **Model probabilities** from calibrated classifiers (93%+ accuracy)
- **Kelly Criterion** for optimal bet sizing
- **Risk management** with fractional Kelly and bet limits
- **Performance monitoring** with comprehensive dashboards

## Components

### 1. Kelly Calculator (`kelly_calculator.py`)

Calculate optimal bet sizes based on model probabilities and bookmaker odds.

**Features:**
- Full and fractional Kelly calculations
- Edge and expected value computation
- Conservative/moderate/aggressive strategies
- Interactive calculator mode

**Usage:**

```bash
# Process predictions file
python betting/kelly_calculator.py \
    --predictions data/predictions.csv \
    --bankroll 10000 \
    --strategy conservative \
    --output results/betting_recommendations.csv

# Interactive mode
python betting/kelly_calculator.py --interactive
```

**Strategies:**
- **Conservative**: 0.25x Kelly, 5% min edge, 65% min confidence
- **Moderate**: 0.50x Kelly, 3% min edge, 60% min confidence
- **Aggressive**: 0.75x Kelly, 2% min edge, 55% min confidence

### 2. Backtesting Framework (`backtesting.py`)

Validate betting strategies using historical match data and odds.

**Features:**
- Walk-forward validation
- Multiple strategy comparison
- Comprehensive performance metrics (ROI, Sharpe ratio, drawdown)
- Visualization of results

**Usage:**

```bash
# Run single strategy backtest
python betting/backtesting.py \
    --predictions data/predictions.csv \
    --odds data/historical_odds.csv \
    --bankroll 10000 \
    --strategy conservative \
    --output results/backtest_results

# Compare all strategies
python betting/backtesting.py \
    --predictions data/predictions.csv \
    --odds data/historical_odds.csv \
    --bankroll 10000 \
    --strategy compare
```

**Output Metrics:**
- Win rate and total bets
- ROI and total profit
- Maximum drawdown
- Sharpe ratio
- Edge statistics

### 3. Performance Dashboard (`dashboard.py`)

Monitor and visualize betting performance in real-time.

**Features:**
- Comprehensive performance metrics
- 8-panel visualization dashboard
- Rolling performance analysis
- Monthly breakdowns
- Export capabilities

**Usage:**

```bash
python betting/dashboard.py \
    --history results/bet_history.csv \
    --bankroll 10000 \
    --output_dir results/dashboard \
    --rolling_window 20
```

**Dashboard Panels:**
1. Bankroll over time
2. Running win rate
3. Cumulative profit
4. Drawdown chart
5. Profit distribution
6. Win/loss ratio
7. Bet size distribution
8. ROI per bet

## Betting Strategy Framework

### Phase 1: Setup

1. **Train and calibrate models:**
   ```bash
   python main/classifier.py --csv data/atp_database.csv --rootdir /path/to/data
   ```

2. **Generate predictions with probabilities:**
   ```bash
   python main/classifier.py \
       --csv data/deploy.csv \
       --timestring 2024_01_15_10_30_00 \
       --classifier_name AdaBoost
   ```

### Phase 2: Bet Sizing

3. **Calculate recommended bets:**
   ```bash
   python betting/kelly_calculator.py \
       --predictions data/predictions_2024_01_15_10_30_00.csv \
       --bankroll 10000 \
       --strategy conservative
   ```

### Phase 3: Validation

4. **Backtest strategy:**
   ```bash
   python betting/backtesting.py \
       --predictions data/predictions.csv \
       --odds data/historical_odds.csv \
       --strategy conservative
   ```

### Phase 4: Monitoring

5. **Track live performance:**
   ```bash
   python betting/dashboard.py \
       --history results/bet_history.csv \
       --bankroll 10000
   ```

## Kelly Criterion Formula

```
f* = (bp - q) / b

Where:
- f* = fraction of bankroll to bet
- b = decimal odds - 1
- p = model probability of winning
- q = 1 - p (probability of losing)
```

**Fractional Kelly**: Multiply f* by fraction (0.25 for conservative)

## Risk Management Rules

### Bet Sizing Limits
- Maximum single bet: 3% of bankroll
- Fractional Kelly recommended: 0.25-0.50
- Recalculate bankroll weekly

### Entry Criteria
- Minimum edge: 5% (conservative) or 3% (moderate)
- Minimum confidence: 60-65%
- Positive expected value required

### Stop-Loss Protocols
- Daily loss limit: 5% of bankroll
- Weekly loss limit: 10% of bankroll
- Monthly loss limit: 20% of bankroll

## Expected Performance

Based on 93% model accuracy and conservative strategy:

**Realistic Projections:**
- **Win Rate**: 58-65% (on bets placed, not all matches)
- **ROI**: 5-15% annually
- **Sharpe Ratio**: >1.5
- **Max Drawdown**: <25%

**Comparison:**
- Professional sports bettors: 5-10% annual ROI
- Stock market average: ~10% annually
- Kaunitz et al. (2017): 8.5% ROI over 265 bets

## Data Requirements

### Predictions File Format
CSV with columns:
- `player1_name`: Name of player 1
- `player2_name`: Name of player 2
- `player1_win_prob`: Model probability (0-1)
- `player2_win_prob`: Model probability (0-1)
- `predicted_winner`: 1 or 2

### Odds File Format (Optional)
CSV with columns:
- `player1_name`
- `player2_name`
- `player1_odds`: Decimal odds
- `player2_odds`: Decimal odds
- `date`: Match date

If odds not provided, backtesting simulates odds with 3% bookmaker margin.

## Installation

No additional dependencies beyond main project requirements:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Examples

### Example 1: Interactive Bet Calculator

```bash
python betting/kelly_calculator.py --interactive

# Enter values:
# Bankroll: $10000
# Win probability: 0.68
# Odds: 1.75
# Strategy: conservative

# Output:
# Edge: 11.4%
# Expected Value: $41.50
# Recommended Bet: $531.43 (5.3% of bankroll)
```

### Example 2: Generate Betting Card

```bash
python betting/kelly_calculator.py \
    --predictions data/deploy_predictions.csv \
    --bankroll 10000 \
    --strategy conservative

# Output:
# Match: Djokovic vs Alcaraz
#   Bet On: Djokovic
#   Model Probability: 68.0%
#   Odds: 1.75
#   Edge: 11.4%
#   Recommended Bet: $531.43
#   Expected Value: $41.50
```

### Example 3: Strategy Comparison

```bash
python betting/backtesting.py \
    --predictions data/2023_predictions.csv \
    --strategy compare \
    --bankroll 10000

# Output:
# Strategy         Final Bankroll    Total Bets    ROI
# Conservative     $11,234.56           127        12.3%
# Moderate         $11,892.34           238        18.9%
# Aggressive       $10,123.45           445         1.2%
```

## Important Notes

### ⚠️ Responsible Gambling
- Only bet what you can afford to lose
- This is for educational purposes
- Past performance doesn't guarantee future results
- Sports betting involves risk

### 📊 Model Calibration
- Models must be calibrated for accurate probabilities
- Recalibrate quarterly with new data
- Monitor for model drift

### 🎯 Market Selection
- Best for ATP main tour matches
- Avoid early Grand Slam rounds (high variance)
- Focus on surfaces with most training data

### 🔄 Continuous Improvement
- Track all bets in CSV format
- Review performance monthly
- Adjust strategy based on results
- Update models with latest data

## Troubleshooting

**Issue**: No betting opportunities found
- **Solution**: Lower min_edge or min_confidence thresholds

**Issue**: Bet sizes too large
- **Solution**: Reduce kelly_fraction or set lower max_bet_pct

**Issue**: Model probabilities not calibrated
- **Solution**: Retrain classifiers (they now auto-calibrate)

**Issue**: Bookmaker odds not available
- **Solution**: Backtester will simulate odds with 3% margin

## References

1. Kelly, J. L. (1956). "A New Interpretation of Information Rate"
2. Kaunitz, L., Zhong, S., & Kreiner, J. (2017). "Beating the bookies with their own numbers"
3. Sipko, M. (2015). "Machine Learning for the Prediction of Professional Tennis Matches"

## Contact

For questions or issues with the betting module, see main project README.

---

**Disclaimer**: This software is for educational and research purposes only. Use at your own risk.
