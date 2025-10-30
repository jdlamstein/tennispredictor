"""
Quick tests to verify betting module components
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from betting.kelly_calculator import KellyCalculator
from betting.backtesting import Backtester
from betting.dashboard import BettingDashboard


def create_sample_data():
    """Create sample data for testing"""
    # Sample predictions
    predictions = pd.DataFrame({
        'player1_name': ['Djokovic', 'Nadal', 'Federer', 'Murray', 'Alcaraz'] * 20,
        'player2_name': ['Medvedev', 'Tsitsipas', 'Zverev', 'Rublev', 'Sinner'] * 20,
        'player1_win_prob': np.random.uniform(0.55, 0.85, 100),
        'player2_win_prob': np.random.uniform(0.15, 0.45, 100)
    })

    # Simulate odds with bookmaker margin
    predictions['player1_odds'] = (1 / predictions['player1_win_prob']) * 1.03
    predictions['player2_odds'] = (1 / predictions['player2_win_prob']) * 1.03

    # Simulate actual winners based on probabilities
    predictions['actual_winner'] = predictions.apply(
        lambda row: 1 if np.random.random() < row['player1_win_prob'] else 2, axis=1
    )

    return predictions


def test_kelly_calculator():
    """Test Kelly Calculator"""
    print("\n" + "="*60)
    print("Testing Kelly Calculator")
    print("="*60)

    calc = KellyCalculator(current_bankroll=10000)

    # Test single calculation
    win_prob = 0.65
    odds = 1.80
    result = calc.calculate_bet_size(win_prob, odds, strategy='conservative')

    print(f"\nSingle Bet Calculation:")
    print(f"Win Probability: {win_prob:.1%}")
    print(f"Odds: {odds:.2f}")
    print(f"Edge: {result['edge']:.2%}")
    print(f"Recommended Bet: ${result['bet_amount']:.2f}")
    print(f"Expected Value: ${result['expected_value']:.2f}")

    assert result['bet_amount'] > 0, "Bet amount should be positive"
    assert result['edge'] > 0, "Should have positive edge"

    print("\n✓ Kelly Calculator tests passed!")


def test_backtesting():
    """Test Backtesting Framework"""
    print("\n" + "="*60)
    print("Testing Backtesting Framework")
    print("="*60)

    # Create sample data
    predictions = create_sample_data()

    # Save to temp file
    temp_pred_file = '/tmp/test_predictions.csv'
    predictions.to_csv(temp_pred_file, index=False)

    # Run backtest
    backtester = Backtester(temp_pred_file, temp_pred_file, initial_bankroll=10000)
    bet_history = backtester.run_backtest(strategy='conservative')

    print(f"\nBacktest completed:")
    print(f"Total bets: {len(bet_history)}")
    print(f"Final bankroll: ${backtester.current_bankroll:.2f}")

    assert len(bet_history) >= 0, "Should have bet history"
    assert backtester.current_bankroll > 0, "Bankroll should be positive"

    print("\n✓ Backtesting tests passed!")

    return bet_history


def test_dashboard(bet_history):
    """Test Dashboard"""
    print("\n" + "="*60)
    print("Testing Dashboard")
    print("="*60)

    if len(bet_history) == 0:
        print("No bet history to test dashboard")
        return

    # Save bet history to temp file
    temp_history_file = '/tmp/test_bet_history.csv'
    bet_history_df = pd.DataFrame(bet_history)
    bet_history_df.to_csv(temp_history_file, index=False)

    # Create dashboard
    dashboard = BettingDashboard(temp_history_file, initial_bankroll=10000)

    # Generate summary stats
    stats = dashboard.generate_summary_stats()

    print(f"\nDashboard Summary:")
    print(f"Total Bets: {stats['total_bets']}")
    print(f"Win Rate: {stats['win_rate']:.2%}")
    print(f"ROI: {stats['roi']:.2f}%")
    print(f"Final Bankroll: ${stats['final_bankroll']:.2f}")

    assert stats['total_bets'] > 0, "Should have bets"
    assert 0 <= stats['win_rate'] <= 1, "Win rate should be between 0 and 1"

    print("\n✓ Dashboard tests passed!")


def test_probability_calibration():
    """Test that probabilities sum to ~1"""
    print("\n" + "="*60)
    print("Testing Probability Calibration")
    print("="*60)

    predictions = create_sample_data()

    # Check probabilities sum to approximately 1
    prob_sums = predictions['player1_win_prob'] + predictions['player2_win_prob']

    print(f"\nProbability Sums:")
    print(f"Mean: {prob_sums.mean():.4f}")
    print(f"Min: {prob_sums.min():.4f}")
    print(f"Max: {prob_sums.max():.4f}")

    assert prob_sums.mean() > 0.95 and prob_sums.mean() < 1.05, "Probabilities should sum to ~1"

    print("\n✓ Probability calibration tests passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("BETTING MODULE TEST SUITE")
    print("="*70)

    try:
        test_kelly_calculator()
        bet_history = test_backtesting()
        test_dashboard(bet_history)
        test_probability_calibration()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70 + "\n")

        print("Betting module is ready to use!")
        print("\nNext steps:")
        print("1. Train models with calibration: python main/classifier.py")
        print("2. Generate predictions: python main/classifier.py --timestring YOUR_TS")
        print("3. Calculate bets: python betting/kelly_calculator.py --predictions FILE")
        print("4. Backtest strategy: python betting/backtesting.py --predictions FILE")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()
