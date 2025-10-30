"""
Backtesting framework for tennis betting strategy
Validates betting strategy using historical match data and odds
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

__author__ = 'Josh Lamstein'


class Backtester:
    def __init__(self, predictions_csv, odds_csv, initial_bankroll=10000):
        """
        Initialize backtester with predictions and historical odds

        Args:
            predictions_csv: CSV with columns [player1_name, player2_name, player1_win_prob, player2_win_prob, match_date]
            odds_csv: CSV with historical odds from tennis-data.co.uk
            initial_bankroll: Starting bankroll in dollars
        """
        self.predictions = pd.read_csv(predictions_csv)
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll

        # Load odds if provided
        if os.path.exists(odds_csv):
            self.odds = pd.read_csv(odds_csv)
            self._match_odds_to_predictions()
        else:
            print(f"Warning: Odds file {odds_csv} not found. Using simulated odds.")
            self._simulate_odds()

        self.bet_history = []
        self.bankroll_history = [initial_bankroll]

    def _match_odds_to_predictions(self):
        """Match historical odds to predictions by player names and dates"""
        # Standardize names and merge
        # This is a placeholder - actual implementation depends on odds file format
        print("Matching odds to predictions...")
        # TODO: Implement actual matching logic based on tennis-data.co.uk format

    def _simulate_odds(self):
        """Simulate odds based on model probabilities with bookmaker margin"""
        print("Simulating odds with 3% bookmaker margin...")
        bookmaker_margin = 1.03

        # Convert probabilities to fair odds, then add margin
        self.predictions['player1_fair_odds'] = 1 / self.predictions['player1_win_prob']
        self.predictions['player2_fair_odds'] = 1 / self.predictions['player2_win_prob']

        # Add bookmaker margin and random noise (±5%)
        noise = np.random.uniform(0.95, 1.05, len(self.predictions))
        self.predictions['player1_odds'] = self.predictions['player1_fair_odds'] * bookmaker_margin * noise
        self.predictions['player2_odds'] = self.predictions['player2_fair_odds'] * bookmaker_margin * noise

        # Add random actual winners for backtesting (based on probabilities)
        self.predictions['actual_winner'] = self.predictions.apply(
            lambda row: 1 if np.random.random() < row['player1_win_prob'] else 2, axis=1
        )

    def kelly_criterion(self, win_prob: float, odds: float, fraction: float = 0.25) -> float:
        """
        Calculate Kelly Criterion bet size

        Args:
            win_prob: Probability of winning (0-1)
            odds: Decimal odds (e.g., 2.5)
            fraction: Fractional Kelly (default 0.25 for conservative)

        Returns:
            Fraction of bankroll to bet
        """
        b = odds - 1  # Net odds
        q = 1 - win_prob

        if win_prob * b - q <= 0:
            return 0  # No edge

        kelly_fraction = (win_prob * b - q) / b
        return max(0, min(kelly_fraction * fraction, 0.03))  # Cap at 3% of bankroll

    def calculate_expected_value(self, win_prob: float, odds: float) -> float:
        """Calculate expected value of a bet"""
        return (win_prob * (odds - 1)) - (1 - win_prob)

    def should_bet(self, win_prob: float, odds: float, min_edge: float = 0.05,
                   min_confidence: float = 0.60) -> bool:
        """
        Determine if a bet meets criteria

        Args:
            win_prob: Model probability
            odds: Bookmaker decimal odds
            min_edge: Minimum edge required (default 5%)
            min_confidence: Minimum confidence threshold (default 60%)

        Returns:
            Boolean whether to place bet
        """
        implied_prob = 1 / odds
        edge = win_prob - implied_prob
        ev = self.calculate_expected_value(win_prob, odds)

        return (edge >= min_edge and
                win_prob >= min_confidence and
                ev > 0)

    def run_backtest(self, strategy: str = 'conservative', min_edge: float = 0.05,
                     min_confidence: float = 0.60, kelly_fraction: float = 0.25):
        """
        Run backtesting simulation

        Args:
            strategy: 'conservative', 'moderate', or 'aggressive'
            min_edge: Minimum edge threshold
            min_confidence: Minimum confidence threshold
            kelly_fraction: Kelly multiplier
        """
        print(f"\n{'='*60}")
        print(f"Running {strategy.upper()} strategy backtest")
        print(f"Initial Bankroll: ${self.initial_bankroll:,.2f}")
        print(f"Min Edge: {min_edge*100:.1f}% | Min Confidence: {min_confidence*100:.1f}%")
        print(f"Kelly Fraction: {kelly_fraction}")
        print(f"{'='*60}\n")

        self.current_bankroll = self.initial_bankroll
        self.bet_history = []
        self.bankroll_history = [self.initial_bankroll]

        total_bets = 0
        winning_bets = 0
        total_staked = 0
        total_profit = 0

        for idx, row in self.predictions.iterrows():
            # Check both players for betting opportunities
            for player_num in [1, 2]:
                win_prob = row[f'player{player_num}_win_prob']
                odds = row[f'player{player_num}_odds']

                if self.should_bet(win_prob, odds, min_edge, min_confidence):
                    # Calculate bet size using Kelly
                    kelly_bet_fraction = self.kelly_criterion(win_prob, odds, kelly_fraction)
                    bet_amount = self.current_bankroll * kelly_bet_fraction

                    if bet_amount > 0:
                        total_bets += 1
                        total_staked += bet_amount

                        # Determine if bet won
                        actual_winner = row['actual_winner']
                        bet_won = (actual_winner == player_num)

                        if bet_won:
                            winning_bets += 1
                            profit = bet_amount * (odds - 1)
                            self.current_bankroll += profit
                            total_profit += profit
                        else:
                            self.current_bankroll -= bet_amount
                            total_profit -= bet_amount

                        # Record bet
                        bet_record = {
                            'match_id': idx,
                            'player1': row['player1_name'],
                            'player2': row['player2_name'],
                            'bet_on_player': player_num,
                            'bet_amount': bet_amount,
                            'odds': odds,
                            'win_prob': win_prob,
                            'edge': win_prob - (1/odds),
                            'won': bet_won,
                            'profit': profit if bet_won else -bet_amount,
                            'bankroll_after': self.current_bankroll
                        }
                        self.bet_history.append(bet_record)
                        self.bankroll_history.append(self.current_bankroll)

        # Calculate statistics
        self._print_results(total_bets, winning_bets, total_staked, total_profit)
        return self.bet_history

    def _print_results(self, total_bets, winning_bets, total_staked, total_profit):
        """Print backtesting results"""
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")

        if total_bets == 0:
            print("No bets placed - criteria too strict or no opportunities")
            return

        win_rate = winning_bets / total_bets
        roi = (total_profit / total_staked) * 100
        final_bankroll = self.current_bankroll
        total_return = ((final_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100

        print(f"Total Bets: {total_bets}")
        print(f"Winning Bets: {winning_bets}")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"\nTotal Staked: ${total_staked:,.2f}")
        print(f"Total Profit: ${total_profit:,.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"\nInitial Bankroll: ${self.initial_bankroll:,.2f}")
        print(f"Final Bankroll: ${final_bankroll:,.2f}")
        print(f"Total Return: {total_return:.2f}%")

        # Calculate additional metrics
        if len(self.bet_history) > 0:
            bet_df = pd.DataFrame(self.bet_history)
            avg_edge = bet_df['edge'].mean()
            avg_odds = bet_df['odds'].mean()
            max_bet = bet_df['bet_amount'].max()

            print(f"\nAverage Edge: {avg_edge*100:.2f}%")
            print(f"Average Odds: {avg_odds:.2f}")
            print(f"Max Single Bet: ${max_bet:,.2f}")

            # Drawdown
            bankroll_series = pd.Series(self.bankroll_history)
            running_max = bankroll_series.expanding().max()
            drawdown = (bankroll_series - running_max) / running_max
            max_drawdown = drawdown.min() * 100

            print(f"Max Drawdown: {max_drawdown:.2f}%")

            # Sharpe ratio (annualized, assuming ~250 betting days per year)
            returns = bet_df['profit'] / bet_df['bet_amount']
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(250)
                print(f"Sharpe Ratio: {sharpe:.2f}")

        print(f"{'='*60}\n")

    def plot_results(self, save_path=None):
        """Plot backtesting results"""
        if len(self.bet_history) == 0:
            print("No bets to plot")
            return

        bet_df = pd.DataFrame(self.bet_history)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Bankroll over time
        axes[0, 0].plot(self.bankroll_history, linewidth=2)
        axes[0, 0].axhline(y=self.initial_bankroll, color='r', linestyle='--', label='Initial Bankroll')
        axes[0, 0].set_title('Bankroll Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Bet Number')
        axes[0, 0].set_ylabel('Bankroll ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Cumulative profit
        cumulative_profit = bet_df['profit'].cumsum()
        axes[0, 1].plot(cumulative_profit, linewidth=2, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Cumulative Profit', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Bet Number')
        axes[0, 1].set_ylabel('Cumulative Profit ($)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Win/Loss distribution
        win_loss = bet_df['won'].value_counts()
        axes[1, 0].bar(['Losses', 'Wins'], [win_loss.get(False, 0), win_loss.get(True, 0)],
                       color=['red', 'green'], alpha=0.7)
        axes[1, 0].set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Count')

        # 4. Profit distribution
        axes[1, 1].hist(bet_df['profit'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_title('Profit Distribution per Bet', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Profit ($)')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def save_bet_history(self, output_path):
        """Save detailed bet history to CSV"""
        if len(self.bet_history) > 0:
            bet_df = pd.DataFrame(self.bet_history)
            bet_df.to_csv(output_path, index=False)
            print(f"Saved bet history to {output_path}")
        else:
            print("No bets to save")


def compare_strategies(predictions_csv, odds_csv, initial_bankroll=10000):
    """Compare different betting strategies"""
    strategies = {
        'Conservative': {'min_edge': 0.05, 'min_confidence': 0.65, 'kelly_fraction': 0.25},
        'Moderate': {'min_edge': 0.03, 'min_confidence': 0.60, 'kelly_fraction': 0.50},
        'Aggressive': {'min_edge': 0.02, 'min_confidence': 0.55, 'kelly_fraction': 0.75}
    }

    results = {}

    for strategy_name, params in strategies.items():
        backtester = Backtester(predictions_csv, odds_csv, initial_bankroll)
        backtester.run_backtest(
            strategy=strategy_name.lower(),
            min_edge=params['min_edge'],
            min_confidence=params['min_confidence'],
            kelly_fraction=params['kelly_fraction']
        )

        results[strategy_name] = {
            'final_bankroll': backtester.current_bankroll,
            'total_bets': len(backtester.bet_history),
            'roi': ((backtester.current_bankroll - initial_bankroll) / initial_bankroll) * 100
        }

    # Print comparison
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Strategy':<15} {'Final Bankroll':<20} {'Total Bets':<15} {'ROI':<10}")
    print(f"{'-'*60}")
    for strategy, metrics in results.items():
        print(f"{strategy:<15} ${metrics['final_bankroll']:>15,.2f}    {metrics['total_bets']:>10}    {metrics['roi']:>8.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backtest tennis betting strategy")
    parser.add_argument('--predictions', required=True,
                        help='CSV file with model predictions and probabilities')
    parser.add_argument('--odds', default='data/historical_odds.csv',
                        help='CSV file with historical betting odds')
    parser.add_argument('--bankroll', type=float, default=10000,
                        help='Initial bankroll in dollars')
    parser.add_argument('--strategy', choices=['conservative', 'moderate', 'aggressive', 'compare'],
                        default='conservative', help='Betting strategy to use')
    parser.add_argument('--output', default='results/backtest_results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    if args.strategy == 'compare':
        compare_strategies(args.predictions, args.odds, args.bankroll)
    else:
        backtester = Backtester(args.predictions, args.odds, args.bankroll)

        # Map strategy to parameters
        strategy_params = {
            'conservative': {'min_edge': 0.05, 'min_confidence': 0.65, 'kelly_fraction': 0.25},
            'moderate': {'min_edge': 0.03, 'min_confidence': 0.60, 'kelly_fraction': 0.50},
            'aggressive': {'min_edge': 0.02, 'min_confidence': 0.55, 'kelly_fraction': 0.75}
        }

        params = strategy_params[args.strategy]
        backtester.run_backtest(
            strategy=args.strategy,
            min_edge=params['min_edge'],
            min_confidence=params['min_confidence'],
            kelly_fraction=params['kelly_fraction']
        )

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backtester.save_bet_history(os.path.join(args.output, f'bet_history_{timestamp}.csv'))
        backtester.plot_results(os.path.join(args.output, f'backtest_plot_{timestamp}.png'))
