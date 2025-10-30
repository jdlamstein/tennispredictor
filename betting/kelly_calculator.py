"""
Kelly Criterion Bet Sizing Calculator
Calculates optimal bet sizes based on model probabilities and bookmaker odds
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple

__author__ = 'Josh Lamstein'


class KellyCalculator:
    def __init__(self, current_bankroll: float = 10000):
        """
        Initialize Kelly Calculator

        Args:
            current_bankroll: Current betting bankroll
        """
        self.current_bankroll = current_bankroll
        self.bet_recommendations = []

    def kelly_criterion(self, win_prob: float, odds: float) -> float:
        """
        Calculate full Kelly Criterion bet size

        Formula: f* = (bp - q) / b
        where:
            f* = fraction of bankroll to bet
            b = decimal odds - 1
            p = probability of winning
            q = 1 - p

        Args:
            win_prob: Probability of winning (0-1)
            odds: Decimal odds (e.g., 2.5)

        Returns:
            Optimal fraction of bankroll to bet (0-1)
        """
        b = odds - 1  # Net odds
        p = win_prob
        q = 1 - p

        # Check if there's an edge
        if p * b - q <= 0:
            return 0.0

        kelly_fraction = (p * b - q) / b
        return max(0.0, kelly_fraction)

    def fractional_kelly(self, win_prob: float, odds: float, fraction: float = 0.25) -> float:
        """
        Calculate fractional Kelly bet size (more conservative)

        Args:
            win_prob: Probability of winning (0-1)
            odds: Decimal odds
            fraction: Kelly fraction multiplier (0.25 = quarter Kelly)

        Returns:
            Fractional Kelly bet size
        """
        full_kelly = self.kelly_criterion(win_prob, odds)
        return full_kelly * fraction

    def calculate_edge(self, win_prob: float, odds: float) -> float:
        """
        Calculate betting edge (model prob - implied prob)

        Args:
            win_prob: Model's win probability
            odds: Bookmaker decimal odds

        Returns:
            Edge as decimal (e.g., 0.05 = 5% edge)
        """
        implied_prob = 1 / odds
        return win_prob - implied_prob

    def calculate_expected_value(self, win_prob: float, odds: float, bet_amount: float) -> float:
        """
        Calculate expected value of a bet

        Args:
            win_prob: Probability of winning
            odds: Decimal odds
            bet_amount: Amount to bet

        Returns:
            Expected value in dollars
        """
        win_return = bet_amount * (odds - 1)
        expected_win = win_prob * win_return
        expected_loss = (1 - win_prob) * bet_amount
        return expected_win - expected_loss

    def calculate_bet_size(self, win_prob: float, odds: float, strategy: str = 'conservative',
                          max_bet_pct: float = 0.03) -> Dict:
        """
        Calculate recommended bet size with multiple strategies

        Args:
            win_prob: Model win probability
            odds: Bookmaker decimal odds
            strategy: 'conservative', 'moderate', or 'aggressive'
            max_bet_pct: Maximum bet as fraction of bankroll

        Returns:
            Dictionary with bet recommendations
        """
        # Calculate Kelly fractions based on strategy
        kelly_fractions = {
            'conservative': 0.25,  # Quarter Kelly
            'moderate': 0.50,      # Half Kelly
            'aggressive': 0.75     # Three-quarter Kelly
        }

        kelly_frac = kelly_fractions.get(strategy, 0.25)
        full_kelly = self.kelly_criterion(win_prob, odds)
        fractional_kelly = full_kelly * kelly_frac

        # Apply maximum bet constraint
        bet_fraction = min(fractional_kelly, max_bet_pct)
        bet_amount = self.current_bankroll * bet_fraction

        # Calculate metrics
        edge = self.calculate_edge(win_prob, odds)
        ev = self.calculate_expected_value(win_prob, odds, bet_amount)

        return {
            'bet_amount': bet_amount,
            'bet_fraction': bet_fraction,
            'full_kelly': full_kelly,
            'fractional_kelly': fractional_kelly,
            'edge': edge,
            'expected_value': ev,
            'roi': ev / bet_amount if bet_amount > 0 else 0,
            'strategy': strategy
        }

    def should_bet(self, win_prob: float, odds: float, min_edge: float = 0.05,
                   min_confidence: float = 0.60, min_ev: float = 0) -> Tuple[bool, str]:
        """
        Determine if a bet meets criteria

        Args:
            win_prob: Model probability
            odds: Bookmaker odds
            min_edge: Minimum edge required
            min_confidence: Minimum confidence threshold
            min_ev: Minimum expected value required

        Returns:
            Tuple of (should_bet, reason)
        """
        edge = self.calculate_edge(win_prob, odds)
        ev = self.calculate_expected_value(win_prob, odds, self.current_bankroll * 0.01)

        if win_prob < min_confidence:
            return False, f"Confidence too low: {win_prob:.1%} < {min_confidence:.1%}"

        if edge < min_edge:
            return False, f"Edge too small: {edge:.1%} < {min_edge:.1%}"

        if ev < min_ev:
            return False, f"Expected value too low: ${ev:.2f}"

        return True, "All criteria met"

    def analyze_match(self, player1_name: str, player2_name: str,
                     player1_prob: float, player2_prob: float,
                     player1_odds: float, player2_odds: float,
                     strategy: str = 'conservative') -> Dict:
        """
        Analyze a single match for betting opportunities

        Args:
            player1_name: Name of player 1
            player2_name: Name of player 2
            player1_prob: Model probability for player 1
            player2_prob: Model probability for player 2
            player1_odds: Bookmaker odds for player 1
            player2_odds: Bookmaker odds for player 2
            strategy: Betting strategy

        Returns:
            Dictionary with analysis results
        """
        result = {
            'player1_name': player1_name,
            'player2_name': player2_name,
            'match_analysis': []
        }

        # Analyze betting on player 1
        bet1_calc = self.calculate_bet_size(player1_prob, player1_odds, strategy)
        should_bet1, reason1 = self.should_bet(player1_prob, player1_odds)

        player1_analysis = {
            'player': player1_name,
            'model_prob': player1_prob,
            'odds': player1_odds,
            'implied_prob': 1 / player1_odds,
            'edge': bet1_calc['edge'],
            'should_bet': should_bet1,
            'reason': reason1,
            'recommended_bet': bet1_calc['bet_amount'] if should_bet1 else 0,
            'expected_value': bet1_calc['expected_value'] if should_bet1 else 0
        }

        # Analyze betting on player 2
        bet2_calc = self.calculate_bet_size(player2_prob, player2_odds, strategy)
        should_bet2, reason2 = self.should_bet(player2_prob, player2_odds)

        player2_analysis = {
            'player': player2_name,
            'model_prob': player2_prob,
            'odds': player2_odds,
            'implied_prob': 1 / player2_odds,
            'edge': bet2_calc['edge'],
            'should_bet': should_bet2,
            'reason': reason2,
            'recommended_bet': bet2_calc['bet_amount'] if should_bet2 else 0,
            'expected_value': bet2_calc['expected_value'] if should_bet2 else 0
        }

        result['match_analysis'] = [player1_analysis, player2_analysis]
        return result

    def process_predictions_file(self, predictions_csv: str, odds_csv: str = None,
                                strategy: str = 'conservative') -> pd.DataFrame:
        """
        Process a file of predictions and generate betting recommendations

        Args:
            predictions_csv: CSV with model predictions
            odds_csv: CSV with bookmaker odds (optional)
            strategy: Betting strategy

        Returns:
            DataFrame with betting recommendations
        """
        predictions = pd.read_csv(predictions_csv)

        # If odds not provided, simulate them
        if odds_csv is None or not os.path.exists(odds_csv):
            print("No odds file provided. Using simulated odds.")
            predictions['player1_odds'] = 1 / predictions['player1_win_prob'] * 1.03
            predictions['player2_odds'] = 1 / predictions['player2_win_prob'] * 1.03

        recommendations = []

        for _, row in predictions.iterrows():
            match_result = self.analyze_match(
                row['player1_name'], row['player2_name'],
                row['player1_win_prob'], row['player2_win_prob'],
                row['player1_odds'], row['player2_odds'],
                strategy
            )

            for player_analysis in match_result['match_analysis']:
                if player_analysis['should_bet']:
                    recommendations.append({
                        'match': f"{row['player1_name']} vs {row['player2_name']}",
                        'bet_on': player_analysis['player'],
                        'model_prob': player_analysis['model_prob'],
                        'odds': player_analysis['odds'],
                        'edge': player_analysis['edge'],
                        'bet_amount': player_analysis['recommended_bet'],
                        'expected_value': player_analysis['expected_value'],
                        'reason': player_analysis['reason']
                    })

        return pd.DataFrame(recommendations)

    def print_betting_card(self, recommendations_df: pd.DataFrame):
        """Print formatted betting recommendations"""
        if len(recommendations_df) == 0:
            print("\nNo betting opportunities found with current criteria.")
            return

        print(f"\n{'='*80}")
        print(f"BETTING RECOMMENDATIONS - Bankroll: ${self.current_bankroll:,.2f}")
        print(f"{'='*80}\n")

        total_stake = recommendations_df['bet_amount'].sum()
        total_ev = recommendations_df['expected_value'].sum()

        for idx, row in recommendations_df.iterrows():
            print(f"Match: {row['match']}")
            print(f"  Bet On: {row['bet_on']}")
            print(f"  Model Probability: {row['model_prob']:.1%}")
            print(f"  Odds: {row['odds']:.2f}")
            print(f"  Edge: {row['edge']:.1%}")
            print(f"  Recommended Bet: ${row['bet_amount']:.2f} ({row['bet_amount']/self.current_bankroll:.1%} of bankroll)")
            print(f"  Expected Value: ${row['expected_value']:.2f}")
            print(f"  {row['reason']}")
            print()

        print(f"{'='*80}")
        print(f"Total Recommended Stake: ${total_stake:.2f} ({total_stake/self.current_bankroll:.1%} of bankroll)")
        print(f"Total Expected Value: ${total_ev:.2f}")
        print(f"Number of Bets: {len(recommendations_df)}")
        print(f"{'='*80}\n")


def interactive_calculator():
    """Interactive command-line Kelly calculator"""
    print("\n" + "="*60)
    print("KELLY CRITERION BET CALCULATOR")
    print("="*60 + "\n")

    bankroll = float(input("Enter your current bankroll: $"))
    calc = KellyCalculator(bankroll)

    while True:
        print("\n" + "-"*60)
        win_prob = float(input("Enter model win probability (0-1): "))
        odds = float(input("Enter bookmaker decimal odds: "))
        strategy = input("Enter strategy (conservative/moderate/aggressive): ").lower()

        if strategy not in ['conservative', 'moderate', 'aggressive']:
            strategy = 'conservative'
            print("Invalid strategy. Using conservative.")

        # Calculate bet size
        bet_calc = calc.calculate_bet_size(win_prob, odds, strategy)
        should_bet, reason = calc.should_bet(win_prob, odds)

        # Print results
        print("\n" + "="*60)
        print("BETTING ANALYSIS")
        print("="*60)
        print(f"Model Probability: {win_prob:.1%}")
        print(f"Bookmaker Odds: {odds:.2f}")
        print(f"Implied Probability: {1/odds:.1%}")
        print(f"Edge: {bet_calc['edge']:.1%}")
        print(f"Expected Value: ${bet_calc['expected_value']:.2f}")
        print(f"\nFull Kelly: {bet_calc['full_kelly']:.1%} of bankroll")
        print(f"{strategy.capitalize()} Kelly: {bet_calc['fractional_kelly']:.1%} of bankroll")
        print(f"\nRECOMMENDED BET: ${bet_calc['bet_amount']:.2f}")
        print(f"({bet_calc['bet_fraction']:.1%} of ${bankroll:.2f} bankroll)")
        print(f"\nShould Bet: {'YES' if should_bet else 'NO'}")
        print(f"Reason: {reason}")
        print("="*60)

        cont = input("\nCalculate another bet? (y/n): ").lower()
        if cont != 'y':
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kelly Criterion Bet Sizing Calculator")
    parser.add_argument('--predictions', type=str,
                        help='CSV file with model predictions')
    parser.add_argument('--odds', type=str,
                        help='CSV file with bookmaker odds')
    parser.add_argument('--bankroll', type=float, default=10000,
                        help='Current bankroll in dollars')
    parser.add_argument('--strategy', choices=['conservative', 'moderate', 'aggressive'],
                        default='conservative', help='Betting strategy')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--output', type=str, default='results/betting_recommendations.csv',
                        help='Output file for recommendations')

    args = parser.parse_args()

    if args.interactive:
        interactive_calculator()
    elif args.predictions:
        calc = KellyCalculator(args.bankroll)
        recommendations = calc.process_predictions_file(args.predictions, args.odds, args.strategy)

        # Print betting card
        calc.print_betting_card(recommendations)

        # Save to file
        if len(recommendations) > 0:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            recommendations.to_csv(args.output, index=False)
            print(f"Saved recommendations to {args.output}")
    else:
        print("Please provide --predictions file or use --interactive mode")
        print("Example: python kelly_calculator.py --predictions predictions.csv --bankroll 10000")
