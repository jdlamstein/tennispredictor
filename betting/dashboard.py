"""
Betting Performance Monitoring Dashboard
Tracks betting performance, ROI, win rates, and other key metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import os
from typing import Dict, List

__author__ = 'Josh Lamstein'

sns.set_style('whitegrid')


class BettingDashboard:
    def __init__(self, bet_history_csv: str, initial_bankroll: float = 10000):
        """
        Initialize betting dashboard

        Args:
            bet_history_csv: CSV file with betting history
            initial_bankroll: Starting bankroll
        """
        self.bet_history = pd.read_csv(bet_history_csv)
        self.initial_bankroll = initial_bankroll

        # Convert to datetime if date column exists
        if 'date' in self.bet_history.columns:
            self.bet_history['date'] = pd.to_datetime(self.bet_history['date'])

        # Calculate cumulative metrics
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate key performance metrics"""
        # Cumulative profit
        self.bet_history['cumulative_profit'] = self.bet_history['profit'].cumsum()

        # Running bankroll
        if 'bankroll_after' not in self.bet_history.columns:
            self.bet_history['bankroll_after'] = self.initial_bankroll + self.bet_history['cumulative_profit']

        # Running win rate
        self.bet_history['cumulative_wins'] = self.bet_history['won'].cumsum()
        self.bet_history['bet_number'] = range(1, len(self.bet_history) + 1)
        self.bet_history['running_win_rate'] = self.bet_history['cumulative_wins'] / self.bet_history['bet_number']

        # Calculate ROI per bet
        self.bet_history['roi_per_bet'] = (self.bet_history['profit'] / self.bet_history['bet_amount']) * 100

        # Calculate drawdown
        self.bet_history['peak_bankroll'] = self.bet_history['bankroll_after'].expanding().max()
        self.bet_history['drawdown'] = (self.bet_history['bankroll_after'] - self.bet_history['peak_bankroll']) / self.bet_history['peak_bankroll'] * 100

    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics"""
        total_bets = len(self.bet_history)
        wins = self.bet_history['won'].sum()
        losses = total_bets - wins
        win_rate = wins / total_bets if total_bets > 0 else 0

        total_staked = self.bet_history['bet_amount'].sum()
        total_profit = self.bet_history['profit'].sum()
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0

        final_bankroll = self.bet_history['bankroll_after'].iloc[-1] if len(self.bet_history) > 0 else self.initial_bankroll
        total_return = ((final_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100

        avg_bet_size = self.bet_history['bet_amount'].mean()
        avg_odds = self.bet_history['odds'].mean()
        avg_edge = self.bet_history['edge'].mean() if 'edge' in self.bet_history.columns else None

        max_win = self.bet_history['profit'].max()
        max_loss = self.bet_history['profit'].min()
        max_drawdown = self.bet_history['drawdown'].min()

        # Sharpe ratio
        returns = self.bet_history['roi_per_bet'] / 100
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Longest winning/losing streaks
        winning_streaks = self._calculate_streaks(self.bet_history['won'])
        longest_win_streak = max(winning_streaks['win_streaks']) if winning_streaks['win_streaks'] else 0
        longest_lose_streak = max(winning_streaks['lose_streaks']) if winning_streaks['lose_streaks'] else 0

        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': roi,
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': final_bankroll,
            'total_return': total_return,
            'avg_bet_size': avg_bet_size,
            'avg_odds': avg_odds,
            'avg_edge': avg_edge,
            'max_win': max_win,
            'max_loss': max_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'longest_win_streak': longest_win_streak,
            'longest_lose_streak': longest_lose_streak
        }

    def _calculate_streaks(self, won_series):
        """Calculate winning and losing streaks"""
        win_streaks = []
        lose_streaks = []
        current_win_streak = 0
        current_lose_streak = 0

        for won in won_series:
            if won:
                current_win_streak += 1
                if current_lose_streak > 0:
                    lose_streaks.append(current_lose_streak)
                    current_lose_streak = 0
            else:
                current_lose_streak += 1
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0

        if current_win_streak > 0:
            win_streaks.append(current_win_streak)
        if current_lose_streak > 0:
            lose_streaks.append(current_lose_streak)

        return {'win_streaks': win_streaks, 'lose_streaks': lose_streaks}

    def print_summary(self):
        """Print summary statistics to console"""
        stats = self.generate_summary_stats()

        print(f"\n{'='*80}")
        print("BETTING PERFORMANCE DASHBOARD")
        print(f"{'='*80}\n")

        print("OVERALL PERFORMANCE")
        print(f"{'-'*80}")
        print(f"Total Bets: {stats['total_bets']}")
        print(f"Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"Win Rate: {stats['win_rate']:.2%}")
        print()

        print("FINANCIAL METRICS")
        print(f"{'-'*80}")
        print(f"Initial Bankroll: ${stats['initial_bankroll']:,.2f}")
        print(f"Final Bankroll: ${stats['final_bankroll']:,.2f}")
        print(f"Total Return: {stats['total_return']:.2f}%")
        print()
        print(f"Total Staked: ${stats['total_staked']:,.2f}")
        print(f"Total Profit: ${stats['total_profit']:,.2f}")
        print(f"ROI: {stats['roi']:.2f}%")
        print()

        print("BET CHARACTERISTICS")
        print(f"{'-'*80}")
        print(f"Average Bet Size: ${stats['avg_bet_size']:,.2f}")
        print(f"Average Odds: {stats['avg_odds']:.2f}")
        if stats['avg_edge']:
            print(f"Average Edge: {stats['avg_edge']:.2%}")
        print()

        print("RISK METRICS")
        print(f"{'-'*80}")
        print(f"Max Single Win: ${stats['max_win']:,.2f}")
        print(f"Max Single Loss: ${stats['max_loss']:,.2f}")
        print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print()

        print("STREAKS")
        print(f"{'-'*80}")
        print(f"Longest Winning Streak: {stats['longest_win_streak']} bets")
        print(f"Longest Losing Streak: {stats['longest_lose_streak']} bets")
        print(f"{'='*80}\n")

    def plot_dashboard(self, save_path: str = None):
        """Generate comprehensive dashboard visualization"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Bankroll Over Time
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.bet_history['bet_number'], self.bet_history['bankroll_after'],
                linewidth=2, color='steelblue', label='Bankroll')
        ax1.axhline(y=self.initial_bankroll, color='red', linestyle='--',
                   linewidth=1.5, label='Initial Bankroll', alpha=0.7)
        ax1.fill_between(self.bet_history['bet_number'],
                        self.initial_bankroll,
                        self.bet_history['bankroll_after'],
                        where=self.bet_history['bankroll_after'] >= self.initial_bankroll,
                        alpha=0.3, color='green', interpolate=True)
        ax1.fill_between(self.bet_history['bet_number'],
                        self.initial_bankroll,
                        self.bet_history['bankroll_after'],
                        where=self.bet_history['bankroll_after'] < self.initial_bankroll,
                        alpha=0.3, color='red', interpolate=True)
        ax1.set_title('Bankroll Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Bet Number')
        ax1.set_ylabel('Bankroll ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Win Rate Over Time
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(self.bet_history['bet_number'], self.bet_history['running_win_rate'] * 100,
                linewidth=2, color='green')
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Running Win Rate', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Bet Number')
        ax2.set_ylabel('Win Rate (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative Profit
        ax3 = fig.add_subplot(gs[1, 0])
        colors = ['green' if x >= 0 else 'red' for x in self.bet_history['cumulative_profit']]
        ax3.plot(self.bet_history['bet_number'], self.bet_history['cumulative_profit'],
                linewidth=2, color='darkgreen')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.fill_between(self.bet_history['bet_number'],
                        0,
                        self.bet_history['cumulative_profit'],
                        where=self.bet_history['cumulative_profit'] >= 0,
                        alpha=0.3, color='green')
        ax3.fill_between(self.bet_history['bet_number'],
                        0,
                        self.bet_history['cumulative_profit'],
                        where=self.bet_history['cumulative_profit'] < 0,
                        alpha=0.3, color='red')
        ax3.set_title('Cumulative Profit', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Bet Number')
        ax3.set_ylabel('Profit ($)')
        ax3.grid(True, alpha=0.3)

        # 4. Drawdown
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.fill_between(self.bet_history['bet_number'],
                        0,
                        self.bet_history['drawdown'],
                        color='red', alpha=0.5)
        ax4.plot(self.bet_history['bet_number'], self.bet_history['drawdown'],
                linewidth=2, color='darkred')
        ax4.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Bet Number')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)

        # 5. Profit Distribution
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(self.bet_history['profit'], bins=30, color='steelblue',
                alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax5.set_title('Profit Distribution', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Profit per Bet ($)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Win/Loss Breakdown
        ax6 = fig.add_subplot(gs[2, 0])
        win_loss_counts = self.bet_history['won'].value_counts()
        colors_pie = ['#ff6b6b', '#51cf66']
        ax6.pie([win_loss_counts.get(False, 0), win_loss_counts.get(True, 0)],
               labels=['Losses', 'Wins'],
               autopct='%1.1f%%',
               colors=colors_pie,
               startangle=90)
        ax6.set_title('Win/Loss Ratio', fontsize=14, fontweight='bold')

        # 7. Bet Size Distribution
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.hist(self.bet_history['bet_amount'], bins=25, color='purple',
                alpha=0.7, edgecolor='black')
        ax7.axvline(x=self.bet_history['bet_amount'].mean(), color='red',
                   linestyle='--', linewidth=2, label='Mean')
        ax7.set_title('Bet Size Distribution', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Bet Amount ($)')
        ax7.set_ylabel('Frequency')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')

        # 8. ROI per Bet
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.scatter(self.bet_history['bet_number'], self.bet_history['roi_per_bet'],
                   c=self.bet_history['won'], cmap='RdYlGn', alpha=0.6, s=30)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax8.set_title('ROI per Bet', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Bet Number')
        ax8.set_ylabel('ROI (%)')
        ax8.grid(True, alpha=0.3)

        plt.suptitle('Tennis Betting Performance Dashboard', fontsize=18, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        else:
            plt.show()

    def plot_rolling_metrics(self, window: int = 20, save_path: str = None):
        """Plot rolling performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Rolling win rate
        rolling_win_rate = self.bet_history['won'].rolling(window=window).mean() * 100
        axes[0, 0].plot(self.bet_history['bet_number'], rolling_win_rate, linewidth=2)
        axes[0, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title(f'{window}-Bet Rolling Win Rate', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Bet Number')
        axes[0, 0].set_ylabel('Win Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)

        # Rolling profit
        rolling_profit = self.bet_history['profit'].rolling(window=window).sum()
        axes[0, 1].plot(self.bet_history['bet_number'], rolling_profit, linewidth=2, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title(f'{window}-Bet Rolling Profit', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Bet Number')
        axes[0, 1].set_ylabel('Profit ($)')
        axes[0, 1].grid(True, alpha=0.3)

        # Rolling ROI
        rolling_roi = (self.bet_history['profit'].rolling(window=window).sum() /
                      self.bet_history['bet_amount'].rolling(window=window).sum()) * 100
        axes[1, 0].plot(self.bet_history['bet_number'], rolling_roi, linewidth=2, color='purple')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title(f'{window}-Bet Rolling ROI', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Bet Number')
        axes[1, 0].set_ylabel('ROI (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # Rolling average edge
        if 'edge' in self.bet_history.columns:
            rolling_edge = self.bet_history['edge'].rolling(window=window).mean() * 100
            axes[1, 1].plot(self.bet_history['bet_number'], rolling_edge, linewidth=2, color='orange')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].set_title(f'{window}-Bet Rolling Average Edge', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Bet Number')
            axes[1, 1].set_ylabel('Edge (%)')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Rolling metrics plot saved to {save_path}")
        else:
            plt.show()

    def export_summary_report(self, output_path: str):
        """Export detailed summary report to CSV"""
        stats = self.generate_summary_stats()

        # Convert to DataFrame
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(output_path, index=False)
        print(f"Summary report exported to {output_path}")

        # Also export detailed metrics by time period
        if 'date' in self.bet_history.columns:
            monthly_stats = self.bet_history.groupby(pd.Grouper(key='date', freq='M')).agg({
                'bet_amount': ['count', 'sum', 'mean'],
                'profit': ['sum', 'mean'],
                'won': ['sum', 'mean']
            })
            monthly_path = output_path.replace('.csv', '_monthly.csv')
            monthly_stats.to_csv(monthly_path)
            print(f"Monthly breakdown exported to {monthly_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Betting Performance Dashboard")
    parser.add_argument('--history', required=True,
                       help='CSV file with betting history')
    parser.add_argument('--bankroll', type=float, default=10000,
                       help='Initial bankroll')
    parser.add_argument('--output_dir', default='results/dashboard',
                       help='Output directory for dashboard files')
    parser.add_argument('--rolling_window', type=int, default=20,
                       help='Window size for rolling metrics')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize dashboard
    dashboard = BettingDashboard(args.history, args.bankroll)

    # Print summary to console
    dashboard.print_summary()

    # Generate timestamp for files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Generate visualizations
    dashboard.plot_dashboard(
        save_path=os.path.join(args.output_dir, f'dashboard_{timestamp}.png')
    )
    dashboard.plot_rolling_metrics(
        window=args.rolling_window,
        save_path=os.path.join(args.output_dir, f'rolling_metrics_{timestamp}.png')
    )

    # Export summary report
    dashboard.export_summary_report(
        output_path=os.path.join(args.output_dir, f'summary_report_{timestamp}.csv')
    )

    print(f"\nAll dashboard files saved to {args.output_dir}")
