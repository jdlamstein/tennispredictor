"""Chronological backtesting engine.

Simulates sports betting using historical model predictions against actual
bookmaker odds. Enforces strict chronological ordering to prevent look-ahead
bias — the model's probability for each match uses only information available
before that match.

Expected input format
---------------------
Both ``bets_df`` and ``odds_df`` must be aligned to the same matches (one row
per match). The backtester does not join data — callers are responsible for
merging model predictions with odds before calling ``run``.

Required columns in the combined DataFrame passed to ``run``:
    date          : str or datetime — match date, used only for sorting
    p1_win_prob   : float in (0, 1) — model probability that player 1 wins
    actual_winner : int — 1 if player 1 won, 2 if player 2 won
    p1_odds       : float — decimal odds offered for player 1 to win
    p2_odds       : float — decimal odds offered for player 2 to win

Optional columns:
    p1_closing_odds, p2_closing_odds : float — closing bookmaker odds for CLV
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.betting.kelly import compute_kelly_stake, devig, expected_value
from src.evaluation import metrics as m

logger = logging.getLogger(__name__)

# Columns the backtester reads from the input DataFrame
_REQUIRED_COLS = {"date", "p1_win_prob", "actual_winner", "p1_odds", "p2_odds"}
_CLOSING_COLS = {"p1_closing_odds", "p2_closing_odds"}


@dataclass
class BacktestConfig:
    """Tunable parameters for a single backtest run.

    Attributes
    ----------
    initial_bankroll : float
        Starting capital in any currency unit (e.g., 1000 = 1000 units).
    kelly_fraction : float
        Fractional Kelly multiplier. Default 0.25 (quarter-Kelly).
    min_ev : float
        Minimum expected value threshold to place a bet. Bets with EV below
        this threshold are skipped. Default 0.02 (2% edge required).
    bet_on : str
        Which side to evaluate for betting: ``"favourite"`` (always bet the
        model's preferred side), ``"value"`` (bet only when EV > min_ev on
        either side). Default ``"value"``.
    max_kelly : float | None
        Hard cap on stake as a fraction of bankroll (e.g. 0.05 = 5% max).
        Prevents exponential blow-up from overconfident Kelly sizing.
        None = uncapped (theoretical maximum, not practical).
    """

    initial_bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    min_ev: float = 0.02
    min_edge: float = 0.05   # model_prob - (1/odds) must exceed this
    max_odds: float = 3.0    # skip bets on heavy underdogs
    bet_on: str = "value"
    max_kelly: float | None = 0.05


@dataclass
class BacktestResult:
    """Output of a single backtest run.

    Attributes
    ----------
    bets : pd.DataFrame
        One row per simulated bet with columns:
        date, p1_win_prob, actual_winner, p1_odds, p2_odds, bet_side,
        stake, payout, profit, bankroll_after, ev.
    metrics : dict[str, float]
        Full metric suite from ``src.evaluation.metrics.compute_all``.
    """

    bets: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)


def _validate(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")


def run(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Simulate fractional-Kelly betting over a historical period.

    Parameters
    ----------
    df : pd.DataFrame
        One row per match with required columns (see module docstring).
        Must be sorted chronologically or will be sorted internally.
    config : BacktestConfig | None
        Backtest parameters. Defaults used if None.

    Returns
    -------
    BacktestResult
        ``bets`` DataFrame with trade-by-trade details, plus ``metrics``.
    """
    if config is None:
        config = BacktestConfig()

    _validate(df)

    # Enforce chronological order — critical for no look-ahead bias.
    df = df.sort_values("date").reset_index(drop=True)

    has_closing = _CLOSING_COLS.issubset(df.columns)

    records: list[dict] = []
    bankroll = config.initial_bankroll

    for _, row in df.iterrows():
        p1_prob = float(row["p1_win_prob"])
        p2_prob = 1.0 - p1_prob
        p1_odds = float(row["p1_odds"])
        p2_odds = float(row["p2_odds"])
        actual = int(row["actual_winner"])
        if actual not in (1, 2):
            raise ValueError(f"actual_winner must be 1 or 2, got {actual}")

        # Clip to open interval — some models (e.g., Naive Bayes) return 0.0 or 1.0
        p1_prob = float(np.clip(p1_prob, 1e-6, 1.0 - 1e-6))
        p2_prob = 1.0 - p1_prob

        ev_p1 = expected_value(p1_prob, p1_odds)
        ev_p2 = expected_value(p2_prob, p2_odds)

        # Edge = model_prob - implied_prob (1/odds)
        edge_p1 = p1_prob - 1.0 / p1_odds
        edge_p2 = p2_prob - 1.0 / p2_odds

        # Choose side to bet
        if config.bet_on == "value":
            p1_ok = (ev_p1 >= ev_p2 and ev_p1 > config.min_ev
                     and edge_p1 >= config.min_edge and p1_odds <= config.max_odds)
            p2_ok = (ev_p2 > ev_p1 and ev_p2 > config.min_ev
                     and edge_p2 >= config.min_edge and p2_odds <= config.max_odds)
            if p1_ok:
                bet_side, p_bet, odds_bet, ev_bet = 1, p1_prob, p1_odds, ev_p1
            elif p2_ok:
                bet_side, p_bet, odds_bet, ev_bet = 2, p2_prob, p2_odds, ev_p2
            else:
                continue  # no value — skip
        else:  # favourite
            bet_side, p_bet, odds_bet, ev_bet = (
                (1, p1_prob, p1_odds, ev_p1) if p1_prob >= p2_prob
                else (2, p2_prob, p2_odds, ev_p2)
            )
            implied = 1.0 / odds_bet
            edge_bet = p_bet - implied
            if ev_bet <= config.min_ev or edge_bet < config.min_edge or odds_bet > config.max_odds:
                continue

        if bankroll <= 0:
            logger.warning("Bankroll exhausted — stopping backtest early.")
            break

        stake = compute_kelly_stake(
            p=p_bet, odds=odds_bet, bankroll=bankroll, fraction=config.kelly_fraction
        )
        if config.max_kelly is not None:
            stake = min(stake, bankroll * config.max_kelly)
        if stake == 0.0:
            continue

        won = actual == bet_side
        payout = stake * odds_bet if won else 0.0
        profit = payout - stake
        bankroll += profit

        record: dict = {
            "date": row["date"],
            "p1_win_prob": p1_prob,
            "actual_winner": actual,
            "p1_odds": p1_odds,
            "p2_odds": p2_odds,
            "bet_side": bet_side,
            "stake": stake,
            "payout": payout,
            "profit": profit,
            "bankroll_after": bankroll,
            "ev": ev_bet,
        }

        if has_closing:
            cl_p1, cl_p2 = devig(float(row["p1_closing_odds"]),
                                  float(row["p2_closing_odds"]))
            record["closing_prob"] = cl_p1 if bet_side == 1 else cl_p2
            record["model_prob"] = p_bet

        records.append(record)

    if not records:
        logger.warning("No bets were placed. Check EV threshold and odds columns.")
        bets = pd.DataFrame()
        return BacktestResult(bets=bets, metrics={"n_bets": 0.0})

    bets = pd.DataFrame(records)

    stakes_arr = bets["stake"].to_numpy()
    payouts_arr = bets["payout"].to_numpy()
    model_probs_arr = bets["model_prob"].to_numpy() if "model_prob" in bets.columns else bets["p1_win_prob"].to_numpy()
    outcomes_arr = (bets["actual_winner"] == bets["bet_side"]).astype(float).to_numpy()

    closing_probs_arr = (
        bets["closing_prob"].to_numpy() if "closing_prob" in bets.columns else None
    )

    computed = m.compute_all(
        stakes=stakes_arr,
        payouts=payouts_arr,
        model_probs=model_probs_arr,
        outcomes=outcomes_arr,
        closing_probs=closing_probs_arr,
    )
    computed["final_bankroll"] = bankroll
    computed["total_profit"] = bankroll - config.initial_bankroll

    return BacktestResult(bets=bets, metrics=computed)


def summary(result: BacktestResult) -> str:
    """Return a human-readable summary string of a backtest result."""
    m_dict = result.metrics
    n = int(m_dict.get("n_bets", 0))
    if n == 0:
        return "No bets placed."

    lines = [
        f"Bets placed   : {n}",
        f"Win rate      : {m_dict.get('win_rate', 0.0):.1f}%",
        f"ROI           : {m_dict.get('roi', 0.0):.2f}%",
        f"Yield / bet   : {m_dict.get('yield', 0.0):.2f}%",
        f"Brier score   : {m_dict.get('brier_score', 0.0):.4f}",
        f"Log-loss      : {m_dict.get('log_loss', 0.0):.4f}",
        f"ECE           : {m_dict.get('ece', 0.0):.4f}",
        f"Sharpe        : {m_dict.get('sharpe', 0.0):.2f}",
        f"Max drawdown  : {m_dict.get('max_drawdown', 0.0):.1f}%",
        f"Total profit  : {m_dict.get('total_profit', 0.0):.2f} units",
        f"Final bankroll: {m_dict.get('final_bankroll', 0.0):.2f} units",
    ]
    if "clv" in m_dict:
        lines.insert(3, f"CLV           : {m_dict['clv']:+.4f}")

    return "\n".join(lines)
