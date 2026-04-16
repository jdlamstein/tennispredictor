"""Betting performance and probability calibration metrics.

All functions are pure — they take numpy arrays and return scalars.
The backtester calls these after each simulation run to populate the
metrics dict.

Metric targets (from plan):
  ROI       > 5%  long-term
  Yield     > 3%  per bet to be sustainable
  Brier     < 0.2
  CLV       > 0   consistently (beating closing line)
  Sharpe    > 1.0
  Drawdown  < 30%
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Small epsilon to prevent log(0) in probability-based metrics.
_EPS = 1e-12


def compute_roi(stakes: np.ndarray, payouts: np.ndarray) -> float:
    """Return on investment as a percentage.

    Parameters
    ----------
    stakes : np.ndarray, shape (n_bets,)
        Amount staked per bet.
    payouts : np.ndarray, shape (n_bets,)
        Total payout received per bet (0 if lost, stake * odds if won).

    Returns
    -------
    float
        ``(total_payout - total_staked) / total_staked * 100``.
        Negative means net loss.
    """
    total_staked = float(np.sum(stakes))
    if total_staked == 0.0:
        return 0.0
    profit = float(np.sum(payouts)) - total_staked
    return profit / total_staked * 100.0


def compute_yield(stakes: np.ndarray, payouts: np.ndarray) -> float:
    """Average profit per bet as a fraction of average stake.

    Parameters
    ----------
    stakes : np.ndarray — stake per bet.
    payouts : np.ndarray — payout per bet.

    Returns
    -------
    float
        Mean profit per bet divided by mean stake, expressed as a percentage.
    """
    n = len(stakes)
    if n == 0:
        return 0.0
    mean_stake = float(np.mean(stakes))
    if mean_stake == 0.0:
        return 0.0
    profits = payouts - stakes
    return float(np.mean(profits)) / mean_stake * 100.0


def compute_brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes.

    Lower is better. A random 50/50 predictor scores 0.25. A perfect
    predictor scores 0.0.

    Parameters
    ----------
    probs : np.ndarray, shape (n,)
        Predicted probability of the positive outcome (player 1 wins).
    outcomes : np.ndarray, shape (n,)
        Binary actual outcomes — 1 if player 1 won, 0 otherwise.

    Returns
    -------
    float
        Brier score in [0, 1].
    """
    probs = np.clip(probs, _EPS, 1.0 - _EPS)
    return float(np.mean((probs - outcomes) ** 2))


def compute_log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Binary cross-entropy loss.

    Parameters
    ----------
    probs : np.ndarray — predicted probability of player 1 winning.
    outcomes : np.ndarray — binary actual outcomes.

    Returns
    -------
    float
        Log-loss (lower is better). sklearn-compatible.
    """
    probs = np.clip(probs, _EPS, 1.0 - _EPS)
    return float(-np.mean(
        outcomes * np.log(probs) + (1.0 - outcomes) * np.log(1.0 - probs)
    ))


def compute_ece(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Measures how well predicted probabilities match empirical win rates.
    A perfectly calibrated model has ECE = 0.

    Parameters
    ----------
    probs : np.ndarray — predicted probabilities in [0, 1].
    outcomes : np.ndarray — binary actual outcomes.
    n_bins : int — number of equal-width probability bins.

    Returns
    -------
    float
        Weighted average of |predicted_prob - empirical_freq| across bins.
    """
    n = len(probs)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        bin_probs = probs[mask]
        bin_outcomes = outcomes[mask]
        bin_size = len(bin_probs)
        mean_pred = float(np.mean(bin_probs))
        empirical_freq = float(np.mean(bin_outcomes))
        ece += (bin_size / n) * abs(mean_pred - empirical_freq)

    return ece


def compute_sharpe(
    bet_returns: np.ndarray,
    periods_per_year: int = 52,
) -> float:
    """Annualised Sharpe ratio of per-bet returns.

    Parameters
    ----------
    bet_returns : np.ndarray — profit / stake for each bet (e.g., +1.0, -1.0).
    periods_per_year : int — assumed betting frequency for annualisation.
        52 ≈ weekly, 365 ≈ daily. Default 52 (weekly ATP schedule).

    Returns
    -------
    float
        Annualised Sharpe ratio. > 1.0 is considered acceptable.
    """
    if len(bet_returns) < 2:
        return 0.0
    mean_ret = float(np.mean(bet_returns))
    std_ret = float(np.std(bet_returns, ddof=1))
    if std_ret == 0.0:
        return 0.0
    return (mean_ret / std_ret) * np.sqrt(periods_per_year)


def compute_max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """Maximum peak-to-trough decline in cumulative P&L.

    Parameters
    ----------
    cumulative_pnl : np.ndarray — running sum of profit/loss after each bet.
        First element should be 0 (no bets placed yet).

    Returns
    -------
    float
        Maximum drawdown as a positive fraction of the bankroll peak.
        E.g., 0.25 means the worst decline was 25% of the high-water mark.
    """
    if len(cumulative_pnl) < 2:
        return 0.0

    peak = cumulative_pnl[0]
    max_dd = 0.0

    for value in cumulative_pnl[1:]:
        if value > peak:
            peak = value
        if peak > 0.0:
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown

    return max_dd


def compute_clv(
    model_probs: np.ndarray,
    closing_probs: np.ndarray,
) -> float:
    """Mean closing-line value (CLV) across all bets.

    CLV measures whether our predicted probabilities beat the bookmaker's
    closing line (the final pre-match market probability). Consistently
    positive CLV is the strongest indicator of genuine betting edge.

    Parameters
    ----------
    model_probs : np.ndarray — our model's probability for each bet.
    closing_probs : np.ndarray — de-vigged closing bookmaker probability.

    Returns
    -------
    float
        Mean difference ``model_prob - closing_prob``. Positive means we
        consistently predicted more accurately than the closing market.
    """
    return float(np.mean(model_probs - closing_probs))


def compute_all(
    stakes: np.ndarray,
    payouts: np.ndarray,
    model_probs: np.ndarray,
    outcomes: np.ndarray,
    closing_probs: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the full metric suite in one call.

    Parameters
    ----------
    stakes : (n_bets,) — stake per bet.
    payouts : (n_bets,) — total payout per bet.
    model_probs : (n_bets,) — predicted probability of the bet winning.
    outcomes : (n_bets,) — actual binary outcome (1 = win, 0 = loss).
    closing_probs : (n_bets,) or None — closing-line probability. If None,
        CLV is omitted from the result dict.

    Returns
    -------
    dict[str, float]
        Keys: roi, yield, brier_score, log_loss, ece, sharpe, max_drawdown,
        n_bets, win_rate. Optionally: clv.
    """
    n = len(stakes)
    if n == 0:
        logger.warning("compute_all called with no bets — returning zeros")
        return {k: 0.0 for k in ("roi", "yield", "brier_score", "log_loss",
                                  "ece", "sharpe", "max_drawdown", "n_bets",
                                  "win_rate")}

    profits = payouts - stakes
    cumulative_pnl = np.concatenate([[0.0], np.cumsum(profits)])
    win_mask = payouts > 0
    bet_returns = np.where(win_mask, payouts / stakes - 1.0, -1.0)

    result: dict[str, float] = {
        "n_bets": float(n),
        "win_rate": float(np.mean(win_mask)) * 100.0,
        "roi": compute_roi(stakes, payouts),
        "yield": compute_yield(stakes, payouts),
        "brier_score": compute_brier_score(model_probs, outcomes),
        "log_loss": compute_log_loss(model_probs, outcomes),
        "ece": compute_ece(model_probs, outcomes),
        "sharpe": compute_sharpe(bet_returns),
        "max_drawdown": compute_max_drawdown(cumulative_pnl) * 100.0,  # as %
    }

    if closing_probs is not None:
        result["clv"] = compute_clv(model_probs, closing_probs)

    return result
