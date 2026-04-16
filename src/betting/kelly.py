"""Kelly Criterion fractional bet sizing.

The Kelly Criterion finds the stake fraction that maximises long-run bankroll
growth (geometric mean). Full Kelly is theoretically optimal but sensitive to
probability estimation errors. Fractional Kelly (quarter- to half-Kelly) is
standard practice for model-based betting.

Reference
---------
Kelly, J.L. (1956). "A New Interpretation of Information Rate."
Bell System Technical Journal, 35(4), 917-926.

Thorp, E.O. (2008). "The Kelly Criterion in Blackjack Sports Betting and the
Stock Market." Handbook of Asset and Liability Management.
"""

from __future__ import annotations

import math


def compute_kelly_stake(
    p: float,
    odds: float,
    bankroll: float,
    fraction: float = 0.25,
) -> float:
    """Return the recommended bet size for one match using fractional Kelly.

    Parameters
    ----------
    p : float
        Model's probability that the chosen side wins. Must be in (0, 1).
    odds : float
        Decimal odds offered by the bookmaker (e.g., 2.0 = even money).
        Net profit per unit staked = ``odds - 1``.
    bankroll : float
        Current available capital. Must be positive.
    fraction : float
        Kelly multiplier applied to the raw Kelly fraction.
        Common values: 1.0 (full), 0.5 (half), 0.25 (quarter-Kelly).
        Default 0.25 provides conservative sizing under model uncertainty.

    Returns
    -------
    float
        Recommended stake in the same units as ``bankroll``.
        Returns 0.0 when the bet has zero or negative expected value.
        Capped at ``bankroll`` so the stake never exceeds available capital.

    Raises
    ------
    ValueError
        If ``p`` is outside (0, 1), ``odds`` ≤ 1, ``bankroll`` ≤ 0, or
        ``fraction`` is not in (0, 1].

    Examples
    --------
    >>> compute_kelly_stake(p=0.6, odds=2.0, bankroll=1000.0)
    50.0  # quarter-Kelly on 20% edge at even money

    >>> compute_kelly_stake(p=0.4, odds=2.0, bankroll=1000.0)
    0.0   # negative EV — do not bet
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    if odds <= 1.0:
        raise ValueError(f"odds must be > 1.0, got {odds}")
    if bankroll <= 0.0:
        raise ValueError(f"bankroll must be positive, got {bankroll}")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    b = odds - 1.0          # net profit per $1 staked
    q = 1.0 - p             # probability of losing

    # Full Kelly fraction
    kelly = (p * b - q) / b

    if kelly <= 0.0:
        return 0.0

    stake = fraction * kelly * bankroll
    return min(stake, bankroll)


def expected_value(p: float, odds: float) -> float:
    """Compute the expected value of a $1 bet.

    Parameters
    ----------
    p : float — probability the bet wins.
    odds : float — decimal odds.

    Returns
    -------
    float
        Expected profit per $1 staked. Positive → bet has edge.
    """
    return p * (odds - 1.0) - (1.0 - p)


def devig(odds_w: float, odds_l: float) -> tuple[float, float]:
    """Remove bookmaker margin (vig) from a two-outcome market.

    Parameters
    ----------
    odds_w : float — decimal odds for the winner side.
    odds_l : float — decimal odds for the loser side.

    Returns
    -------
    tuple[float, float]
        Fair probability for the winner and loser respectively.
        Each returned value is in (0, 1) and they sum to 1.0.

    Examples
    --------
    >>> devig(1.91, 1.91)  # typical 5% vig on even-money market
    (0.5, 0.5)
    """
    implied_w = 1.0 / odds_w
    implied_l = 1.0 / odds_l
    total = implied_w + implied_l   # > 1.0 due to vig
    return implied_w / total, implied_l / total
