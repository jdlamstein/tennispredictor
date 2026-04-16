"""Tests for Kelly Criterion stake-sizing invariants.

The Kelly function doesn't exist yet (Phase 1). These tests define the
expected behavior so implementation can be driven by them (TDD).

Run `pytest -k kelly` — these will be collected but will fail until
src/betting/kelly.py is implemented.
"""

import sys
import os
import math

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _import_kelly():
    """Deferred import so tests can be collected before the module exists."""
    try:
        from src.betting.kelly import compute_kelly_stake
        return compute_kelly_stake
    except ImportError:
        pytest.skip("src/betting/kelly.py not yet implemented (Phase 1)")


class TestKellyStakeBounds:
    """Kelly stakes must be non-negative and capped at the bankroll."""

    def test_positive_ev_yields_positive_stake(self) -> None:
        """When model probability exceeds implied probability, stake > 0."""
        compute_kelly_stake = _import_kelly()
        # p_model=0.7, decimal_odds=2.0, fraction=1.0 → full Kelly
        stake = compute_kelly_stake(p=0.7, odds=2.0, bankroll=1000.0, fraction=1.0)
        assert stake > 0, "Positive EV should produce a positive stake"

    def test_zero_ev_yields_zero_stake(self) -> None:
        """When model probability equals implied probability, no bet should be placed."""
        compute_kelly_stake = _import_kelly()
        # p=0.5, decimal_odds=2.0 → EV exactly 0
        stake = compute_kelly_stake(p=0.5, odds=2.0, bankroll=1000.0, fraction=1.0)
        assert math.isclose(stake, 0.0, abs_tol=1e-9)

    def test_negative_ev_yields_zero_stake(self) -> None:
        """When the bet has negative EV, stake must be 0 (never bet negative EV)."""
        compute_kelly_stake = _import_kelly()
        # p=0.3, decimal_odds=1.5 → negative EV
        stake = compute_kelly_stake(p=0.3, odds=1.5, bankroll=1000.0, fraction=1.0)
        assert stake == 0.0, "Negative EV should yield zero stake"

    def test_stake_never_exceeds_bankroll(self) -> None:
        """Regardless of edge, stake must not exceed bankroll."""
        compute_kelly_stake = _import_kelly()
        bankroll = 500.0
        stake = compute_kelly_stake(p=0.99, odds=100.0, bankroll=bankroll, fraction=1.0)
        assert stake <= bankroll, "Stake must never exceed the bankroll"

    def test_fractional_kelly_reduces_stake(self) -> None:
        """Quarter-Kelly stake must be lower than full-Kelly stake."""
        compute_kelly_stake = _import_kelly()
        full = compute_kelly_stake(p=0.6, odds=2.0, bankroll=1000.0, fraction=1.0)
        quarter = compute_kelly_stake(p=0.6, odds=2.0, bankroll=1000.0, fraction=0.25)
        assert quarter < full, "Fractional Kelly must produce a smaller stake"

    def test_stake_scales_with_bankroll(self) -> None:
        """Doubling the bankroll should double the stake (Kelly is proportional)."""
        compute_kelly_stake = _import_kelly()
        s1 = compute_kelly_stake(p=0.6, odds=2.0, bankroll=1000.0, fraction=0.5)
        s2 = compute_kelly_stake(p=0.6, odds=2.0, bankroll=2000.0, fraction=0.5)
        assert math.isclose(s2, 2 * s1, rel_tol=1e-9)
