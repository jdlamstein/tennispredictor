"""Unit tests for src/evaluation/metrics.py — all functions have hand-calculable values."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import (
    compute_all,
    compute_brier_score,
    compute_clv,
    compute_ece,
    compute_log_loss,
    compute_max_drawdown,
    compute_roi,
    compute_sharpe,
    compute_yield,
)


class TestComputeRoi:
    def test_breakeven(self):
        stakes = np.array([10.0, 10.0])
        payouts = np.array([10.0, 10.0])
        assert compute_roi(stakes, payouts) == pytest.approx(0.0)

    def test_positive_roi(self):
        stakes = np.array([10.0])
        payouts = np.array([12.0])
        assert compute_roi(stakes, payouts) == pytest.approx(20.0)

    def test_total_loss(self):
        stakes = np.array([10.0, 10.0])
        payouts = np.array([0.0, 0.0])
        assert compute_roi(stakes, payouts) == pytest.approx(-100.0)

    def test_zero_stakes_returns_zero(self):
        assert compute_roi(np.array([0.0]), np.array([0.0])) == 0.0


class TestComputeYield:
    def test_empty_returns_zero(self):
        assert compute_yield(np.array([]), np.array([])) == 0.0

    def test_50_percent_yield(self):
        stakes = np.array([10.0, 10.0])
        payouts = np.array([15.0, 15.0])
        # profit = 5 each, mean_stake = 10, yield = 5/10 * 100 = 50%
        assert compute_yield(stakes, payouts) == pytest.approx(50.0)

    def test_zero_mean_stake_returns_zero(self):
        assert compute_yield(np.array([0.0]), np.array([0.0])) == 0.0


class TestComputeBrierScore:
    def test_perfect_predictor_near_zero(self):
        probs = np.array([0.99, 0.01])
        outcomes = np.array([1.0, 0.0])
        assert compute_brier_score(probs, outcomes) < 0.001

    def test_random_predictor_near_quarter(self):
        probs = np.full(1000, 0.5)
        outcomes = np.where(np.arange(1000) % 2 == 0, 1.0, 0.0)
        assert compute_brier_score(probs, outcomes) == pytest.approx(0.25, abs=1e-6)

    def test_worst_predictor_near_one(self):
        # Always wrong: says 0.99 but outcome is 0
        probs = np.array([0.99, 0.99])
        outcomes = np.array([0.0, 0.0])
        assert compute_brier_score(probs, outcomes) > 0.97


class TestComputeLogLoss:
    def test_perfect_predictor_near_zero(self):
        probs = np.array([1.0 - 1e-9, 1e-9])
        outcomes = np.array([1.0, 0.0])
        assert compute_log_loss(probs, outcomes) < 0.01

    def test_coin_flip_near_ln2(self):
        probs = np.full(1000, 0.5)
        outcomes = (np.arange(1000) % 2 == 0).astype(float)
        assert compute_log_loss(probs, outcomes) == pytest.approx(np.log(2), abs=1e-6)


class TestComputeEce:
    def test_perfectly_calibrated_returns_zero(self):
        # Each bin: predicted prob == empirical freq
        rng = np.random.default_rng(1)
        n = 10000
        probs = rng.uniform(0, 1, n)
        # outcomes drawn from Bernoulli(probs)
        outcomes = (rng.random(n) < probs).astype(float)
        ece = compute_ece(probs, outcomes)
        assert ece < 0.05  # well-calibrated random model should have low ECE

    def test_empty_returns_zero(self):
        assert compute_ece(np.array([]), np.array([])) == 0.0

    def test_overconfident_has_high_ece(self):
        # Always says 0.9 but wins only 50% of the time
        probs = np.full(200, 0.9)
        outcomes = (np.arange(200) % 2 == 0).astype(float)
        ece = compute_ece(probs, outcomes)
        assert ece > 0.3


class TestComputeSharpe:
    def test_fewer_than_two_returns_zero(self):
        assert compute_sharpe(np.array([1.0])) == 0.0
        assert compute_sharpe(np.array([])) == 0.0

    def test_constant_returns_zero(self):
        assert compute_sharpe(np.full(100, 1.0)) == 0.0

    def test_positive_mean_positive_sharpe(self):
        returns = np.full(100, 0.1)
        returns[::5] = -0.1  # some variance
        sharpe = compute_sharpe(returns)
        assert sharpe > 0.0


class TestComputeMaxDrawdown:
    def test_no_drawdown(self):
        pnl = np.array([0.0, 10.0, 20.0, 30.0])
        assert compute_max_drawdown(pnl) == 0.0

    def test_known_drawdown(self):
        # Peak at 100, trough at 50 → drawdown = 50%
        pnl = np.array([0.0, 100.0, 50.0, 80.0])
        assert compute_max_drawdown(pnl) == pytest.approx(0.5)

    def test_single_element_returns_zero(self):
        assert compute_max_drawdown(np.array([0.0])) == 0.0

    def test_always_below_zero_no_drawdown(self):
        # Peak never above 0, so no defined drawdown
        pnl = np.array([0.0, -10.0, -20.0])
        assert compute_max_drawdown(pnl) == 0.0


class TestComputeClv:
    def test_positive_clv(self):
        # mean([0.6-0.5, 0.7-0.5, 0.55-0.5]) = mean([0.1, 0.2, 0.05]) = 0.1167
        model = np.array([0.6, 0.7, 0.55])
        closing = np.array([0.5, 0.5, 0.5])
        assert compute_clv(model, closing) == pytest.approx(0.11667, abs=1e-4)

    def test_zero_clv_same_probs(self):
        p = np.array([0.5, 0.6, 0.4])
        assert compute_clv(p, p) == pytest.approx(0.0)


class TestComputeAll:
    def test_zero_bets_returns_zeros(self):
        result = compute_all(
            np.array([]), np.array([]), np.array([]), np.array([])
        )
        assert result["roi"] == 0.0
        assert result["n_bets"] == 0.0

    def test_smoke_test_with_bets(self):
        rng = np.random.default_rng(42)
        n = 50
        stakes = rng.uniform(5, 20, n)
        outcomes = (rng.random(n) > 0.45).astype(float)
        payouts = np.where(outcomes, stakes * 1.9, 0.0)
        probs = rng.uniform(0.4, 0.7, n)
        result = compute_all(stakes, payouts, probs, outcomes)
        assert "roi" in result
        assert "sharpe" in result
        assert "brier_score" in result
        assert result["n_bets"] == pytest.approx(n)

    def test_clv_included_when_closing_probs_provided(self):
        stakes = np.array([10.0])
        payouts = np.array([18.0])
        probs = np.array([0.6])
        outcomes = np.array([1.0])
        closing = np.array([0.5])
        result = compute_all(stakes, payouts, probs, outcomes, closing_probs=closing)
        assert "clv" in result
        assert result["clv"] == pytest.approx(0.1)

    def test_clv_absent_when_not_provided(self):
        stakes = np.array([10.0])
        payouts = np.array([0.0])
        probs = np.array([0.5])
        outcomes = np.array([0.0])
        result = compute_all(stakes, payouts, probs, outcomes)
        assert "clv" not in result
