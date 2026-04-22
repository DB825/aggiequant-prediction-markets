"""Backtest correctness tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aggie_pm.backtest import (
    _bet_and_pnl,
    _max_drawdown,
    _sharpe,
    brier_score,
    calibration_bins,
    default_model_factory,
    expected_calibration_error,
    log_loss,
    walk_forward_backtest,
)
from aggie_pm.data import generate_synthetic_markets


# ---------------------------------------------------------------------------
# Scoring primitive properties
# ---------------------------------------------------------------------------


def test_brier_perfect_forecast_is_zero():
    y = np.array([0, 1, 0, 1])
    p = y.astype(float)
    assert brier_score(p, y) == pytest.approx(0.0)


def test_brier_of_50_50_equals_quarter():
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.5)
    assert brier_score(p, y) == pytest.approx(0.25)


def test_log_loss_lower_bounded_by_zero():
    y = np.array([0, 1])
    p = np.array([1e-6, 1 - 1e-6])
    assert log_loss(p, y) >= 0
    assert log_loss(p, y) < 1e-4


def test_log_loss_is_proper_scoring_rule():
    """A constant forecast p minimises expected log-loss when p equals the
    base rate. Check numerically."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=2000)
    best_p = 0.3
    grid = np.linspace(0.05, 0.95, 19)
    losses = [log_loss(np.full_like(y, p, dtype=float), y) for p in grid]
    best_ix = int(np.argmin(losses))
    assert abs(grid[best_ix] - best_p) <= 0.07


def test_calibration_bins_returns_10_bins_by_default():
    p = np.linspace(0, 1, 500, endpoint=False) + 1 / 1000
    y = (p > 0.5).astype(int)
    bins = calibration_bins(p, y)
    assert len(bins) == 10


def test_ece_perfect_calibration_is_small():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, size=5000)
    y = rng.binomial(1, p)
    # Perfect probabilistic forecast -> ECE should be small (noise only).
    assert expected_calibration_error(p, y) < 0.05


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------


def test_bet_and_pnl_no_bet_when_no_edge():
    side, stake, pnl = _bet_and_pnl(
        p_model=0.5, market_prob=0.5, spread=0.02, y=1,
        kelly_fraction=0.25, min_edge=0.02, max_position=0.05, fee_bps=20,
    )
    assert side == "none"
    assert stake == 0.0
    assert pnl == 0.0


def test_bet_and_pnl_yes_win():
    # Model says 0.70, market at 0.55 w/ tiny spread -> clear YES edge.
    side, stake, pnl = _bet_and_pnl(
        p_model=0.70, market_prob=0.55, spread=0.02, y=1,
        kelly_fraction=0.5, min_edge=0.02, max_position=0.10, fee_bps=0,
    )
    assert side == "YES"
    assert 0 < stake <= 0.10
    assert pnl > 0


def test_bet_and_pnl_no_bet_pays_off_when_y_zero():
    side, stake, pnl = _bet_and_pnl(
        p_model=0.20, market_prob=0.45, spread=0.02, y=0,
        kelly_fraction=0.5, min_edge=0.02, max_position=0.10, fee_bps=0,
    )
    assert side == "NO"
    assert pnl > 0


def test_bet_and_pnl_losing_bet_is_negative():
    side, stake, pnl = _bet_and_pnl(
        p_model=0.80, market_prob=0.55, spread=0.02, y=0,  # bet YES, YES lost
        kelly_fraction=0.5, min_edge=0.02, max_position=0.10, fee_bps=0,
    )
    assert side == "YES"
    assert pnl < 0


def test_bet_and_pnl_max_position_caps_stake():
    side, stake, _ = _bet_and_pnl(
        p_model=0.95, market_prob=0.20, spread=0.01, y=1,
        kelly_fraction=1.0, min_edge=0.02, max_position=0.03, fee_bps=0,
    )
    assert side == "YES"
    assert stake == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Drawdown and Sharpe
# ---------------------------------------------------------------------------


def test_max_drawdown_monotonic_up_is_zero():
    path = np.array([1.0, 1.1, 1.2, 1.3])
    assert _max_drawdown(path) == pytest.approx(0.0)


def test_max_drawdown_matches_manual():
    path = np.array([1.0, 1.2, 0.9, 1.1, 0.6])
    # peak reaches 1.2, trough 0.6 -> dd = (0.6 - 1.2) / 1.2 = -0.5
    assert _max_drawdown(path) == pytest.approx(-0.5)


def test_sharpe_zero_variance_is_nan():
    r = np.zeros(50)
    s = _sharpe(r)
    assert np.isnan(s)


def test_sharpe_positive_on_positive_drift():
    rng = np.random.default_rng(2)
    r = rng.normal(0.002, 0.01, size=500)
    assert _sharpe(r) > 0


# ---------------------------------------------------------------------------
# Walk-forward end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bt_result():
    df = generate_synthetic_markets(n_events=600, seed=55)
    return walk_forward_backtest(
        df, default_model_factory, n_folds=3, min_train_frac=0.4,
    )


def test_walk_forward_runs_all_models(bt_result):
    expected = {"market_prior", "base_rate", "logistic", "knn", "gbm"}
    assert expected <= set(bt_result.results.keys())


def test_walk_forward_leaderboard_shape(bt_result):
    df = bt_result.summary_table()
    for col in ("model", "brier", "log_loss", "n_bets", "sharpe", "max_drawdown"):
        assert col in df.columns
    assert len(df) == len(bt_result.results)


def test_market_prior_places_no_bets(bt_result):
    r = bt_result.results["market_prior"]
    assert r.n_bets == 0
    assert r.gross_pnl == pytest.approx(0.0)


def test_logistic_is_competitive_with_market_log_loss(bt_result):
    """The logistic model should be at least within spitting distance of the
    market's log-loss, since it has the market logit as a feature."""
    r_log = bt_result.results["logistic"].log_loss
    r_mkt = bt_result.results["market_prior"].log_loss
    assert r_log < r_mkt * 1.15


def test_walk_forward_rejects_too_large_train_frac():
    df = generate_synthetic_markets(n_events=50, seed=1)
    with pytest.raises(ValueError):
        walk_forward_backtest(df, default_model_factory, min_train_frac=1.5)
