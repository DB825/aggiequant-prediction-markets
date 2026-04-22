"""Pareto-front model selection tests."""

from __future__ import annotations

import pandas as pd

from aggie_pm.pareto import pareto_front, pareto_mask, rank_by_domination_count


def test_pareto_front_keeps_non_dominated_rows():
    df = pd.DataFrame(
        [
            {"model": "low_loss", "log_loss": 0.40, "sharpe": 0.2, "max_drawdown": -0.30},
            {"model": "balanced", "log_loss": 0.42, "sharpe": 0.9, "max_drawdown": -0.18},
            {"model": "dominated", "log_loss": 0.50, "sharpe": 0.1, "max_drawdown": -0.40},
        ]
    )
    front = pareto_front(
        df,
        objectives=(("log_loss", "min"), ("sharpe", "max"), ("max_drawdown", "max")),
    )
    assert set(front["model"]) == {"low_loss", "balanced"}


def test_domination_count_matches_mask():
    df = pd.DataFrame(
        [
            {"model": "a", "log_loss": 0.4, "sharpe": 1.0},
            {"model": "b", "log_loss": 0.5, "sharpe": 0.9},
            {"model": "c", "log_loss": 0.6, "sharpe": 0.8},
        ]
    )
    mask = pareto_mask(df, objectives=(("log_loss", "min"), ("sharpe", "max")))
    counts = rank_by_domination_count(df, objectives=(("log_loss", "min"), ("sharpe", "max")))
    assert mask.tolist() == [True, False, False]
    assert counts.tolist() == [0, 1, 2]
