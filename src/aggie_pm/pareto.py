"""Multi-objective model selection via Pareto fronts.

A leaderboard sorted on log-loss hides the fact that the best-calibrated
model is not always the best PnL, nor the best drawdown, nor the best
Sharpe. The honest question is: *which models are Pareto-optimal across
the objectives I actually care about?* Every dominated model can be
discarded without loss; every non-dominated model lives on the efficient
frontier and represents a real tradeoff a portfolio operator would have
to consciously pick.

This module computes non-dominated sets on arbitrary
(objective, direction) tuples. Default objectives for the prediction-
markets pipeline:

- log_loss   (min)
- brier      (min)
- ece        (min)
- sharpe     (max)
- sortino    (max)
- max_drawdown  (max; the column is stored as a negative number like -0.27,
                 so "larger" means "shallower drawdown")

The Pareto front is O(n^2) in the number of candidates, which is fine
for a model leaderboard with a few dozen rows. For larger search spaces
(e.g., hyperparameter sweeps) the same routine scales comfortably to
the thousands.

References
----------
Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary
    Algorithms.* Non-dominated sorting / NSGA-II framing.
Bailey, D., Borwein, J., Lopez de Prado, M., Zhu, Q. (2014). *The
    Probability of Backtest Overfitting.* Motivates reporting the
    frontier rather than a single best-by-sharpe pick.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

Direction = str  # "min" or "max"

DEFAULT_OBJECTIVES: tuple[tuple[str, Direction], ...] = (
    ("log_loss", "min"),
    ("brier", "min"),
    ("ece", "min"),
    ("sharpe", "max"),
    ("max_drawdown", "max"),
)


def _normalise(matrix: np.ndarray, directions: Sequence[Direction]) -> np.ndarray:
    """Flip columns so every objective is *minimise*, then return unchanged.

    We only need the direction-corrected matrix for dominance comparison,
    not the scaled magnitudes, so no per-column rescaling is needed.
    """
    out = matrix.astype(float).copy()
    for j, d in enumerate(directions):
        if d == "max":
            out[:, j] = -out[:, j]
        elif d != "min":
            raise ValueError(f"direction must be 'min' or 'max', got {d!r}")
    return out


def pareto_mask(
    df: pd.DataFrame,
    objectives: Sequence[tuple[str, Direction]] = DEFAULT_OBJECTIVES,
) -> np.ndarray:
    """Return a boolean mask marking Pareto-optimal rows in ``df``.

    A row ``i`` is Pareto-optimal iff no other row ``j`` is weakly better
    on every objective and strictly better on at least one. ``NaN``
    values are treated as "worst possible" so models missing an
    objective never dominate; if every row has NaN for a column, that
    column is ignored.
    """
    cols = [c for c, _ in objectives if c in df.columns]
    dirs = [d for c, d in objectives if c in df.columns]
    if not cols:
        raise ValueError("None of the requested objectives are columns of df.")

    raw = df[cols].to_numpy(dtype=float)
    # Drop columns that are all-NaN so they do not kill the front.
    usable = ~np.all(np.isnan(raw), axis=0)
    raw = raw[:, usable]
    dirs = [d for d, keep in zip(dirs, usable) if keep]

    # NaN becomes +inf under min so it never dominates.
    raw = np.where(np.isnan(raw), np.inf, raw)
    m = _normalise(raw, dirs)

    n = m.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # j dominates i if m[j] <= m[i] everywhere and < somewhere.
        weakly_better = np.all(m <= m[i], axis=1)
        strictly_better = np.any(m < m[i], axis=1)
        dominated_by = weakly_better & strictly_better
        dominated_by[i] = False
        if np.any(dominated_by):
            mask[i] = False
    return mask


def pareto_front(
    df: pd.DataFrame,
    objectives: Sequence[tuple[str, Direction]] = DEFAULT_OBJECTIVES,
    *,
    sort_by: str | None = None,
) -> pd.DataFrame:
    """Return the Pareto-optimal subset of ``df``.

    The returned frame preserves all input columns and is optionally
    sorted by ``sort_by`` (ascending if that column is a "min"
    objective, descending if "max"). If ``sort_by`` is None the
    original row order is preserved.
    """
    mask = pareto_mask(df, objectives)
    out = df.loc[mask].copy()
    if sort_by is not None and sort_by in out.columns:
        direction = next((d for c, d in objectives if c == sort_by), "min")
        out = out.sort_values(sort_by, ascending=(direction == "min")).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def rank_by_domination_count(
    df: pd.DataFrame,
    objectives: Sequence[tuple[str, Direction]] = DEFAULT_OBJECTIVES,
) -> pd.Series:
    """Count, for each row, how many other rows it is dominated by.

    Rows with count 0 are exactly the Pareto front. This gives a useful
    "softness" signal when the strict front is tiny: the rows with
    count 1 or 2 are the near-frontier and worth inspecting before
    discarding.
    """
    cols = [c for c, _ in objectives if c in df.columns]
    dirs = [d for c, d in objectives if c in df.columns]
    raw = df[cols].to_numpy(dtype=float)
    usable = ~np.all(np.isnan(raw), axis=0)
    raw = raw[:, usable]
    dirs = [d for d, keep in zip(dirs, usable) if keep]
    raw = np.where(np.isnan(raw), np.inf, raw)
    m = _normalise(raw, dirs)

    n = m.shape[0]
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        weakly_better = np.all(m <= m[i], axis=1)
        strictly_better = np.any(m < m[i], axis=1)
        dominated_by = weakly_better & strictly_better
        dominated_by[i] = False
        counts[i] = int(np.sum(dominated_by))
    return pd.Series(counts, index=df.index, name="dominated_by_count")


def kelly_fraction_frontier(
    sweep_result: pd.DataFrame,
    *,
    fraction_col: str = "kelly_fraction",
    objectives: Iterable[tuple[str, Direction]] = (("sharpe", "max"), ("max_drawdown", "max"), ("final_bankroll", "max")),
) -> pd.DataFrame:
    """Convenience wrapper for Kelly-fraction sweeps.

    Pass a DataFrame indexed by (or with a column of) Kelly fractions and
    rows of backtest metrics. Returns the Pareto-optimal fractions on the
    (Sharpe, max-DD, bankroll) surface, which is the tradeoff the user
    is implicitly making when they pick a single fraction.
    """
    objectives = tuple(objectives)
    frontier = pareto_front(sweep_result, objectives=objectives, sort_by=None)
    if fraction_col in frontier.columns:
        frontier = frontier.sort_values(fraction_col).reset_index(drop=True)
    return frontier
