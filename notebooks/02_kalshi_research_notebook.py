"""Kalshi research notebook for aggie_pm.

Run this as a script::

    python notebooks/02_kalshi_research_notebook.py

or open it in VS Code / Jupyter. The ``# %%`` markers become notebook
cells. The notebook tries to load cached/fresh Kalshi data first and falls
back to synthetic data when the network is unavailable, so it remains useful
inside restricted environments.
"""

# %% [markdown]
# # Kalshi resolved-market research notebook
#
# This notebook is the portfolio-facing path:
#
# 1. Pull or load cached resolved Kalshi markets.
# 2. Normalize them into the package schema.
# 3. Audit the real-data feature set.
# 4. Run the same walk-forward backtest used on synthetic data.
# 5. Select models on a Pareto frontier instead of a single metric.
# 6. Sketch where Prosperity, data engineering, Pareto, sentiment, and
#    broader ML work plug into the next iteration.

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aggie_pm.backtest import default_model_factory, walk_forward_backtest
from aggie_pm.data import generate_synthetic_markets
from aggie_pm.diagnostics import format_dataset_profile, profile_market_dataset
from aggie_pm.features import build_features
from aggie_pm.kalshi import load_kalshi_resolved
from aggie_pm.pareto import pareto_front, rank_by_domination_count

ROOT = Path(__file__).resolve().parents[1]
KALSHI_CACHE = ROOT / "data" / "kalshi_cache"

# %% [markdown]
# ## 1. Load Kalshi data
#
# Kalshi's public market-data endpoints do not require authentication for
# market reads. Older settled markets move to the historical tier, so the
# notebook defaults to `source="historical"` and caches the raw JSON.

# %%
def load_dataset() -> tuple[pd.DataFrame, str]:
    try:
        df = load_kalshi_resolved(
            source="historical",
            max_pages=50,
            page_limit=1000,
            cache_dir=KALSHI_CACHE,
            refresh=False,
        )
        if df.empty:
            raise RuntimeError("Kalshi returned no resolved binary markets.")
        return df, "kalshi"
    except Exception as exc:  # pragma: no cover - notebook resilience
        print(f"Kalshi load skipped: {exc!r}")
        df = generate_synthetic_markets(n_events=1200, seed=20260421)
        df["feat_liquidity"] = 0.0
        return df, "synthetic_fallback"


df, source_name = load_dataset()
print(f"source={source_name} rows={len(df):,} columns={len(df.columns)}")
print(df.head(5).to_string(index=False))

# %% [markdown]
# ## 2. Dataset audit
#
# Before modeling, check whether this is enough data for walk-forward
# evaluation and whether the market itself is already a strong baseline.

# %%
audit = (
    df.groupby("category")
    .agg(
        n=("event_id", "count"),
        yes_rate=("resolved", "mean"),
        avg_market_prob=("market_prob", "mean"),
        avg_spread=("market_spread", "mean"),
    )
    .sort_values("n", ascending=False)
)
print(audit.round(4).to_string())

profile = profile_market_dataset(df)
print("\n" + format_dataset_profile(profile))
print("\nTop dataset slices:")
print(profile["slices"].sort_values("n_events", ascending=False).head(12).round(4).to_string(index=False))

if "feat_liquidity" in df.columns:
    print("\nLiquidity quantiles:")
    print(df["feat_liquidity"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).round(4))

# %% [markdown]
# ## 3. Optional sentiment hook
#
# A sentiment pipeline can join onto this frame before `build_features`.
# The feature builder automatically includes `feat_sentiment` and
# `feat_news_volume` when those columns exist. Replace this neutral stub
# with scores from a news/social/LLM sentiment job keyed by event ticker,
# market ticker, or timestamp.

# %%
df = df.copy()
if "feat_sentiment" not in df.columns:
    df["feat_sentiment"] = 0.0
if "feat_news_volume" not in df.columns:
    df["feat_news_volume"] = 0.0

feature_preview, _ = build_features(df.iloc[: min(len(df), 400)].copy())
print(f"feature_count={len(feature_preview.feature_names)}")
print(pd.Series(feature_preview.feature_names).to_string(index=False))

# %% [markdown]
# ## 4. Walk-forward backtest
#
# This is the same evaluation contract as the CLI. The split is contiguous
# in time; category base rates are frozen from train to test; every model
# pays spread and fee costs before PnL is reported.

# %%
n_folds = min(5, max(2, len(df) // 250))
result = walk_forward_backtest(
    df,
    default_model_factory,
    n_folds=n_folds,
    kelly_fraction=0.25,
    min_edge=0.02,
    max_position=0.05,
    fee_bps=20.0,
)

leaderboard = result.summary_table()
leaderboard["dominated_by_count"] = rank_by_domination_count(leaderboard)
print(leaderboard.round(4).to_string(index=False))

# %% [markdown]
# ## 5. Pareto frontier
#
# A single sorted leaderboard is not enough. In practice, the preferred
# model depends on whether the operator values log loss, calibration,
# final bankroll, Sharpe, or shallower drawdown most.

# %%
front = pareto_front(
    leaderboard,
    objectives=(
        ("log_loss", "min"),
        ("brier", "min"),
        ("ece", "min"),
        ("sharpe", "max"),
        ("max_drawdown", "max"),
        ("final_bankroll", "max"),
    ),
    sort_by="log_loss",
)
print(front.round(4).to_string(index=False))

# %% [markdown]
# ## 6. Research findings and next iterations
#
# Use this final cell as the concise project memo.

# %%
findings = [
    {
        "theme": "Kalshi datasets",
        "finding": "Public resolved markets can be normalized into the same schema as the synthetic DGP, which keeps the pipeline auditable.",
        "next_step": "Pull candlesticks at fixed horizons before settlement so the backtest avoids using an end-of-life snapshot.",
    },
    {
        "theme": "Prosperity-style market making",
        "finding": "Bid/ask midpoint, last-trade pressure, spread-normalized momentum, and liquidity translate naturally into prediction-market features.",
        "next_step": "Add orderbook depth snapshots and inventory-aware constraints for live-paper trading simulations.",
    },
    {
        "theme": "Pareto realizations",
        "finding": "Non-dominated model selection is a better story than ranking only by log loss or Sharpe.",
        "next_step": "Sweep Kelly fraction, min edge, and max position, then select risk settings on a Pareto frontier.",
    },
    {
        "theme": "Data engineering",
        "finding": "The repo already has a bronze/silver/gold shape: raw cached JSON, normalized market schema, and report artifacts.",
        "next_step": "Version each data pull and emit data-quality checks before training.",
    },
    {
        "theme": "Sentiment",
        "finding": "The feature builder now accepts neutral sentiment/news-volume hooks without changing model code.",
        "next_step": "Join event-level news or social sentiment by ticker and timestamp, then test whether it beats the market prior out-of-sample.",
    },
    {
        "theme": "ML techniques",
        "finding": "The current lineup covers interpretable, nonparametric, boosted, calibrated, shrunk, and stacked models.",
        "next_step": "Add time-aware hyperparameter sweeps, permutation importance, and conformal prediction intervals.",
    },
]

memo = pd.DataFrame(findings)
print(memo.to_string(index=False))
