"""Step-by-step walkthrough of the aggie_pm pipeline.

Run this as a script::

    python notebooks/01_walkthrough.py

or open it in VS Code / Jupyter - each ``# %%`` marker is a cell.

The goal is to show *what* each stage does on a small example before
running the full CLI. Reading this once is the fastest way to
understand the codebase.
"""

# %% [markdown]
# # aggie_pm walkthrough
#
# Five cells:
# 1. Generate a small synthetic dataset.
# 2. Build features and inspect the design matrix.
# 3. Fit the model zoo on a training split.
# 4. Score every model on a held-out split.
# 5. Run the walk-forward backtest end-to-end and print the leaderboard.

# %%
import numpy as np
import pandas as pd

from aggie_pm.data import generate_synthetic_markets
from aggie_pm.features import build_features
from aggie_pm.models import train_model_zoo
from aggie_pm.backtest import (
    walk_forward_backtest,
    default_model_factory,
    brier_score,
    log_loss,
    expected_calibration_error,
)
from aggie_pm.report import format_report

# %% [markdown]
# ## 1. Generate data

# %%
df = generate_synthetic_markets(n_events=800, seed=2026)
print(df.head())
print(f"\n{len(df)} events, {df['category'].nunique()} categories")
print(df.groupby("category")["resolved"].mean().round(3).rename("base_rate_yes"))

# %% [markdown]
# ## 2. Build features
#
# Note the leak-safe contract: base rates are computed on the training
# split and passed unchanged into the test split.

# %%
train = df.iloc[:600].copy()
test = df.iloc[600:].copy()

fm_train, cat_rates = build_features(train)
fm_test, _ = build_features(test, category_base_rates=cat_rates)

print("feature names:", fm_train.feature_names)
print("X_train shape:", fm_train.X.shape)
print("X_test  shape:", fm_test.X.shape)

# %% [markdown]
# ## 3. Fit the model zoo

# %%
zoo = train_model_zoo(fm_train)
print("models:", [m.name for m in zoo])

# %% [markdown]
# ## 4. Score on the held-out split

# %%
rows = []
for m in zoo:
    p = m.predict(fm_test)
    rows.append(
        {
            "model": m.name,
            "brier": brier_score(p, fm_test.y),
            "log_loss": log_loss(p, fm_test.y),
            "ece": expected_calibration_error(p, fm_test.y),
        }
    )
print(pd.DataFrame(rows).sort_values("log_loss").to_string(index=False))

# %% [markdown]
# ## 5. Full walk-forward backtest
#
# This is what ``aggie-pm run`` wraps. Fewer folds here to keep the
# walkthrough fast.

# %%
result = walk_forward_backtest(
    df,
    default_model_factory,
    n_folds=3,
    kelly_fraction=0.25,
    min_edge=0.02,
    max_position=0.05,
    fee_bps=20.0,
)
print(format_report(result))
