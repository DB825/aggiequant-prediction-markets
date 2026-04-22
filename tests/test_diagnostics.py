"""Dataset diagnostics tests."""

from __future__ import annotations

import pandas as pd

from aggie_pm.data import generate_synthetic_markets
from aggie_pm.diagnostics import format_dataset_profile, profile_market_dataset, save_dataset_profile


def test_profile_market_dataset_returns_summary_and_slices():
    df = generate_synthetic_markets(n_events=300, seed=202)
    df["feat_liquidity"] = range(len(df))

    profile = profile_market_dataset(df)

    assert {"dataset_summary", "missingness", "slices"} <= set(profile)
    summary = profile["dataset_summary"].iloc[0]
    assert summary["n_events"] == 300
    assert summary["market_log_loss"] > 0
    assert {"category", "market_prob_bucket", "spread_bucket"} <= set(profile["slices"]["slice"])


def test_format_dataset_profile_mentions_core_counts():
    df = generate_synthetic_markets(n_events=100, seed=203)
    text = format_dataset_profile(profile_market_dataset(df))
    assert "events=100" in text
    assert "market_log_loss" in text


def test_save_dataset_profile_writes_flat_csvs(tmp_path):
    df = generate_synthetic_markets(n_events=120, seed=204)
    out = save_dataset_profile(profile_market_dataset(df), tmp_path)

    assert (out / "dataset_summary.csv").exists()
    assert (out / "missingness.csv").exists()
    assert (out / "slices.csv").exists()
    assert (out / "dataset_profile.txt").exists()
    loaded = pd.read_csv(out / "dataset_summary.csv")
    assert int(loaded.loc[0, "n_events"]) == 120
