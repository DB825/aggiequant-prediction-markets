"""Synthetic DGP tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aggie_pm.data import (
    CATEGORIES,
    generate_synthetic_markets,
    load_markets_csv,
)


def test_generate_shape_and_schema():
    df = generate_synthetic_markets(n_events=500, seed=1)
    assert len(df) == 500
    required = {
        "event_id",
        "category",
        "question",
        "market_prob",
        "market_spread",
        "open_ts",
        "resolve_ts",
        "true_prob",
        "resolved",
        "feat_signal",
        "feat_momentum",
        "feat_dispersion",
    }
    assert required <= set(df.columns)


def test_generate_is_deterministic():
    a = generate_synthetic_markets(n_events=300, seed=123)
    b = generate_synthetic_markets(n_events=300, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_generate_different_seeds_differ():
    a = generate_synthetic_markets(n_events=200, seed=1)
    b = generate_synthetic_markets(n_events=200, seed=2)
    assert not np.allclose(a["market_prob"].to_numpy(), b["market_prob"].to_numpy())


def test_probabilities_are_bounded():
    df = generate_synthetic_markets(n_events=1000, seed=3)
    for col in ("market_prob", "true_prob"):
        assert (df[col] > 0).all() and (df[col] < 1).all()
    assert (df["market_spread"] >= 0.005).all()
    assert (df["market_spread"] <= 0.10).all()


def test_resolved_is_binary():
    df = generate_synthetic_markets(n_events=500, seed=4)
    assert set(df["resolved"].unique()) <= {0, 1}


def test_time_order_non_decreasing():
    df = generate_synthetic_markets(n_events=500, seed=5)
    assert (df["open_ts"].diff().dropna() >= 0).all()
    # resolve_ts must be strictly after open_ts
    assert (df["resolve_ts"] > df["open_ts"]).all()


def test_all_categories_represented():
    df = generate_synthetic_markets(n_events=3000, seed=6)
    assert set(df["category"].unique()) == set(CATEGORIES)


def test_market_is_informative_but_not_perfect():
    # Market logit should correlate with true logit, but not 1.0 - otherwise
    # there's nothing for a model to learn.
    df = generate_synthetic_markets(n_events=3000, seed=8)
    mp = np.clip(df["market_prob"].to_numpy(), 1e-6, 1 - 1e-6)
    tp = np.clip(df["true_prob"].to_numpy(), 1e-6, 1 - 1e-6)
    ml = np.log(mp / (1 - mp))
    tl = np.log(tp / (1 - tp))
    corr = float(np.corrcoef(ml, tl)[0, 1])
    assert 0.5 < corr < 0.98, f"corr={corr}"


def test_load_markets_csv_roundtrip(tmp_path):
    df = generate_synthetic_markets(n_events=100, seed=9)
    path = tmp_path / "mk.csv"
    df.to_csv(path, index=False)
    loaded = load_markets_csv(path)
    assert len(loaded) == 100
    assert set(loaded.columns) >= {
        "event_id", "category", "market_prob", "market_spread",
        "open_ts", "resolve_ts", "resolved",
        "feat_signal", "feat_momentum", "feat_dispersion",
    }


def test_load_markets_csv_missing_optional_features_is_filled(tmp_path):
    df = generate_synthetic_markets(n_events=50, seed=10)
    df = df.drop(columns=["feat_signal", "feat_momentum", "feat_dispersion"])
    path = tmp_path / "mk.csv"
    df.to_csv(path, index=False)
    loaded = load_markets_csv(path)
    for col in ("feat_signal", "feat_momentum", "feat_dispersion"):
        assert col in loaded.columns
        assert (loaded[col] == 0.0).all()


def test_load_markets_csv_rejects_missing_required(tmp_path):
    df = generate_synthetic_markets(n_events=10, seed=11).drop(columns=["market_prob"])
    path = tmp_path / "mk.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_markets_csv(path)
