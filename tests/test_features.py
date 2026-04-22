"""Feature-engineering tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aggie_pm.data import CATEGORIES, generate_synthetic_markets
from aggie_pm.features import build_features


def test_feature_matrix_shape_matches_df(small_df):
    fm, _ = build_features(small_df)
    assert fm.X.shape[0] == len(small_df)
    assert fm.X.shape[1] == len(fm.feature_names)
    assert len(fm.y) == len(small_df)


def test_feature_names_are_unique(fm_small):
    assert len(fm_small.feature_names) == len(set(fm_small.feature_names))


def test_market_logit_feature_is_correct(small_df):
    fm, _ = build_features(small_df)
    idx = fm.feature_names.index("market_logit")
    mp = np.clip(small_df["market_prob"].to_numpy(), 1e-6, 1 - 1e-6)
    expected = np.log(mp / (1 - mp))
    np.testing.assert_allclose(fm.X[:, idx], expected, rtol=1e-5)


def test_category_one_hot_sums_to_one(fm_small):
    cat_cols = [fm_small.feature_names.index(f"cat_{c}") for c in CATEGORIES]
    sub = fm_small.X[:, cat_cols]
    sums = sub.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0)


def test_unknown_category_uses_other_bucket(small_df):
    df = small_df.copy()
    df.loc[df.index[0], "category"] = "custom-kalshi-category"
    fm, rates = build_features(df)
    other_ix = fm.feature_names.index("cat_other")
    assert fm.X[0, other_ix] == 1.0
    assert "custom-kalshi-category" in rates


def test_optional_real_data_signals_are_included(small_df):
    df = small_df.copy()
    df["feat_liquidity"] = np.log1p(np.arange(len(df)))
    df["feat_sentiment"] = np.linspace(-1, 1, len(df))
    fm, _ = build_features(df)
    assert "feat_liquidity" in fm.feature_names
    assert "feat_sentiment" in fm.feature_names


def test_base_rates_frozen_prevent_leakage():
    df = generate_synthetic_markets(n_events=600, seed=13)
    train = df.iloc[:400].copy()
    test = df.iloc[400:].copy()

    _, cat_rates = build_features(train)
    # Apply frozen rates to test; the rates must come from train only.
    fm_test, returned = build_features(test, category_base_rates=cat_rates)
    for c in cat_rates:
        assert returned[c] == cat_rates[c]

    # cat_rates computed from test-only data must differ (otherwise DGP issue).
    _, test_rates = build_features(test)
    diffs = [abs(cat_rates[c] - test_rates[c]) for c in cat_rates]
    assert max(diffs) > 1e-3


def test_interaction_features_present(fm_small):
    assert "mkt_logit_x_ttr" in fm_small.feature_names
    assert "mkt_logit_x_spread" in fm_small.feature_names


def test_no_nans_or_infs(fm_small):
    assert np.isfinite(fm_small.X).all()
    assert np.isfinite(fm_small.market_prob).all()
    assert np.isfinite(fm_small.market_spread).all()
