"""Kalshi adapter tests."""

from __future__ import annotations

import numpy as np

from aggie_pm.features import build_features
from aggie_pm.kalshi import (
    build_orderbook_features,
    canonicalize_category,
    kalshi_markets_to_dataframe,
)


RAW_MARKETS = [
    {
        "ticker": "KXNBA-26APR01-BOS",
        "event_ticker": "KXNBA-26APR01",
        "title": "Will Boston win the game?",
        "yes_sub_title": "Boston wins",
        "open_time": "2026-04-01T15:00:00Z",
        "settlement_ts": "2026-04-02T02:00:00Z",
        "yes_bid_dollars": "0.5200",
        "yes_ask_dollars": "0.5800",
        "last_price_dollars": "0.5700",
        "previous_price_dollars": "0.5000",
        "volume_fp": "1234.00",
        "volume_24h_fp": "400.00",
        "open_interest_fp": "900.00",
        "result": "yes",
    },
    {
        "ticker": "KXCPI-26APR-CPI",
        "event_ticker": "KXCPI-26APR",
        "category": "economics",
        "title": "Will CPI come in above expectations?",
        "open_time": "2026-04-03T15:00:00Z",
        "settlement_ts": "2026-04-04T15:00:00Z",
        "yes_bid_dollars": "0.3000",
        "yes_ask_dollars": "0.3400",
        "last_price_dollars": "0.3200",
        "previous_price_dollars": "0.3500",
        "volume_fp": "300.00",
        "volume_24h_fp": "90.00",
        "open_interest_fp": "100.00",
        "result": "no",
    },
]


def test_canonicalize_category_maps_raw_and_text():
    assert canonicalize_category("economics") == "macro"
    assert canonicalize_category("", text="NBA finals winner") == "sports"
    assert canonicalize_category("unknown") == "other"


def test_kalshi_markets_to_dataframe_maps_schema_and_prices():
    df = kalshi_markets_to_dataframe(RAW_MARKETS)
    assert len(df) == 2
    assert {
        "event_id",
        "category",
        "question",
        "market_prob",
        "market_spread",
        "open_ts",
        "resolve_ts",
        "resolved",
    } <= set(df.columns)
    assert set(df["resolved"]) == {0, 1}
    assert set(df["category"]) == {"sports", "macro"}
    np.testing.assert_allclose(df.loc[df["category"] == "sports", "market_prob"], [0.55])
    np.testing.assert_allclose(df.loc[df["category"] == "sports", "market_spread"], [0.06])


def test_orderbook_features_feed_feature_builder():
    df = build_orderbook_features(kalshi_markets_to_dataframe(RAW_MARKETS))
    for col in ("feat_signal", "feat_momentum", "feat_dispersion", "feat_liquidity"):
        assert col in df.columns
        assert np.isfinite(df[col]).all()

    fm, _ = build_features(df)
    assert "feat_liquidity" in fm.feature_names
    assert np.isfinite(fm.X).all()
