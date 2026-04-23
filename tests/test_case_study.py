"""Kalshi point-in-time case-study tests."""

from __future__ import annotations

import pandas as pd
import pytest

from aggie_pm.case_study import build_snapshot_dataset, candlesticks_to_frame


def test_candlesticks_to_frame_normalizes_nested_fields():
    frame = candlesticks_to_frame(
        "KXTEST-1",
        [
            {
                "end_period_ts": 100,
                "yes_bid": {"open": "0.40", "high": "0.45", "low": "0.35", "close": "0.42"},
                "yes_ask": {"open": "0.46", "high": "0.50", "low": "0.44", "close": "0.47"},
                "price": {
                    "open": "0.43",
                    "high": "0.46",
                    "low": "0.40",
                    "close": "0.44",
                    "previous": "0.41",
                },
                "volume": "10.00",
                "open_interest": "20.00",
            }
        ],
    )

    assert frame.loc[0, "ticker"] == "KXTEST-1"
    assert frame.loc[0, "yes_bid_close"] == 0.42
    assert frame.loc[0, "yes_ask_close"] == 0.47
    assert frame.loc[0, "candle_volume"] == 10.0


def test_build_snapshot_dataset_uses_candles_not_terminal_prices():
    close = pd.Timestamp("2026-01-31T00:00:00Z")
    markets = pd.DataFrame(
        [
            {
                "event_id": "KXTEST-1",
                "event_ticker": "KXTEST",
                "series_ticker": "KXTEST",
                "category": "macro",
                "question": "Above threshold?",
                "open_time": "2026-01-01T00:00:00Z",
                "close_time": close.isoformat(),
                "resolved": 1,
                "market_prob": 0.9999,
                "market_spread": 0.9998,
                "event_market_count": 3,
            }
        ]
    )
    candles = candlesticks_to_frame(
        "KXTEST-1",
        [
            {
                "end_period_ts": int((close - pd.Timedelta(days=8)).timestamp()),
                "yes_bid": {"close": "0.30"},
                "yes_ask": {"close": "0.36"},
                "price": {
                    "open": "0.20",
                    "high": "0.40",
                    "low": "0.20",
                    "close": "0.33",
                    "previous": "0.25",
                },
                "volume": "100.00",
                "open_interest": "150.00",
            },
            {
                "end_period_ts": int((close - pd.Timedelta(days=1)).timestamp()),
                "yes_bid": {"close": "0.80"},
                "yes_ask": {"close": "0.86"},
                "price": {
                    "open": "0.70",
                    "high": "0.90",
                    "low": "0.70",
                    "close": "0.82",
                    "previous": "0.75",
                },
                "volume": "200.00",
                "open_interest": "300.00",
            },
        ],
    )

    snapshots = build_snapshot_dataset(markets, {"KXTEST-1": candles}, horizon_days=(7, 1))

    assert len(snapshots) == 2
    seven_day = snapshots[snapshots["horizon_days"] == 7].iloc[0]
    one_day = snapshots[snapshots["horizon_days"] == 1].iloc[0]
    assert seven_day["market_prob"] == pytest.approx(0.33)
    assert one_day["market_prob"] == pytest.approx(0.83)
    assert seven_day["market_spread"] == pytest.approx(0.06)
    assert seven_day["label_available_ts"] > seven_day["snapshot_ts"]
