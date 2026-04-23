"""Relative-value ladder tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aggie_pm.relative_value import (
    backtest_pair_trades,
    build_monotonic_pair_opportunities,
    extract_threshold,
    repair_ladder_probabilities,
    select_pair_trades,
)


def _ladder_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "E-T0.0__h1d",
                "market_id": "E-T0.0",
                "event_ticker": "E",
                "horizon_days": 1,
                "category": "macro",
                "question": "Above 0.0%",
                "market_prob": 0.40,
                "market_spread": 0.02,
                "raw_yes_bid": 0.39,
                "raw_yes_ask": 0.41,
                "resolved": 1,
                "open_ts": 1,
                "resolve_ts": 10,
                "snapshot_ts": 1,
                "label_available_ts": 10,
            },
            {
                "event_id": "E-T0.1__h1d",
                "market_id": "E-T0.1",
                "event_ticker": "E",
                "horizon_days": 1,
                "category": "macro",
                "question": "Above 0.1%",
                "market_prob": 0.50,
                "market_spread": 0.02,
                "raw_yes_bid": 0.49,
                "raw_yes_ask": 0.51,
                "resolved": 0,
                "open_ts": 1,
                "resolve_ts": 10,
                "snapshot_ts": 1,
                "label_available_ts": 10,
            },
            {
                "event_id": "E-T0.2__h1d",
                "market_id": "E-T0.2",
                "event_ticker": "E",
                "horizon_days": 1,
                "category": "macro",
                "question": "Above 0.2%",
                "market_prob": 0.20,
                "market_spread": 0.02,
                "raw_yes_bid": 0.19,
                "raw_yes_ask": 0.21,
                "resolved": 0,
                "open_ts": 1,
                "resolve_ts": 10,
                "snapshot_ts": 1,
                "label_available_ts": 10,
            },
        ]
    )


def test_extract_threshold_from_question_and_ticker():
    assert extract_threshold("Above -0.1%", "ignored") == pytest.approx(-0.1)
    assert extract_threshold("No explicit threshold", "KXFED-26JAN-T4.75") == pytest.approx(4.75)
    assert np.isnan(extract_threshold("No threshold", "NOPE"))


def test_repair_ladder_probabilities_enforces_monotonicity():
    repaired = repair_ladder_probabilities(_ladder_df())
    ordered = repaired.sort_values("threshold")

    assert ordered["adjacent_violation_count"].max() == 1
    assert np.all(np.diff(ordered["repaired_prob"]) <= 1e-12)
    assert ordered["abs_repair_gap"].max() > 0


def test_pair_opportunity_backtest_positive_when_ladder_crosses_after_fees():
    df = _ladder_df()
    df.loc[df["market_id"] == "E-T0.0", "raw_yes_ask"] = 0.40
    df.loc[df["market_id"] == "E-T0.1", "raw_yes_bid"] = 0.55
    repaired = repair_ladder_probabilities(df)
    opportunities = build_monotonic_pair_opportunities(repaired, fee_bps=20.0)
    trades = select_pair_trades(opportunities, min_edge=0.0)
    summary = backtest_pair_trades(trades, stake_fraction=0.01)

    assert not trades.empty
    assert trades.iloc[0]["edge_after_fee"] > 0
    assert summary["n_trades"] >= 1
    assert summary["final_bankroll"] > 1.0
