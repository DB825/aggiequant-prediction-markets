"""CLI smoke test."""

from __future__ import annotations

from pathlib import Path

from aggie_pm.cli import main


def test_cli_run_synthetic(capsys, tmp_path):
    rc = main(["run", "--n-events", "300", "--folds", "2",
               "--out", str(tmp_path / "reports")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dataset profile" in out
    assert "walk-forward backtest" in out
    assert "Leaderboard" in out
    assert (tmp_path / "reports" / "summary.txt").exists()
    assert (tmp_path / "reports" / "leaderboard.csv").exists()
    assert (tmp_path / "reports" / "dataset_summary.csv").exists()
    assert (tmp_path / "reports" / "normalized_dataset.csv").exists()


def test_cli_profile_saved_csv(capsys, tmp_path):
    from aggie_pm.data import generate_synthetic_markets

    csv_path = tmp_path / "markets.csv"
    generate_synthetic_markets(n_events=80, seed=333).to_csv(csv_path, index=False)

    rc = main(["profile", "--csv", str(csv_path), "--out", str(tmp_path / "profile")])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Dataset profile" in out
    assert (tmp_path / "profile" / "dataset_summary.csv").exists()


def test_cli_run_with_sweep(capsys, tmp_path):
    rc = main([
        "run",
        "--n-events",
        "220",
        "--folds",
        "2",
        "--sweep",
        "--sweep-kelly",
        "0.05",
        "--sweep-min-edge",
        "0.02",
        "--sweep-max-position",
        "0.01",
        "--out",
        str(tmp_path / "reports"),
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Trading-rule sweep" in out
    assert (tmp_path / "reports" / "trading_sweep.csv").exists()


def test_cli_ladder_study(capsys, tmp_path):
    rows = [
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
            "raw_yes_ask": 0.40,
            "open_ts": 1,
            "resolve_ts": 10,
            "snapshot_ts": 1,
            "label_available_ts": 10,
            "resolved": 1,
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
            "raw_yes_bid": 0.55,
            "raw_yes_ask": 0.56,
            "open_ts": 1,
            "resolve_ts": 10,
            "snapshot_ts": 1,
            "label_available_ts": 10,
            "resolved": 0,
        },
    ]
    csv_path = tmp_path / "snapshots.csv"
    import pandas as pd

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    rc = main([
        "ladder-study",
        "--csv",
        str(csv_path),
        "--out",
        str(tmp_path / "ladder"),
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "ladder-study report" in out
    assert (tmp_path / "ladder" / "ladder_study.md").exists()
    assert (tmp_path / "ladder" / "pair_trades.csv").exists()
