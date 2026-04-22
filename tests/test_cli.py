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
