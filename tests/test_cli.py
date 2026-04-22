"""CLI smoke test."""

from __future__ import annotations

from pathlib import Path

from aggie_pm.cli import main


def test_cli_run_synthetic(capsys, tmp_path):
    rc = main(["run", "--n-events", "300", "--folds", "2",
               "--out", str(tmp_path / "reports")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "walk-forward backtest" in out
    assert "Leaderboard" in out
    assert (tmp_path / "reports" / "summary.txt").exists()
    assert (tmp_path / "reports" / "leaderboard.csv").exists()
