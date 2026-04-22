"""Pretty-print a BacktestResult and save artifacts to disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult, ModelResult
from .diagnostics import format_dataset_profile, profile_market_dataset, save_dataset_profile


def _reliability_sparkline(cal: pd.DataFrame, width: int = 20) -> str:
    """Tiny ASCII reliability diagram.

    Shows empirical_rate minus avg_pred per bin; '|' is perfect, '+' is over,
    '-' is under. Empty bins render as '.'.
    """
    out = []
    for _, row in cal.iterrows():
        n = row["n"]
        if n == 0:
            out.append(".")
            continue
        gap = row["emp_rate"] - row["avg_pred"]
        if abs(gap) < 0.02:
            out.append("|")
        elif gap > 0:
            out.append("+" if gap < 0.1 else "#")
        else:
            out.append("-" if gap > -0.1 else "=")
    return "".join(out).ljust(width)[:width]


def _format_model_block(r: ModelResult) -> str:
    lines = [
        f"-- {r.name} " + "-" * max(60 - len(r.name), 3),
        f"  brier    {r.brier:.4f}    log_loss {r.log_loss:.4f}    ECE {r.ece:.4f}"
        f"    AUC {r.roc_auc:.3f}",
        f"  n_bets   {r.n_bets:>6d}    coverage {r.coverage:.3f}    hit_rate {r.hit_rate:.3f}"
        f"    gross_pnl {r.gross_pnl:+.4f}    bankroll {r.final_bankroll:.4f}",
        f"  sharpe   {r.sharpe:>6.2f}    sortino  {r.sortino:>6.2f}"
        f"    max_dd   {r.max_drawdown:+.3f}",
        f"  turnover {r.turnover:.4f}    avg_edge {r.avg_trade_edge:.4f}"
        f"    profit_factor {r.profit_factor:.3f}    YES/NO {r.n_yes_bets}/{r.n_no_bets}",
        f"  calib    [{_reliability_sparkline(r.calibration)}]  (bins 0.0 -> 1.0; '|' calibrated, +/- off)",
    ]
    if not r.per_category_pnl.empty:
        lines.append("  per-category PnL:")
        for _, row in r.per_category_pnl.iterrows():
            lines.append(
                f"    {row['category']:<14s} n={int(row['n_events']):>4d}"
                f"  bets={int(row['n_bets']):>4d}  pnl={row['pnl']:+.4f}"
            )
    return "\n".join(lines)


def format_report(
    result: BacktestResult,
    dataset_profile: dict[str, pd.DataFrame] | None = None,
) -> str:
    header = [
        "=" * 72,
        "  AggieQuant prediction-markets walk-forward backtest",
        "=" * 72,
        f"  folds: {result.n_folds}    test events: {result.n_events}",
        "",
    ]
    if dataset_profile is not None:
        header.append(format_dataset_profile(dataset_profile))
        header.append("")
    header.extend(["Leaderboard (sorted by log-loss, lower is better):", ""])
    table = result.summary_table()
    display_cols = [
        "model",
        "brier",
        "log_loss",
        "log_loss_edge_vs_market",
        "ece",
        "roc_auc",
        "n_bets",
        "coverage",
        "gross_pnl",
        "final_bankroll",
        "sharpe",
        "max_drawdown",
    ]
    display = table[[c for c in display_cols if c in table.columns]]
    header.append(display.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    header.append("")
    header.append("Per-model detail:")
    blocks = [_format_model_block(r) for r in result.results.values()]
    return "\n".join(header + [""] + blocks)


def save_report(
    result: BacktestResult,
    out_dir: str | Path,
    *,
    dataset: pd.DataFrame | None = None,
    dataset_profile: dict[str, pd.DataFrame] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if dataset_profile is None and dataset is not None:
        dataset_profile = profile_market_dataset(dataset)
    (out / "summary.txt").write_text(format_report(result, dataset_profile), encoding="utf-8")
    result.summary_table().to_csv(out / "leaderboard.csv", index=False)
    if dataset is not None:
        dataset.to_csv(out / "normalized_dataset.csv", index=False)
    if dataset_profile is not None:
        save_dataset_profile(dataset_profile, out)
    for name, r in result.results.items():
        safe = name.replace("(", "_").replace(")", "").replace(",", "_").replace(" ", "")
        r.calibration.to_csv(out / f"calibration_{safe}.csv", index=False)
        r.per_category_pnl.to_csv(out / f"pnl_by_category_{safe}.csv", index=False)
        r.slices.to_csv(out / f"slices_{safe}.csv", index=False)
        r.bets.to_csv(out / f"bets_{safe}.csv", index=False)
    return out
