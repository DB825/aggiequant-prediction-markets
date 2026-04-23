"""Relative-value tools for Kalshi threshold ladders.

Recurring macro contracts such as CPI and Fed decisions often list a ladder of
YES contracts:

    P(value > 0.1%), P(value > 0.2%), ...

Those probabilities must be non-increasing as the threshold rises. This module
turns that mathematical constraint into a research workflow: parse strikes,
repair noisy curves with isotonic regression, detect executable dominance
arbitrage, and backtest any trades after fees.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


THRESHOLD_TEXT_RE = re.compile(r"\bAbove\s+(-?\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
THRESHOLD_TICKER_RE = re.compile(r"-T(-?\d+(?:\.\d+)?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LadderStudyResult:
    """Artifacts from a threshold-ladder relative-value run."""

    repaired_ladders: pd.DataFrame
    pair_opportunities: pd.DataFrame
    trades: pd.DataFrame
    summary: pd.DataFrame
    report_path: Path


def extract_threshold(question: object, market_id: object = "") -> float:
    """Extract a numeric threshold from a Kalshi question or market ticker."""
    question_match = THRESHOLD_TEXT_RE.search(str(question))
    if question_match:
        return float(question_match.group(1))
    ticker_match = THRESHOLD_TICKER_RE.search(str(market_id))
    if ticker_match:
        return float(ticker_match.group(1))
    return float("nan")


def _clip_prob(value: float) -> float:
    return float(np.clip(value, 1e-4, 1 - 1e-4))


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _markdown_table(df: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(format(value, floatfmt) if math.isfinite(value) else "nan")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _max_drawdown(path: np.ndarray) -> float:
    if len(path) == 0:
        return 0.0
    peak = np.maximum.accumulate(path)
    return float(((path - peak) / peak).min())


def _sharpe(returns: np.ndarray, n_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return float("nan")
    sd = float(np.std(returns, ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float(np.mean(returns) / sd * math.sqrt(n_per_year))


def _sortino(returns: np.ndarray, n_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return float("nan")
    downside = returns[returns < 0]
    if len(downside) < 2:
        return float("nan")
    sd = float(np.std(downside, ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float(np.mean(returns) / sd * math.sqrt(n_per_year))


def prepare_ladder_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add threshold and executable bid/ask columns to a snapshot dataset."""
    out = df.copy()
    id_col = "market_id" if "market_id" in out.columns else "event_id"
    out["threshold"] = [
        extract_threshold(question, market_id)
        for question, market_id in zip(out["question"], out[id_col])
    ]
    out = out.dropna(subset=["threshold"]).copy()
    if "raw_yes_bid" not in out.columns:
        out["raw_yes_bid"] = out["market_prob"] - out["market_spread"] / 2.0
    if "raw_yes_ask" not in out.columns:
        out["raw_yes_ask"] = out["market_prob"] + out["market_spread"] / 2.0
    out["raw_yes_bid"] = out["raw_yes_bid"].map(_clip_prob)
    out["raw_yes_ask"] = out["raw_yes_ask"].map(_clip_prob)
    out["market_prob"] = out["market_prob"].map(_clip_prob)
    out["market_spread"] = np.maximum(out["raw_yes_ask"] - out["raw_yes_bid"], 1e-4)
    return out.sort_values(["event_ticker", "horizon_days", "threshold"], kind="mergesort").reset_index(drop=True)


def repair_ladder_probabilities(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str] = ("event_ticker", "horizon_days"),
) -> pd.DataFrame:
    """Project each threshold ladder onto a non-increasing probability curve."""
    work = prepare_ladder_frame(df)
    frames: list[pd.DataFrame] = []
    for _, group in work.groupby(list(group_cols), sort=True, dropna=False):
        g = group.sort_values("threshold", kind="mergesort").copy()
        n = len(g)
        adjacent_diffs = g["market_prob"].diff().iloc[1:]
        adjacent_violations = adjacent_diffs[adjacent_diffs > 0]
        if n >= 2:
            weights = 1.0 / np.square(np.maximum(g["market_spread"].to_numpy(dtype=float), 0.005))
            weights = np.clip(weights, 1.0, 10_000.0)
            iso = IsotonicRegression(
                increasing=False,
                y_min=1e-4,
                y_max=1 - 1e-4,
                out_of_bounds="clip",
            )
            repaired = iso.fit_transform(
                g["threshold"].to_numpy(dtype=float),
                g["market_prob"].to_numpy(dtype=float),
                sample_weight=weights,
            )
        else:
            repaired = g["market_prob"].to_numpy(dtype=float)
        g["repaired_prob"] = repaired
        g["repair_gap"] = g["repaired_prob"] - g["market_prob"]
        g["abs_repair_gap"] = g["repair_gap"].abs()
        g["ladder_size"] = n
        g["adjacent_violation_count"] = int(len(adjacent_violations))
        g["max_adjacent_violation"] = (
            float(adjacent_violations.max()) if len(adjacent_violations) else 0.0
        )
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_monotonic_pair_opportunities(
    repaired: pd.DataFrame,
    *,
    fee_bps: float = 20.0,
) -> pd.DataFrame:
    """Build all nested-threshold dominance spreads.

    For thresholds lo < hi, the event {value > hi} is a subset of
    {value > lo}. Buying YES(lo) and NO(hi) pays at least 1.0, and pays 2.0 if
    the realized value lands between lo and hi. It is an executable arbitrage
    when the package costs less than 1.0 after fees.
    """
    rows: list[dict[str, object]] = []
    if repaired.empty:
        return pd.DataFrame()
    fee = fee_bps / 10_000.0
    for (event_ticker, horizon), group in repaired.groupby(["event_ticker", "horizon_days"], sort=True):
        g = group.sort_values("threshold", kind="mergesort").reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(len(g)):
            low = g.iloc[i]
            for j in range(i + 1, len(g)):
                high = g.iloc[j]
                cost = float(low["raw_yes_ask"] + (1.0 - high["raw_yes_bid"]))
                fee_cost = fee * cost
                min_payoff = 1.0
                realized_payoff = float(int(low["resolved"]) + (1 - int(high["resolved"])))
                edge_before_fee = min_payoff - cost
                edge_after_fee = min_payoff - cost - fee_cost
                rows.append(
                    {
                        "event_ticker": event_ticker,
                        "horizon_days": int(horizon),
                        "lower_market_id": low.get("market_id", low["event_id"]),
                        "higher_market_id": high.get("market_id", high["event_id"]),
                        "lower_threshold": float(low["threshold"]),
                        "higher_threshold": float(high["threshold"]),
                        "lower_ask": float(low["raw_yes_ask"]),
                        "higher_bid": float(high["raw_yes_bid"]),
                        "package_cost": cost,
                        "fee_cost": fee_cost,
                        "edge_before_fee": edge_before_fee,
                        "edge_after_fee": edge_after_fee,
                        "realized_payoff": realized_payoff,
                        "pnl_per_unit": realized_payoff - cost - fee_cost,
                        "label_available_ts": int(low.get("label_available_ts", low.get("resolve_ts", 0))),
                        "snapshot_ts": int(low.get("snapshot_ts", low.get("open_ts", 0))),
                        "lower_resolved": int(low["resolved"]),
                        "higher_resolved": int(high["resolved"]),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["edge_after_fee", "edge_before_fee"], ascending=False, kind="mergesort").reset_index(drop=True)


def select_pair_trades(
    opportunities: pd.DataFrame,
    *,
    min_edge: float = 0.0,
) -> pd.DataFrame:
    """Select executable dominance spreads after fees."""
    if opportunities.empty:
        return opportunities.copy()
    return opportunities[opportunities["edge_after_fee"] > min_edge].copy().reset_index(drop=True)


def backtest_pair_trades(
    trades: pd.DataFrame,
    *,
    stake_fraction: float = 0.01,
) -> dict[str, float | int]:
    """Compound fixed-fraction capital across selected dominance spreads."""
    if trades.empty:
        return {
            "n_trades": 0,
            "gross_pnl": 0.0,
            "final_bankroll": 1.0,
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": 0.0,
            "hit_rate": float("nan"),
            "turnover": 0.0,
            "avg_edge_after_fee": float("nan"),
        }
    work = trades.sort_values(["label_available_ts", "snapshot_ts"], kind="mergesort").copy()
    returns: list[float] = []
    for _, row in work.iterrows():
        cost = max(_finite_float(row["package_cost"], 1.0), 1e-6)
        pnl_per_cost = _finite_float(row["pnl_per_unit"], 0.0) / cost
        returns.append(stake_fraction * pnl_per_cost)
    returns_arr = np.asarray(returns, dtype=float)
    bankroll = np.r_[1.0, np.cumprod(1.0 + returns_arr)]
    return {
        "n_trades": int(len(work)),
        "gross_pnl": float(returns_arr.sum()),
        "final_bankroll": float(bankroll[-1]),
        "sharpe": _sharpe(returns_arr),
        "sortino": _sortino(returns_arr),
        "max_drawdown": _max_drawdown(bankroll),
        "hit_rate": float((returns_arr > 0).mean()),
        "turnover": float(stake_fraction * len(work)),
        "avg_edge_after_fee": float(work["edge_after_fee"].mean()),
    }


def summarize_ladder_study(
    repaired: pd.DataFrame,
    opportunities: pd.DataFrame,
    trades: pd.DataFrame,
    trade_summary: dict[str, float | int],
) -> pd.DataFrame:
    """Return one-row headline diagnostics for a ladder study."""
    if repaired.empty:
        return pd.DataFrame([{**trade_summary}])
    panels = repaired[["event_ticker", "horizon_days"]].drop_duplicates()
    adjacent_violation_panels = (
        repaired.groupby(["event_ticker", "horizon_days"])["adjacent_violation_count"].max() > 0
    ).sum()
    executable_before_fee = (
        int((opportunities["edge_before_fee"] > 0).sum()) if not opportunities.empty else 0
    )
    near_crosses = (
        int((opportunities["edge_before_fee"] >= 0).sum()) if not opportunities.empty else 0
    )
    row = {
        "n_ladder_rows": int(len(repaired)),
        "n_ladder_panels": int(len(panels)),
        "n_event_tickers": int(repaired["event_ticker"].nunique()),
        "n_pair_opportunities": int(len(opportunities)),
        "n_panels_with_adjacent_violation": int(adjacent_violation_panels),
        "n_adjacent_violations": int(
            repaired.groupby(["event_ticker", "horizon_days"])["adjacent_violation_count"].max().sum()
        ),
        "max_adjacent_violation": float(repaired["max_adjacent_violation"].max()),
        "avg_abs_repair_gap": float(repaired["abs_repair_gap"].mean()),
        "max_abs_repair_gap": float(repaired["abs_repair_gap"].max()),
        "n_crossed_pairs_before_fee": executable_before_fee,
        "n_near_crossed_pairs_before_fee": near_crosses,
        "max_edge_before_fee": (
            float(opportunities["edge_before_fee"].max()) if not opportunities.empty else float("nan")
        ),
        "max_edge_after_fee": (
            float(opportunities["edge_after_fee"].max()) if not opportunities.empty else float("nan")
        ),
        **trade_summary,
    }
    return pd.DataFrame([row])


def write_ladder_study_markdown(
    *,
    path: str | Path,
    summary: pd.DataFrame,
    opportunities: pd.DataFrame,
    trades: pd.DataFrame,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    top_cols = [
        "event_ticker",
        "horizon_days",
        "lower_threshold",
        "higher_threshold",
        "lower_ask",
        "higher_bid",
        "package_cost",
        "edge_before_fee",
        "edge_after_fee",
        "pnl_per_unit",
    ]
    lines = [
        "# Kalshi Threshold-Ladder Relative-Value Study",
        "",
        "## Thesis",
        "",
        (
            "Recurring CPI/Fed threshold markets imply a probability ladder. "
            "For thresholds `a < b`, `P(X > a)` must be at least `P(X > b)`. "
            "The strategy searches for executable violations by buying YES on "
            "the lower threshold and NO on the higher threshold."
        ),
        "",
        "## Headline Diagnostics",
        "",
        _markdown_table(summary),
        "",
        "## Best Pair Opportunities",
        "",
        _markdown_table(opportunities[[c for c in top_cols if c in opportunities.columns]].head(20))
        if not opportunities.empty
        else "No pair opportunities were generated.",
        "",
        "## Executed Trades",
        "",
        _markdown_table(trades[[c for c in top_cols if c in trades.columns]].head(20))
        if not trades.empty
        else "No pair trades cleared the after-fee edge threshold.",
        "",
        "## Interpretation",
        "",
        (
            "A zero-trade result is still informative: it means the sampled "
            "CPI/Fed ladders were internally consistent enough that hard "
            "dominance arbitrage did not survive bid/ask and fee modeling. "
            "The next layer of alpha should compare the repaired ladder to an "
            "external macro fair value model rather than relying on pure "
            "exchange-internal no-arbitrage violations."
        ),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_ladder_study(
    df: pd.DataFrame,
    *,
    out_dir: str | Path,
    fee_bps: float = 20.0,
    min_edge: float = 0.0,
    stake_fraction: float = 0.01,
) -> LadderStudyResult:
    """Run threshold-ladder repair, arbitrage scan, and backtest."""
    repaired = repair_ladder_probabilities(df)
    opportunities = build_monotonic_pair_opportunities(repaired, fee_bps=fee_bps)
    trades = select_pair_trades(opportunities, min_edge=min_edge)
    trade_summary = backtest_pair_trades(trades, stake_fraction=stake_fraction)
    summary = summarize_ladder_study(repaired, opportunities, trades, trade_summary)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(out / "ladder_repair.csv", index=False)
    opportunities.to_csv(out / "pair_opportunities.csv", index=False)
    trades.to_csv(out / "pair_trades.csv", index=False)
    summary.to_csv(out / "ladder_summary.csv", index=False)
    report_path = write_ladder_study_markdown(
        path=out / "ladder_study.md",
        summary=summary,
        opportunities=opportunities,
        trades=trades,
    )
    return LadderStudyResult(
        repaired_ladders=repaired,
        pair_opportunities=opportunities,
        trades=trades,
        summary=summary,
        report_path=report_path,
    )

