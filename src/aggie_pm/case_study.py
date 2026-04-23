"""Kalshi macro case-study builder.

The normal Kalshi market endpoint is useful for labels and metadata, but it
is not a leak-safe trading snapshot: near settlement, bid/ask quotes can be
effectively terminal. This module builds point-in-time rows from historical
candlesticks, then uses the settled market row only as the final label.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import BacktestResult, sweep_trading_rules, walk_forward_backtest
from .diagnostics import profile_market_dataset
from .kalshi import KalshiClient, load_kalshi_resolved
from .models import (
    BaseRateModel,
    IsotonicCalibratedModel,
    LogisticModel,
    MarketPriorModel,
    MicrostructureGBMModel,
    MicrostructureResidualModel,
)
from .report import save_report

DEFAULT_MACRO_SERIES = ("KXCPI", "KXFED")
DEFAULT_HORIZON_DAYS = (30, 14, 7, 3, 1)


@dataclass(frozen=True)
class CaseStudyResult:
    """Artifacts from a Kalshi case-study run."""

    markets: pd.DataFrame
    snapshots: pd.DataFrame
    backtest: BacktestResult
    trading_sweep: pd.DataFrame
    report_dir: Path
    case_study_path: Path


def _parse_float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _nested_float(row: dict, key: str, subkey: str) -> float:
    value = row.get(key) or {}
    if not isinstance(value, dict):
        return float("nan")
    return _parse_float(value.get(subkey))


def _timestamp_seconds(value: object) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())


def _clip_prob(value: float) -> float:
    return float(np.clip(value, 1e-4, 1 - 1e-4))


def _first_finite(*values: float) -> float:
    for value in values:
        if math.isfinite(value):
            return float(value)
    return float("nan")


def _nonnegative_or_zero(value: float) -> float:
    return float(value) if math.isfinite(value) and value > 0 else 0.0


def _markdown_table(df: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    """Tiny Markdown table formatter without optional tabulate dependency."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(format(value, floatfmt) if math.isfinite(value) else "nan")
            else:
                vals.append(str(value))
        rows.append(vals)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(vals) + " |" for vals in rows)
    return "\n".join(lines)


def candlesticks_to_frame(ticker: str, candles: Iterable[dict]) -> pd.DataFrame:
    """Normalize Kalshi historical candlesticks into flat numeric columns."""
    rows: list[dict[str, float | int | str]] = []
    for candle in candles:
        rows.append(
            {
                "ticker": ticker,
                "end_ts": int(candle.get("end_period_ts", 0) or 0),
                "yes_bid_open": _nested_float(candle, "yes_bid", "open"),
                "yes_bid_high": _nested_float(candle, "yes_bid", "high"),
                "yes_bid_low": _nested_float(candle, "yes_bid", "low"),
                "yes_bid_close": _nested_float(candle, "yes_bid", "close"),
                "yes_ask_open": _nested_float(candle, "yes_ask", "open"),
                "yes_ask_high": _nested_float(candle, "yes_ask", "high"),
                "yes_ask_low": _nested_float(candle, "yes_ask", "low"),
                "yes_ask_close": _nested_float(candle, "yes_ask", "close"),
                "price_open": _nested_float(candle, "price", "open"),
                "price_high": _nested_float(candle, "price", "high"),
                "price_low": _nested_float(candle, "price", "low"),
                "price_close": _nested_float(candle, "price", "close"),
                "price_mean": _nested_float(candle, "price", "mean"),
                "price_previous": _nested_float(candle, "price", "previous"),
                "candle_volume": _parse_float(candle.get("volume")),
                "candle_open_interest": _parse_float(candle.get("open_interest")),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("end_ts", kind="mergesort").reset_index(drop=True)


def _snapshot_row(
    *,
    market: pd.Series,
    candle: pd.Series,
    horizon_days: int,
) -> dict[str, object] | None:
    bid = _clip_prob(_parse_float(candle.get("yes_bid_close")))
    ask = _clip_prob(_parse_float(candle.get("yes_ask_close")))
    if not math.isfinite(bid) or not math.isfinite(ask) or ask <= bid:
        return None

    price_close = _parse_float(candle.get("price_close"))
    price_previous = _parse_float(candle.get("price_previous"))
    price_open = _parse_float(candle.get("price_open"))
    price_high = _parse_float(candle.get("price_high"))
    price_low = _parse_float(candle.get("price_low"))
    volume = _nonnegative_or_zero(_parse_float(candle.get("candle_volume")))
    open_interest = _nonnegative_or_zero(_parse_float(candle.get("candle_open_interest")))

    midpoint = _clip_prob((bid + ask) / 2.0)
    traded_price = _clip_prob(_first_finite(price_close, price_previous, midpoint))
    anchor_open = _clip_prob(_first_finite(price_open, price_previous, midpoint))
    hi = _clip_prob(_first_finite(price_high, max(midpoint, anchor_open)))
    lo = _clip_prob(_first_finite(price_low, min(midpoint, anchor_open)))
    spread = float(np.clip(ask - bid, 1e-4, 0.9998))

    snapshot_ts = int(candle["end_ts"])
    label_available_ts = _timestamp_seconds(market["close_time"])
    event_market_count = int(market.get("event_market_count", 1) or 1)

    return {
        "event_id": f"{market['event_id']}__h{horizon_days}d",
        "market_id": str(market["event_id"]),
        "event_ticker": str(market.get("event_ticker", "")),
        "series_ticker": str(market.get("series_ticker", "")),
        "category": str(market.get("category", "macro")),
        "question": str(market.get("question", "")),
        "horizon_days": int(horizon_days),
        "snapshot_ts": snapshot_ts,
        "label_available_ts": label_available_ts,
        "open_ts": snapshot_ts,
        "resolve_ts": label_available_ts,
        "snapshot_time": pd.to_datetime(snapshot_ts, unit="s", utc=True).isoformat(),
        "close_time": pd.Timestamp(market["close_time"]).isoformat(),
        "market_prob": midpoint,
        "market_spread": spread,
        "resolved": int(market["resolved"]),
        "raw_yes_bid": bid,
        "raw_yes_ask": ask,
        "raw_last_price": traded_price,
        "raw_prev_price": _clip_prob(_first_finite(price_previous, anchor_open)),
        "raw_volume": volume,
        "raw_volume_24h": volume,
        "raw_open_interest": open_interest,
        "event_market_count": event_market_count,
        "feat_signal": traded_price - _clip_prob(_first_finite(price_previous, traded_price)),
        "feat_momentum": midpoint - anchor_open,
        "feat_dispersion": max(hi - lo, spread),
        "feat_liquidity": math.log1p(volume + open_interest),
        "feat_trade_activity": math.log1p(volume),
        "feat_open_interest": math.log1p(open_interest),
        "feat_volume_share": float(volume / max(volume + open_interest, 1.0)),
        "feat_event_market_count": math.log1p(event_market_count),
    }


def build_snapshot_dataset(
    markets: pd.DataFrame,
    candle_frames: dict[str, pd.DataFrame],
    *,
    horizon_days: Iterable[int] = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Create leak-safe fixed-horizon snapshots from settled markets/candles."""
    rows: list[dict[str, object]] = []
    for _, market in markets.iterrows():
        ticker = str(market["event_id"])
        candles = candle_frames.get(ticker)
        if candles is None or candles.empty:
            continue
        close_ts = _timestamp_seconds(market["close_time"])
        for horizon in horizon_days:
            asof_ts = close_ts - int(horizon) * 86_400
            eligible = candles[candles["end_ts"] <= asof_ts]
            if eligible.empty:
                continue
            row = _snapshot_row(
                market=market,
                candle=eligible.iloc[-1],
                horizon_days=int(horizon),
            )
            if row is not None:
                rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["label_available_ts", "snapshot_ts"], kind="mergesort").reset_index(drop=True)


def _fetch_one_candle_frame(
    *,
    market: pd.Series,
    cache_dir: Path,
    period_interval: int,
    refresh: bool,
) -> tuple[str, pd.DataFrame]:
    ticker = str(market["event_id"])
    start_ts = _timestamp_seconds(market["open_time"])
    end_ts = _timestamp_seconds(market["close_time"])
    cache_path = cache_dir / f"{ticker}_{period_interval}_{start_ts}_{end_ts}.json"

    if cache_path.exists() and not refresh:
        candles = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        client = KalshiClient()
        candles = client.fetch_historical_market_candlesticks(
            ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=period_interval,
        )
        cache_path.write_text(json.dumps(candles), encoding="utf-8")

    return ticker, candlesticks_to_frame(ticker, candles)


def fetch_candle_frames(
    markets: pd.DataFrame,
    *,
    cache_dir: str | Path,
    period_interval: int = 1440,
    refresh: bool = False,
    workers: int = 1,
) -> dict[str, pd.DataFrame]:
    """Fetch/cache historical candle frames for each market ticker."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    unique = markets.drop_duplicates("event_id").reset_index(drop=True)
    if workers <= 1:
        for _, market in unique.iterrows():
            ticker, frame = _fetch_one_candle_frame(
                market=market,
                cache_dir=cache,
                period_interval=period_interval,
                refresh=refresh,
            )
            out[ticker] = frame
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _fetch_one_candle_frame,
                market=market,
                cache_dir=cache,
                period_interval=period_interval,
                refresh=refresh,
            )
            for _, market in unique.iterrows()
        ]
        for future in as_completed(futures):
            ticker, frame = future.result()
            out[ticker] = frame
    return out


def focused_case_study_model_factory(frame):
    """Model lineup for small/medium real-data case studies."""
    return [
        MarketPriorModel().fit(frame),
        BaseRateModel().fit(frame),
        LogisticModel(name="logistic_C0.1", C=0.1).fit(frame),
        LogisticModel(name="logistic_C1.0", C=1.0).fit(frame),
        IsotonicCalibratedModel(base=LogisticModel(name="logistic_C0.1", C=0.1)).fit(frame),
        MicrostructureResidualModel().fit(frame),
        MicrostructureGBMModel(max_iter=120, max_leaf_nodes=15).fit(frame),
    ]


def _format_pct(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{100 * value:.1f}%"


def write_case_study_markdown(
    *,
    path: str | Path,
    series: tuple[str, ...],
    markets: pd.DataFrame,
    snapshots: pd.DataFrame,
    result: BacktestResult,
    trading_sweep: pd.DataFrame,
) -> Path:
    """Write a concise research note for resume/interview review."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    leaderboard = result.summary_table()
    display_cols = [
        "model",
        "brier",
        "log_loss",
        "log_loss_edge_vs_market",
        "ece",
        "roc_auc",
        "n_bets",
        "gross_pnl",
        "sharpe",
        "sortino",
        "max_drawdown",
        "hit_rate",
    ]
    top_sweep_cols = [
        "model",
        "kelly_fraction",
        "min_edge",
        "max_position",
        "max_trade_spread",
        "n_bets",
        "gross_pnl",
        "sharpe",
        "sortino",
        "max_drawdown",
        "hit_rate",
    ]
    best = leaderboard.sort_values("log_loss", kind="mergesort").iloc[0]
    best_trade = trading_sweep.iloc[0] if not trading_sweep.empty else None
    start = pd.to_datetime(markets["close_time"]).min()
    end = pd.to_datetime(markets["close_time"]).max()
    lines = [
        "# Kalshi Macro Prediction-Market Case Study",
        "",
        "## Research Question",
        "",
        (
            "Can point-in-time Kalshi macro market microstructure improve resolved-outcome "
            "forecasts without relying on terminal settlement snapshots?"
        ),
        "",
        "## Data",
        "",
        f"- Series: {', '.join(series)}.",
        f"- Settled markets: {len(markets):,}.",
        f"- Point-in-time snapshots: {len(snapshots):,}.",
        f"- Settlement span: {start.date()} to {end.date()}.",
        f"- Snapshot horizons: {', '.join(str(x) for x in sorted(snapshots['horizon_days'].unique(), reverse=True))} days before close.",
        "",
        "Labels are final market resolutions; tradable features come from historical candlesticks only.",
        "The walk-forward split sorts on `label_available_ts`, so a market label enters training only after settlement.",
        "",
        "## Model And Tuning",
        "",
        "- Baselines: market prior and smoothed base rate.",
        "- Linear models: logistic regression with `C=0.1` and `C=1.0`, plus isotonic calibration.",
        "- Microstructure models: market-prior residual logistic and nonlinear residual GBM.",
        "- Trading sweep: Kelly fraction, minimum edge, max position, and spread gate rescored on out-of-fold predictions.",
        "",
        "## Forecast Leaderboard",
        "",
        _markdown_table(leaderboard[[c for c in display_cols if c in leaderboard.columns]]),
        "",
        "## Trading-Rule Sweep",
        "",
        _markdown_table(
            trading_sweep[[c for c in top_sweep_cols if c in trading_sweep.columns]].head(15)
        )
        if not trading_sweep.empty
        else "No sweep rows produced.",
        "",
        "## Takeaway",
        "",
        (
            f"Best forecast model by log loss: `{best['model']}` "
            f"(log loss {best['log_loss']:.4f}, AUC {best['roc_auc']:.3f})."
        ),
    ]
    if best_trade is not None:
        lines.append(
            "Best trading sweep row: "
            f"`{best_trade['model']}` with {int(best_trade['n_bets'])} bets, "
            f"Sharpe {best_trade['sharpe']:.2f}, max drawdown {_format_pct(float(best_trade['max_drawdown']))}, "
            f"hit rate {_format_pct(float(best_trade['hit_rate']))}."
        )
    lines.extend(
        [
            "",
            "Metrics should be interpreted as a research backtest, not production PnL: the dataset is real, "
            "but exchange fees, queue position, market impact, and exact fill mechanics are approximated.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_kalshi_macro_case_study(
    *,
    series: Iterable[str] = DEFAULT_MACRO_SERIES,
    horizon_days: Iterable[int] = DEFAULT_HORIZON_DAYS,
    market_pages: int = 100,
    page_limit: int = 1000,
    period_interval: int = 1440,
    cache_dir: str | Path = "data/kalshi_cache/case_study",
    out_data: str | Path = "reports/kalshi_macro_snapshots.csv",
    report_dir: str | Path = "reports/kalshi_macro_case_study",
    refresh: bool = False,
    workers: int = 1,
    n_folds: int = 6,
) -> CaseStudyResult:
    """Fetch data, build snapshots, backtest, sweep, and write artifacts."""
    series_tuple = tuple(str(item).strip() for item in series if str(item).strip())
    if not series_tuple:
        raise ValueError("at least one series ticker is required")

    market_frames: list[pd.DataFrame] = []
    for ticker in series_tuple:
        frame = load_kalshi_resolved(
            source="historical",
            series_ticker=ticker,
            max_pages=market_pages,
            page_limit=page_limit,
            refresh=refresh,
        )
        if not frame.empty:
            frame["series_ticker"] = ticker
            market_frames.append(frame)
    if not market_frames:
        raise ValueError("Kalshi query returned no resolved markets for the requested series")

    markets = (
        pd.concat(market_frames, ignore_index=True)
        .drop_duplicates("event_id")
        .sort_values("close_time", kind="mergesort")
        .reset_index(drop=True)
    )
    candle_cache = Path(cache_dir) / "candles"
    candle_frames = fetch_candle_frames(
        markets,
        cache_dir=candle_cache,
        period_interval=period_interval,
        refresh=refresh,
        workers=workers,
    )
    snapshots = build_snapshot_dataset(markets, candle_frames, horizon_days=horizon_days)
    if snapshots.empty:
        raise ValueError("no point-in-time snapshots could be built from historical candles")

    out_data_path = Path(out_data)
    out_data_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_csv(out_data_path, index=False)

    result = walk_forward_backtest(
        snapshots,
        focused_case_study_model_factory,
        n_folds=n_folds,
        kelly_fraction=0.05,
        min_edge=0.10,
        max_position=0.01,
        fee_bps=20.0,
        max_trade_spread=0.20,
    )
    sweep = sweep_trading_rules(
        result,
        kelly_fractions=(0.02, 0.05, 0.10),
        min_edges=(0.05, 0.10, 0.15),
        max_positions=(0.005, 0.01, 0.02),
        max_trade_spreads=(0.05, 0.10, 0.20, 0.50, None),
        fee_bps=20.0,
    )
    profile = profile_market_dataset(snapshots)
    report_path = save_report(result, report_dir, dataset=snapshots, dataset_profile=profile, trading_sweep=sweep)
    note_path = write_case_study_markdown(
        path=report_path / "case_study.md",
        series=series_tuple,
        markets=markets,
        snapshots=snapshots,
        result=result,
        trading_sweep=sweep,
    )
    return CaseStudyResult(
        markets=markets,
        snapshots=snapshots,
        backtest=result,
        trading_sweep=sweep,
        report_dir=report_path,
        case_study_path=note_path,
    )
