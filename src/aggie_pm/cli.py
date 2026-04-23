"""Command-line entry point.

Usage::

    python -m aggie_pm run                  # synthetic data, default knobs
    python -m aggie_pm run --csv path.csv   # your own data in the same schema
    python -m aggie_pm run --out reports/   # save artifacts under reports/

The defaults are chosen so that ``python -m aggie_pm run`` from the pod
directory just works and prints a full report to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .backtest import default_model_factory, sweep_trading_rules, walk_forward_backtest
from .data import generate_synthetic_markets, load_markets_csv
from .diagnostics import format_dataset_profile, profile_market_dataset, save_dataset_profile
from .report import format_report, save_report


def _add_kalshi_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--kalshi-source", choices=("historical", "live"), default="historical",
                   help="Kalshi endpoint tier to query")
    p.add_argument("--kalshi-series", default=None, help="optional Kalshi series_ticker filter")
    p.add_argument("--kalshi-event", default=None, help="optional Kalshi event_ticker filter")
    p.add_argument("--kalshi-tickers", default=None, help="optional comma-separated market tickers")
    p.add_argument("--kalshi-pages", type=int, default=50, help="max Kalshi pages to fetch")
    p.add_argument("--kalshi-page-limit", type=int, default=1000, help="Kalshi page size")
    p.add_argument("--kalshi-refresh", action="store_true", help="ignore cached Kalshi JSON")


def _load_kalshi_from_args(args: argparse.Namespace):
    from .kalshi import load_kalshi_resolved

    return load_kalshi_resolved(
        source=args.kalshi_source,
        series_ticker=args.kalshi_series,
        event_ticker=args.kalshi_event,
        tickers=args.kalshi_tickers,
        max_pages=args.kalshi_pages,
        page_limit=args.kalshi_page_limit,
        refresh=args.kalshi_refresh,
    )


def _parse_float_grid(text: str, *, allow_none: bool = False) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if allow_none and item.lower() in {"none", "all", "unbounded"}:
            values.append(None)
        else:
            values.append(float(item))
    if not values:
        raise ValueError("grid must contain at least one value")
    return tuple(values)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aggie_pm", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the full pipeline and print the report")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, default=None, help="optional real-data CSV")
    source.add_argument("--kalshi", action="store_true", help="fetch resolved Kalshi markets")
    run.add_argument("--out", type=Path, default=None, help="directory to save artifacts")
    run.add_argument("--save-dataset", type=Path, default=None,
                     help="write the normalized dataset used for the run to CSV")
    run.add_argument("--n-events", type=int, default=2000, help="synthetic dataset size")
    run.add_argument("--seed", type=int, default=20260421, help="synthetic dataset seed")
    _add_kalshi_args(run)
    run.add_argument("--folds", type=int, default=6, help="walk-forward folds")
    run.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
    run.add_argument("--min-edge", type=float, default=0.02, help="min model-vs-market edge to bet")
    run.add_argument("--max-position", type=float, default=0.05, help="max bankroll fraction per bet")
    run.add_argument("--fee-bps", type=float, default=20.0, help="round-trip fee in basis points")
    run.add_argument("--max-trade-spread", type=float, default=None,
                     help="optional tradability gate: do not trade rows wider than this spread")
    run.add_argument("--min-trade-liquidity", type=float, default=None,
                     help="optional tradability gate on feat_liquidity")
    run.add_argument("--min-trade-volume", type=float, default=None,
                     help="optional tradability gate on raw_volume")
    run.add_argument("--min-trade-open-interest", type=float, default=None,
                     help="optional tradability gate on raw_open_interest")
    run.add_argument("--sweep", action="store_true",
                     help="rescore out-of-fold predictions across trading-rule settings")
    run.add_argument("--sweep-kelly", default="0.05,0.10,0.25",
                     help="comma-separated Kelly fractions for --sweep")
    run.add_argument("--sweep-min-edge", default="0.02,0.05,0.10",
                     help="comma-separated edge thresholds for --sweep")
    run.add_argument("--sweep-max-position", default="0.01,0.02,0.05",
                     help="comma-separated max-position caps for --sweep")
    run.add_argument("--sweep-max-trade-spread", default="none",
                     help="comma-separated spread gates for --sweep; use 'none' for no gate")

    extract = sub.add_parser(
        "extract-kalshi",
        help="fetch resolved Kalshi markets, normalize them, and write a reusable CSV",
    )
    _add_kalshi_args(extract)
    extract.add_argument("--out", type=Path, default=Path("data/kalshi_resolved.csv"),
                         help="CSV path for the normalized dataset")
    extract.add_argument("--profile-out", type=Path, default=None,
                         help="optional directory for dataset profile artifacts")

    profile = sub.add_parser("profile", help="profile a normalized market CSV without fitting models")
    profile.add_argument("--csv", type=Path, required=True, help="normalized market CSV to profile")
    profile.add_argument("--out", type=Path, default=None, help="directory to save profile artifacts")

    case = sub.add_parser(
        "case-study",
        help="build a leak-safe multi-year Kalshi macro snapshot case study",
    )
    case.add_argument("--series", default="KXCPI,KXFED",
                      help="comma-separated recurring Kalshi series tickers")
    case.add_argument("--horizons-days", default="30,14,7,3,1",
                      help="comma-separated days-before-close snapshot horizons")
    case.add_argument("--kalshi-pages", type=int, default=100,
                      help="max historical market pages per series")
    case.add_argument("--kalshi-page-limit", type=int, default=1000,
                      help="Kalshi historical market page size")
    case.add_argument("--period-interval", type=int, default=1440,
                      choices=(1, 60, 1440), help="historical candle interval in minutes")
    case.add_argument("--cache-dir", type=Path, default=Path("data/kalshi_cache/case_study"),
                      help="cache directory for market/candle API responses")
    case.add_argument("--out-data", type=Path, default=Path("reports/kalshi_macro_snapshots.csv"),
                      help="output CSV for point-in-time snapshots")
    case.add_argument("--out", type=Path, default=Path("reports/kalshi_macro_case_study"),
                      help="directory to save case-study backtest artifacts")
    case.add_argument("--kalshi-refresh", action="store_true",
                      help="ignore cached Kalshi JSON and re-fetch")
    case.add_argument("--workers", type=int, default=1, help="parallel candle fetch workers")
    case.add_argument("--folds", type=int, default=6, help="walk-forward folds")

    ladder = sub.add_parser(
        "ladder-study",
        help="scan threshold ladders for monotonic relative-value/arbitrage opportunities",
    )
    ladder.add_argument("--csv", type=Path, default=Path("reports/kalshi_macro_snapshots.csv"),
                        help="point-in-time snapshot CSV from case-study")
    ladder.add_argument("--out", type=Path, default=Path("reports/kalshi_ladder_study"),
                        help="directory to save ladder-study artifacts")
    ladder.add_argument("--fee-bps", type=float, default=20.0,
                        help="round-trip fee estimate in basis points")
    ladder.add_argument("--min-edge", type=float, default=0.0,
                        help="minimum after-fee guaranteed edge for pair trades")
    ladder.add_argument("--stake", type=float, default=0.01,
                        help="bankroll fraction allocated per selected pair trade")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "run":
        if args.csv is not None:
            df = load_markets_csv(args.csv)
            print(f"loaded {len(df)} events from {args.csv}")
        elif args.kalshi:
            df = _load_kalshi_from_args(args)
            if df.empty:
                print("Kalshi query returned no resolved binary markets after filtering.")
                return 2
            print(
                f"loaded {len(df)} resolved Kalshi markets "
                f"from {args.kalshi_source} source"
            )
        else:
            df = generate_synthetic_markets(n_events=args.n_events, seed=args.seed)
            print(f"generated {len(df)} synthetic events (seed={args.seed})")

        data_profile = profile_market_dataset(df)
        print(format_dataset_profile(data_profile))

        if args.save_dataset is not None:
            args.save_dataset.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save_dataset, index=False)
            print(f"normalized dataset written to {args.save_dataset.resolve()}")

        result = walk_forward_backtest(
            df,
            model_factory=default_model_factory,
            n_folds=args.folds,
            kelly_fraction=args.kelly,
            min_edge=args.min_edge,
            max_position=args.max_position,
            fee_bps=args.fee_bps,
            max_trade_spread=args.max_trade_spread,
            min_trade_liquidity=args.min_trade_liquidity,
            min_trade_volume=args.min_trade_volume,
            min_trade_open_interest=args.min_trade_open_interest,
        )

        print(format_report(result, data_profile))
        trading_sweep = None
        if args.sweep:
            trading_sweep = sweep_trading_rules(
                result,
                kelly_fractions=_parse_float_grid(args.sweep_kelly),
                min_edges=_parse_float_grid(args.sweep_min_edge),
                max_positions=_parse_float_grid(args.sweep_max_position),
                max_trade_spreads=_parse_float_grid(args.sweep_max_trade_spread, allow_none=True),
                fee_bps=args.fee_bps,
            )
            display_cols = [
                "model",
                "kelly_fraction",
                "min_edge",
                "max_position",
                "max_trade_spread",
                "n_bets",
                "gross_pnl",
                "final_bankroll",
                "sharpe",
                "max_drawdown",
            ]
            print("\nTrading-rule sweep (top 10 by Sharpe):")
            print(
                trading_sweep[[c for c in display_cols if c in trading_sweep.columns]]
                .head(10)
                .to_string(index=False, float_format=lambda v: f"{v:.4f}")
            )

        if args.out is not None:
            out = save_report(
                result,
                args.out,
                dataset=df,
                dataset_profile=data_profile,
                trading_sweep=trading_sweep,
            )
            print(f"\nartifacts written to {out.resolve()}")
        return 0

    if args.cmd == "extract-kalshi":
        df = _load_kalshi_from_args(args)
        if df.empty:
            print("Kalshi query returned no resolved binary markets after filtering.")
            return 2
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df)} normalized Kalshi markets to {args.out.resolve()}")
        data_profile = profile_market_dataset(df)
        print(format_dataset_profile(data_profile))
        profile_out = args.profile_out or args.out.parent / f"{args.out.stem}_profile"
        save_dataset_profile(data_profile, profile_out)
        print(f"profile artifacts written to {profile_out.resolve()}")
        return 0

    if args.cmd == "profile":
        df = load_markets_csv(args.csv)
        data_profile = profile_market_dataset(df)
        print(format_dataset_profile(data_profile))
        if args.out is not None:
            out = save_dataset_profile(data_profile, args.out)
            print(f"profile artifacts written to {out.resolve()}")
        return 0

    if args.cmd == "case-study":
        from .case_study import run_kalshi_macro_case_study

        series = tuple(item.strip() for item in args.series.split(",") if item.strip())
        horizons = tuple(int(float(item.strip())) for item in args.horizons_days.split(",") if item.strip())
        result = run_kalshi_macro_case_study(
            series=series,
            horizon_days=horizons,
            market_pages=args.kalshi_pages,
            page_limit=args.kalshi_page_limit,
            period_interval=args.period_interval,
            cache_dir=args.cache_dir,
            out_data=args.out_data,
            report_dir=args.out,
            refresh=args.kalshi_refresh,
            workers=args.workers,
            n_folds=args.folds,
        )
        print(
            f"built {len(result.snapshots)} point-in-time snapshots "
            f"from {len(result.markets)} settled markets"
        )
        print(f"snapshot dataset written to {args.out_data.resolve()}")
        print(f"case-study report written to {result.case_study_path.resolve()}")
        print(f"artifacts written to {result.report_dir.resolve()}")
        print("\nLeaderboard:")
        print(result.backtest.summary_table().to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\nTrading-rule sweep (top 10 by Sharpe):")
        display_cols = [
            "model",
            "kelly_fraction",
            "min_edge",
            "max_position",
            "max_trade_spread",
            "n_bets",
            "gross_pnl",
            "sharpe",
            "max_drawdown",
        ]
        print(
            result.trading_sweep[[c for c in display_cols if c in result.trading_sweep.columns]]
            .head(10)
            .to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )
        return 0

    if args.cmd == "ladder-study":
        from .relative_value import run_ladder_study

        df = load_markets_csv(args.csv)
        result = run_ladder_study(
            df,
            out_dir=args.out,
            fee_bps=args.fee_bps,
            min_edge=args.min_edge,
            stake_fraction=args.stake,
        )
        print(f"ladder-study report written to {result.report_path.resolve()}")
        print(result.summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        display_cols = [
            "event_ticker",
            "horizon_days",
            "lower_threshold",
            "higher_threshold",
            "package_cost",
            "edge_before_fee",
            "edge_after_fee",
            "pnl_per_unit",
        ]
        print("\nTop pair opportunities:")
        print(
            result.pair_opportunities[[c for c in display_cols if c in result.pair_opportunities.columns]]
            .head(10)
            .to_string(index=False, float_format=lambda v: f"{v:.4f}")
            if not result.pair_opportunities.empty
            else "none"
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
