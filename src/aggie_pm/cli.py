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

from .backtest import default_model_factory, walk_forward_backtest
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
        )

        print(format_report(result, data_profile))

        if args.out is not None:
            out = save_report(result, args.out, dataset=df, dataset_profile=data_profile)
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

    return 1


if __name__ == "__main__":
    sys.exit(main())
