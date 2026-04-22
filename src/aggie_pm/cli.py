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
from .report import format_report, save_report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aggie_pm", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the full pipeline and print the report")
    run.add_argument("--csv", type=Path, default=None, help="optional real-data CSV")
    run.add_argument("--out", type=Path, default=None, help="directory to save artifacts")
    run.add_argument("--n-events", type=int, default=2000, help="synthetic dataset size")
    run.add_argument("--seed", type=int, default=20260421, help="synthetic dataset seed")
    run.add_argument("--folds", type=int, default=6, help="walk-forward folds")
    run.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
    run.add_argument("--min-edge", type=float, default=0.02, help="min model-vs-market edge to bet")
    run.add_argument("--max-position", type=float, default=0.05, help="max bankroll fraction per bet")
    run.add_argument("--fee-bps", type=float, default=20.0, help="round-trip fee in basis points")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "run":
        if args.csv is not None:
            df = load_markets_csv(args.csv)
            print(f"loaded {len(df)} events from {args.csv}")
        else:
            df = generate_synthetic_markets(n_events=args.n_events, seed=args.seed)
            print(f"generated {len(df)} synthetic events (seed={args.seed})")

        result = walk_forward_backtest(
            df,
            model_factory=default_model_factory,
            n_folds=args.folds,
            kelly_fraction=args.kelly,
            min_edge=args.min_edge,
            max_position=args.max_position,
            fee_bps=args.fee_bps,
        )

        print(format_report(result))

        if args.out is not None:
            out = save_report(result, args.out)
            print(f"\nartifacts written to {out.resolve()}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
