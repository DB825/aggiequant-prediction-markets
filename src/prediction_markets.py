"""Starter scoring tools for a prediction markets pod.

The point of this file is not to trade. It teaches how to evaluate probability
forecasts before anyone talks about execution.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarketForecast:
    event_id: str
    category: str
    question: str
    market_prob: float
    model_prob: float
    resolved: int


def clamp_probability(value: float) -> float:
    return min(max(value, 1e-9), 1 - 1e-9)


def load_forecasts(path: Path) -> list[MarketForecast]:
    rows: list[MarketForecast] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                MarketForecast(
                    event_id=row["event_id"],
                    category=row["category"],
                    question=row["question"],
                    market_prob=float(row["market_prob"]),
                    model_prob=float(row["model_prob"]),
                    resolved=int(row["resolved"]),
                )
            )
    return rows


def brier_score(rows: list[MarketForecast], probability_field: str) -> float:
    errors = []
    for row in rows:
        probability = getattr(row, probability_field)
        errors.append((probability - row.resolved) ** 2)
    return sum(errors) / len(errors)


def log_loss(rows: list[MarketForecast], probability_field: str) -> float:
    losses = []
    for row in rows:
        probability = clamp_probability(getattr(row, probability_field))
        outcome = row.resolved
        losses.append(-(outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability)))
    return sum(losses) / len(losses)


def calibration_bins(
    rows: list[MarketForecast],
    probability_field: str,
    bins: int = 5,
) -> list[tuple[str, int, float, float]]:
    bucketed: list[list[MarketForecast]] = [[] for _ in range(bins)]
    for row in rows:
        probability = clamp_probability(getattr(row, probability_field))
        index = min(int(probability * bins), bins - 1)
        bucketed[index].append(row)

    output: list[tuple[str, int, float, float]] = []
    for index, bucket in enumerate(bucketed):
        low = index / bins
        high = (index + 1) / bins
        label = f"{low:.1f}-{high:.1f}"
        if not bucket:
            output.append((label, 0, float("nan"), float("nan")))
            continue
        avg_probability = sum(getattr(row, probability_field) for row in bucket) / len(bucket)
        realized_rate = sum(row.resolved for row in bucket) / len(bucket)
        output.append((label, len(bucket), avg_probability, realized_rate))
    return output


def edge_table(rows: list[MarketForecast], min_edge: float = 0.05) -> list[tuple[str, str, float, int]]:
    opportunities = []
    for row in rows:
        edge = row.model_prob - row.market_prob
        if abs(edge) >= min_edge:
            side = "YES" if edge > 0 else "NO"
            opportunities.append((row.event_id, side, edge, row.resolved))
    return sorted(opportunities, key=lambda item: abs(item[2]), reverse=True)


def print_summary(rows: list[MarketForecast]) -> None:
    print(f"events: {len(rows)}")
    for field in ("market_prob", "model_prob"):
        print(f"\n{field}")
        print(f"  brier:   {brier_score(rows, field):.4f}")
        print(f"  logloss: {log_loss(rows, field):.4f}")
        print("  calibration:")
        for label, count, avg_probability, realized_rate in calibration_bins(rows, field):
            if count == 0:
                print(f"    {label}: empty")
            else:
                print(
                    f"    {label}: n={count}, avg_p={avg_probability:.2f}, realized={realized_rate:.2f}"
                )

    print("\nmodel-minus-market edges above 5 percentage points")
    for event_id, side, edge, resolved in edge_table(rows):
        print(f"  {event_id}: {side} edge={edge:+.2%}, resolved={resolved}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_markets.csv")
    print_summary(load_forecasts(path))


if __name__ == "__main__":
    main()
