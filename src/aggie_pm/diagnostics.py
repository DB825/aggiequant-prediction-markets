"""Dataset diagnostics for large prediction-market extracts.

The modeling code needs a narrow schema, but real exchange pulls need a
broader audit before they are trusted: coverage by category, price/spread
regimes, duplicate IDs, missing fields, and the market-prior baseline. This
module keeps those checks flat and CSV-friendly so a large Kalshi pull can be
inspected before any model is fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CORE_COLUMNS = (
    "event_id",
    "category",
    "question",
    "market_prob",
    "market_spread",
    "open_ts",
    "resolve_ts",
    "resolved",
)


def _brier(p: pd.Series, y: pd.Series) -> float:
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p.to_numpy(dtype=float) - y.to_numpy(dtype=float)) ** 2))


def _log_loss(p: pd.Series, y: pd.Series) -> float:
    if len(p) == 0:
        return float("nan")
    arr_p = np.clip(p.to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    arr_y = y.to_numpy(dtype=float)
    return float(-np.mean(arr_y * np.log(arr_p) + (1 - arr_y) * np.log(1 - arr_p)))


def _safe_auc(p: pd.Series, y: pd.Series) -> float:
    if y.nunique(dropna=True) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y.to_numpy(dtype=int), p.to_numpy(dtype=float)))
    except Exception:
        return float("nan")


def _safe_average_precision(p: pd.Series, y: pd.Series) -> float:
    if y.nunique(dropna=True) < 2:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y.to_numpy(dtype=int), p.to_numpy(dtype=float)))
    except Exception:
        return float("nan")


def _slice_metrics(df: pd.DataFrame, labels: pd.Series, slice_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = df.copy()
    work["_slice_value"] = labels.astype("string").fillna("missing")
    for value, g in work.groupby("_slice_value", dropna=False, sort=True):
        if g.empty:
            continue
        rows.append(
            {
                "slice": slice_name,
                "value": str(value),
                "n_events": int(len(g)),
                "yes_rate": float(g["resolved"].mean()),
                "avg_market_prob": float(g["market_prob"].mean()),
                "market_bias": float(g["market_prob"].mean() - g["resolved"].mean()),
                "market_brier": _brier(g["market_prob"], g["resolved"]),
                "market_log_loss": _log_loss(g["market_prob"], g["resolved"]),
                "avg_spread": float(g["market_spread"].mean()),
                "median_spread": float(g["market_spread"].median()),
                "avg_liquidity": (
                    float(g["feat_liquidity"].mean()) if "feat_liquidity" in g.columns else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def profile_market_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return flat diagnostic tables for a normalized market DataFrame.

    The input should follow the package's core schema. Optional columns such as
    ``event_ticker``, ``close_time``, and ``feat_liquidity`` are picked up when
    present. All returned objects are DataFrames so they can be written to CSV
    without custom serialization.
    """
    if df.empty:
        summary = pd.DataFrame(
            [
                {
                    "n_events": 0,
                    "n_unique_events": 0,
                    "n_categories": 0,
                    "yes_rate": float("nan"),
                    "avg_market_prob": float("nan"),
                    "market_brier": float("nan"),
                    "market_log_loss": float("nan"),
                    "market_auc": float("nan"),
                    "market_average_precision": float("nan"),
                    "avg_spread": float("nan"),
                    "median_spread": float("nan"),
                    "p10_spread": float("nan"),
                    "p90_spread": float("nan"),
                    "avg_liquidity": float("nan"),
                    "duplicate_event_ids": 0,
                    "min_open_ts": float("nan"),
                    "max_resolve_ts": float("nan"),
                    "avg_time_to_resolution": float("nan"),
                }
            ]
        )
        return {
            "dataset_summary": summary,
            "missingness": pd.DataFrame(columns=["column", "missing", "missing_rate"]),
            "slices": pd.DataFrame(),
        }

    missing_cols = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns for profiling: {missing_cols}")

    work = df.copy()
    work["market_prob"] = pd.to_numeric(work["market_prob"], errors="coerce")
    work["market_spread"] = pd.to_numeric(work["market_spread"], errors="coerce")
    work["resolved"] = pd.to_numeric(work["resolved"], errors="coerce")
    work["open_ts"] = pd.to_numeric(work["open_ts"], errors="coerce")
    work["resolve_ts"] = pd.to_numeric(work["resolve_ts"], errors="coerce")
    if "feat_liquidity" in work.columns:
        work["feat_liquidity"] = pd.to_numeric(work["feat_liquidity"], errors="coerce")

    valid_scoring = work.dropna(subset=["market_prob", "resolved"]).copy()
    summary_row = {
        "n_events": int(len(work)),
        "n_unique_events": int(work["event_id"].nunique(dropna=True)),
        "n_unique_event_tickers": (
            int(work["event_ticker"].nunique(dropna=True)) if "event_ticker" in work.columns else 0
        ),
        "n_categories": int(work["category"].nunique(dropna=True)),
        "yes_rate": float(valid_scoring["resolved"].mean()) if len(valid_scoring) else float("nan"),
        "avg_market_prob": float(valid_scoring["market_prob"].mean()) if len(valid_scoring) else float("nan"),
        "market_brier": _brier(valid_scoring["market_prob"], valid_scoring["resolved"]),
        "market_log_loss": _log_loss(valid_scoring["market_prob"], valid_scoring["resolved"]),
        "market_auc": _safe_auc(valid_scoring["market_prob"], valid_scoring["resolved"]),
        "market_average_precision": _safe_average_precision(
            valid_scoring["market_prob"], valid_scoring["resolved"]
        ),
        "avg_spread": float(work["market_spread"].mean()),
        "median_spread": float(work["market_spread"].median()),
        "p10_spread": float(work["market_spread"].quantile(0.10)),
        "p90_spread": float(work["market_spread"].quantile(0.90)),
        "avg_liquidity": (
            float(work["feat_liquidity"].mean()) if "feat_liquidity" in work.columns else float("nan")
        ),
        "duplicate_event_ids": int(work["event_id"].duplicated().sum()),
        "min_open_ts": float(work["open_ts"].min()),
        "max_resolve_ts": float(work["resolve_ts"].max()),
        "avg_time_to_resolution": float((work["resolve_ts"] - work["open_ts"]).mean()),
    }
    if "close_time" in work.columns and work["close_time"].notna().any():
        close = pd.to_datetime(work["close_time"], errors="coerce")
        summary_row["first_close_time"] = str(close.min())
        summary_row["last_close_time"] = str(close.max())

    missingness = (
        work.isna()
        .sum()
        .rename("missing")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missingness["missing_rate"] = missingness["missing"] / max(len(work), 1)
    missingness = missingness.sort_values(["missing", "column"], ascending=[False, True]).reset_index(drop=True)

    slice_frames = [
        _slice_metrics(work, work["category"].astype("string"), "category"),
        _slice_metrics(
            work,
            pd.cut(
                work["market_prob"],
                bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                include_lowest=True,
            ),
            "market_prob_bucket",
        ),
        _slice_metrics(
            work,
            pd.cut(
                work["market_spread"],
                bins=[0.0, 0.01, 0.03, 0.05, 0.10, np.inf],
                include_lowest=True,
            ),
            "spread_bucket",
        ),
        _slice_metrics(
            work,
            pd.cut(
                work["resolve_ts"] - work["open_ts"],
                bins=[-np.inf, 1, 3, 7, 14, 30, np.inf],
                include_lowest=True,
            ),
            "time_to_resolution_bucket",
        ),
    ]
    if "feat_liquidity" in work.columns and work["feat_liquidity"].nunique(dropna=True) > 1:
        try:
            liquidity_labels = pd.qcut(work["feat_liquidity"], q=4, duplicates="drop")
            slice_frames.append(_slice_metrics(work, liquidity_labels, "liquidity_quartile"))
        except ValueError:
            pass

    return {
        "dataset_summary": pd.DataFrame([summary_row]),
        "missingness": missingness,
        "slices": pd.concat(slice_frames, ignore_index=True),
    }


def format_dataset_profile(profile: dict[str, pd.DataFrame]) -> str:
    """Format the high-signal part of a dataset profile for stdout."""
    summary = profile["dataset_summary"].iloc[0]
    lines = [
        "Dataset profile:",
        (
            f"  events={int(summary['n_events']):,}  "
            f"categories={int(summary['n_categories']):,}  "
            f"duplicates={int(summary['duplicate_event_ids']):,}"
        ),
        (
            f"  yes_rate={summary['yes_rate']:.3f}  "
            f"market_prob={summary['avg_market_prob']:.3f}  "
            f"market_log_loss={summary['market_log_loss']:.4f}  "
            f"market_auc={summary['market_auc']:.3f}"
        ),
        (
            f"  spread_avg={summary['avg_spread']:.4f}  "
            f"spread_p10={summary['p10_spread']:.4f}  "
            f"spread_p90={summary['p90_spread']:.4f}"
        ),
    ]
    if "first_close_time" in summary.index:
        lines.append(f"  close_time_range={summary['first_close_time']} -> {summary['last_close_time']}")

    slices = profile.get("slices", pd.DataFrame())
    if not slices.empty:
        top = (
            slices[slices["slice"] == "category"]
            .sort_values("n_events", ascending=False)
            .head(6)
        )
        if not top.empty:
            cat_bits = [f"{row['value']}:{int(row['n_events'])}" for _, row in top.iterrows()]
            lines.append("  largest_categories=" + ", ".join(cat_bits))
    return "\n".join(lines)


def save_dataset_profile(profile: dict[str, pd.DataFrame], out_dir: str | Path) -> Path:
    """Write dataset diagnostics to ``out_dir`` as flat CSV artifacts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in profile.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "dataset_profile.txt").write_text(format_dataset_profile(profile), encoding="utf-8")
    return out
