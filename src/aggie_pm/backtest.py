"""Walk-forward backtest with fractional Kelly sizing.

The only honest backtest on a time-ordered market is walk-forward: at each
step, the model sees only data that closed strictly before the test window
opens. Any cross-validation that shuffles the whole dataset leaks future
information through the market price (Bailey, Borwein, Lopez de Prado,
Zhu 2014, "The Probability of Backtest Overfitting").

Bet sizing uses fractional Kelly on a binary market priced at ``q``:

    For a YES bet at ask price q_ask, model probability p:
        edge  = p - q_ask
        kelly = edge / (1 - q_ask)             # binary Kelly fraction
    For a NO bet at bid price q_bid (we buy NO at 1 - q_bid):
        edge  = (1 - q_bid) - (1 - p)
              = p_no_ask - (1 - p)             with p_no_ask = 1 - q_bid
        kelly = ((1 - p) - q_no_ask) / (1 - q_no_ask)

We scale by ``kelly_fraction`` (default 0.25 - quarter Kelly is a standard
practitioner choice; Thorp 2006, "The Kelly Criterion in Blackjack, Sports
Betting, and the Stock Market"). We require a minimum absolute edge to
filter noise, and cap position size as a fraction of bankroll so a single
event can't blow the book up.

Reported metrics:

- Brier score, log loss per model.
- Calibration bins (reliability diagram data).
- PnL per bet, cumulative PnL, bankroll path.
- Sharpe and Sortino on per-event returns (annualised to 252 bets).
- Max drawdown of the bankroll path.
- Hit rate (fraction of bets that won).
- Per-category PnL breakdown.
- AUC, average precision, bet coverage, turnover, profit factor, and
  forecast/trading slices by category, price, spread, liquidity, and tenor.
- Optional tradability gates so forecasting skill is scored on all events
  while trading metrics only use markets with executable spreads/liquidity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import FeatureMatrix, build_features
from .models import Model


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            rows.append(
                {"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": 0,
                 "avg_pred": float("nan"), "emp_rate": float("nan")}
            )
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                "n": int(mask.sum()),
                "avg_pred": float(p[mask].mean()),
                "emp_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """ECE: weighted |avg_pred - empirical_rate| across bins."""
    df = calibration_bins(p, y, n_bins=n_bins)
    df = df.dropna()
    if df.empty:
        return float("nan")
    w = df["n"] / df["n"].sum()
    return float((w * (df["avg_pred"] - df["emp_rate"]).abs()).sum())


def _roc_auc(p: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _average_precision(p: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


# ---------------------------------------------------------------------------
# Kelly sizing and PnL
# ---------------------------------------------------------------------------


def _bet_and_pnl(
    p_model: float,
    market_prob: float,
    spread: float,
    y: int,
    *,
    kelly_fraction: float,
    min_edge: float,
    max_position: float,
    fee_bps: float,
) -> tuple[str, float, float]:
    """Return (side, position_fraction_of_bankroll, pnl_fraction_of_bankroll).

    ``pnl_fraction_of_bankroll`` is the return on the *whole* bankroll for
    this event, not on the staked amount, so bankroll compounds cleanly.
    """

    half = spread / 2.0
    ask_yes = min(max(market_prob + half, 1e-6), 1 - 1e-6)
    bid_yes = min(max(market_prob - half, 1e-6), 1 - 1e-6)
    # YES edge at ask
    edge_yes = p_model - ask_yes
    kelly_yes = edge_yes / max(1 - ask_yes, 1e-6)
    # NO edge: buy NO at (1 - bid_yes)
    ask_no = 1 - bid_yes
    p_no = 1 - p_model
    edge_no = p_no - ask_no
    kelly_no = edge_no / max(1 - ask_no, 1e-6)

    fee = fee_bps / 10_000.0

    if edge_yes > min_edge and edge_yes >= edge_no:
        stake = max(min(kelly_yes * kelly_fraction, max_position), 0.0)
        if stake <= 0:
            return ("none", 0.0, 0.0)
        payoff = (1 - ask_yes) if y == 1 else -ask_yes
        pnl = stake * (payoff - fee)
        return ("YES", stake, pnl)

    if edge_no > min_edge:
        stake = max(min(kelly_no * kelly_fraction, max_position), 0.0)
        if stake <= 0:
            return ("none", 0.0, 0.0)
        payoff = (1 - ask_no) if y == 0 else -ask_no
        pnl = stake * (payoff - fee)
        return ("NO", stake, pnl)

    return ("none", 0.0, 0.0)


def _is_finite_number(x: float | int | None) -> bool:
    if x is None:
        return False
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _series_float(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return float("nan")
    try:
        return float(row[col])
    except (TypeError, ValueError):
        return float("nan")


def _trade_eligibility(
    *,
    spread: float,
    liquidity: float = float("nan"),
    volume: float = float("nan"),
    open_interest: float = float("nan"),
    max_trade_spread: float | None = None,
    min_trade_liquidity: float | None = None,
    min_trade_volume: float | None = None,
    min_trade_open_interest: float | None = None,
) -> tuple[bool, str]:
    """Return whether a row is eligible to trade plus a compact reason code."""
    reasons: list[str] = []
    if max_trade_spread is not None and (not _is_finite_number(spread) or spread > max_trade_spread):
        reasons.append("spread")
    if min_trade_liquidity is not None and (
        not _is_finite_number(liquidity) or liquidity < min_trade_liquidity
    ):
        reasons.append("liquidity")
    if min_trade_volume is not None and (not _is_finite_number(volume) or volume < min_trade_volume):
        reasons.append("volume")
    if min_trade_open_interest is not None and (
        not _is_finite_number(open_interest) or open_interest < min_trade_open_interest
    ):
        reasons.append("open_interest")
    if reasons:
        return False, "|".join(reasons)
    return True, "ok"


def _edge_snapshot(p_model: float, market_prob: float, spread: float) -> tuple[float, float, str, float]:
    """Return YES edge, NO edge, best side, and best executable edge."""
    half = spread / 2.0
    ask_yes = min(max(market_prob + half, 1e-6), 1 - 1e-6)
    bid_yes = min(max(market_prob - half, 1e-6), 1 - 1e-6)
    edge_yes = p_model - ask_yes
    ask_no = 1 - bid_yes
    edge_no = (1 - p_model) - ask_no
    if edge_yes >= edge_no:
        return float(edge_yes), float(edge_no), "YES", float(edge_yes)
    return float(edge_yes), float(edge_no), "NO", float(edge_no)


def _max_drawdown(bankroll_path: np.ndarray) -> float:
    peak = np.maximum.accumulate(bankroll_path)
    dd = (bankroll_path - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def _sharpe(returns: np.ndarray, n_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return float("nan")
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return mu / sd * math.sqrt(n_per_year)


def _sortino(returns: np.ndarray, n_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return float("nan")
    mu = float(np.mean(returns))
    downside = returns[returns < 0]
    if len(downside) < 2:
        return float("nan")
    sd = float(np.std(downside, ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return mu / sd * math.sqrt(n_per_year)


def _profit_factor(pnl: pd.Series) -> float:
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    if losses <= 1e-12:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def _slice_metrics(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Compute forecast and trading diagnostics over useful data slices."""
    if bets_df.empty:
        return pd.DataFrame()

    work = bets_df.copy()
    frames: list[pd.DataFrame] = []

    def add_slice(labels: pd.Series, name: str) -> None:
        local = work.copy()
        local["_slice_value"] = labels.astype("string").fillna("missing")
        rows: list[dict[str, object]] = []
        for value, g in local.groupby("_slice_value", dropna=False, sort=True):
            traded = g[g["side"] != "none"]
            tradeable = int(g["trade_allowed"].sum()) if "trade_allowed" in g.columns else int(len(g))
            rows.append(
                {
                    "slice": name,
                    "value": str(value),
                    "n_events": int(len(g)),
                    "n_tradeable_events": tradeable,
                    "tradeable_event_rate": float(tradeable / len(g)) if len(g) else float("nan"),
                    "n_bets": int(len(traded)),
                    "coverage": float(len(traded) / len(g)) if len(g) else float("nan"),
                    "brier": brier_score(g["model_prob"].to_numpy(), g["resolved"].to_numpy()),
                    "log_loss": log_loss(g["model_prob"].to_numpy(), g["resolved"].to_numpy()),
                    "avg_model_prob": float(g["model_prob"].mean()),
                    "avg_market_prob": float(g["market_prob"].mean()),
                    "avg_best_edge": float(g["best_edge"].mean()),
                    "avg_trade_edge": (
                        float(traded["trade_edge"].mean()) if len(traded) else float("nan")
                    ),
                    "hit_rate": (
                        float((traded["pnl"] > 0).mean()) if len(traded) else float("nan")
                    ),
                    "pnl": float(traded["pnl"].sum()) if len(traded) else 0.0,
                    "turnover": float(traded["stake"].sum()) if len(traded) else 0.0,
                    "profit_factor": _profit_factor(traded["pnl"]) if len(traded) else float("nan"),
                }
            )
        frames.append(pd.DataFrame(rows))

    add_slice(work["category"].astype("string"), "category")
    add_slice(
        pd.cut(work["market_prob"], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True),
        "market_prob_bucket",
    )
    add_slice(
        pd.cut(work["spread"], bins=[0.0, 0.01, 0.03, 0.05, 0.10, np.inf], include_lowest=True),
        "spread_bucket",
    )
    add_slice(
        pd.cut(
            work["time_to_resolution"],
            bins=[-np.inf, 1, 3, 7, 14, 30, np.inf],
            include_lowest=True,
        ),
        "time_to_resolution_bucket",
    )
    if "feat_liquidity" in work.columns and work["feat_liquidity"].nunique(dropna=True) > 1:
        try:
            add_slice(pd.qcut(work["feat_liquidity"], q=4, duplicates="drop"), "liquidity_quartile")
        except ValueError:
            pass
    if "trade_block_reason" in work.columns:
        add_slice(work["trade_block_reason"].astype("string"), "trade_block_reason")
    add_slice(work["side"].astype("string"), "trade_side")

    return pd.concat(frames, ignore_index=True)


def _summarize_trade_frame(traded: pd.DataFrame, *, n_events: int) -> dict[str, float | int]:
    returns = traded["pnl"].to_numpy() if len(traded) else np.array([], dtype=float)
    bankroll = np.r_[1.0, np.cumprod(1.0 + returns)] if len(traded) else np.array([1.0])
    wins = traded.loc[traded["pnl"] > 0, "pnl"] if len(traded) else pd.Series(dtype=float)
    losses = traded.loc[traded["pnl"] < 0, "pnl"] if len(traded) else pd.Series(dtype=float)
    return {
        "n_bets": int(len(traded)),
        "coverage": float(len(traded) / n_events) if n_events else float("nan"),
        "hit_rate": float((traded["pnl"] > 0).mean()) if len(traded) else float("nan"),
        "gross_pnl": float(traded["pnl"].sum()) if len(traded) else 0.0,
        "final_bankroll": float(bankroll[-1]) if len(bankroll) else 1.0,
        "sharpe": _sharpe(returns),
        "sortino": _sortino(returns),
        "max_drawdown": _max_drawdown(bankroll) if len(bankroll) else 0.0,
        "turnover": float(traded["stake"].sum()) if len(traded) else 0.0,
        "avg_stake": float(traded["stake"].mean()) if len(traded) else float("nan"),
        "avg_trade_edge": float(traded["trade_edge"].mean()) if len(traded) else float("nan"),
        "median_trade_edge": float(traded["trade_edge"].median()) if len(traded) else float("nan"),
        "profit_factor": _profit_factor(traded["pnl"]) if len(traded) else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "n_yes_bets": int((traded["side"] == "YES").sum()) if len(traded) else 0,
        "n_no_bets": int((traded["side"] == "NO").sum()) if len(traded) else 0,
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    name: str
    brier: float
    log_loss: float
    ece: float
    roc_auc: float
    average_precision: float
    n_tradeable_events: int
    tradeable_event_rate: float
    n_bets: int
    coverage: float
    hit_rate: float
    gross_pnl: float
    final_bankroll: float
    sharpe: float
    sortino: float
    max_drawdown: float
    turnover: float
    avg_stake: float
    avg_trade_edge: float
    median_trade_edge: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    n_yes_bets: int
    n_no_bets: int
    calibration: pd.DataFrame
    per_category_pnl: pd.DataFrame
    slices: pd.DataFrame
    bets: pd.DataFrame


@dataclass
class BacktestResult:
    n_folds: int
    n_events: int
    results: dict[str, ModelResult] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        rows = []
        market = self.results.get("market_prior")
        for r in self.results.values():
            rows.append(
                {
                    "model": r.name,
                    "brier": r.brier,
                    "log_loss": r.log_loss,
                    "ece": r.ece,
                    "roc_auc": r.roc_auc,
                    "average_precision": r.average_precision,
                    "brier_edge_vs_market": (
                        market.brier - r.brier if market is not None else float("nan")
                    ),
                    "log_loss_edge_vs_market": (
                        market.log_loss - r.log_loss if market is not None else float("nan")
                    ),
                    "n_tradeable_events": r.n_tradeable_events,
                    "tradeable_event_rate": r.tradeable_event_rate,
                    "n_bets": r.n_bets,
                    "coverage": r.coverage,
                    "hit_rate": r.hit_rate,
                    "gross_pnl": r.gross_pnl,
                    "final_bankroll": r.final_bankroll,
                    "sharpe": r.sharpe,
                    "sortino": r.sortino,
                    "max_drawdown": r.max_drawdown,
                    "turnover": r.turnover,
                    "avg_stake": r.avg_stake,
                    "avg_trade_edge": r.avg_trade_edge,
                    "median_trade_edge": r.median_trade_edge,
                    "profit_factor": r.profit_factor,
                    "avg_win": r.avg_win,
                    "avg_loss": r.avg_loss,
                    "n_yes_bets": r.n_yes_bets,
                    "n_no_bets": r.n_no_bets,
                }
            )
        return pd.DataFrame(rows).sort_values("log_loss", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------


def walk_forward_backtest(
    df: pd.DataFrame,
    model_factory,
    *,
    n_folds: int = 6,
    min_train_frac: float = 0.4,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.02,
    max_position: float = 0.05,
    fee_bps: float = 20.0,
    max_trade_spread: float | None = None,
    min_trade_liquidity: float | None = None,
    min_trade_volume: float | None = None,
    min_trade_open_interest: float | None = None,
) -> BacktestResult:
    """Run a walk-forward backtest.

    Parameters
    ----------
    df : DataFrame
        The full market history. Must match the schema produced by
        ``data.generate_synthetic_markets``. If ``label_available_ts`` is
        present, walk-forward folds are ordered by that timestamp so labels
        from point-in-time snapshots only enter training after resolution.
    model_factory : Callable[[FeatureMatrix], list[Model]]
        Called on each training fold to produce a fresh, fitted list of
        models. ``aggie_pm.models.train_model_zoo`` is the canonical choice.
    n_folds : int
        Number of walk-forward test folds. The first ``min_train_frac`` of
        the data is reserved as the initial training window; the remainder
        is split into ``n_folds`` contiguous test windows.
    kelly_fraction, min_edge, max_position, fee_bps :
        Bet-sizing knobs. Defaults reflect common practitioner settings for
        a small retail-scale book.
    max_trade_spread, min_trade_liquidity, min_trade_volume, min_trade_open_interest :
        Optional tradability gates. Forecast metrics still score all events;
        rows failing these gates are marked in the bet log and cannot trade.
    """

    sort_cols = ["label_available_ts", "open_ts"] if "label_available_ts" in df.columns else ["open_ts"]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    n = len(df)
    start = int(n * min_train_frac)
    if start >= n:
        raise ValueError("min_train_frac too large for dataset size")
    fold_edges = np.linspace(start, n, n_folds + 1, dtype=int)

    # Per-model accumulators
    preds: dict[str, list[np.ndarray]] = {}
    ys: list[np.ndarray] = []
    bets_by_model: dict[str, list[dict]] = {}

    for fold_ix in range(n_folds):
        lo, hi = int(fold_edges[fold_ix]), int(fold_edges[fold_ix + 1])
        if hi <= lo:
            continue
        train_df = df.iloc[:lo].copy()
        test_df = df.iloc[lo:hi].copy()

        fm_train, cat_rates = build_features(train_df)
        fm_test, _ = build_features(test_df, category_base_rates=cat_rates)

        models = model_factory(fm_train)

        ys.append(fm_test.y.copy())
        for m in models:
            p = m.predict(fm_test)
            preds.setdefault(m.name, []).append(p)
            # PnL per event
            rows = bets_by_model.setdefault(m.name, [])
            for i in range(len(fm_test.y)):
                test_row = test_df.iloc[i]
                edge_yes, edge_no, best_side, best_edge = _edge_snapshot(
                    p_model=float(p[i]),
                    market_prob=float(fm_test.market_prob[i]),
                    spread=float(fm_test.market_spread[i]),
                )
                liquidity = _series_float(test_row, "feat_liquidity")
                volume = _series_float(test_row, "raw_volume")
                open_interest = _series_float(test_row, "raw_open_interest")
                trade_allowed, block_reason = _trade_eligibility(
                    spread=float(fm_test.market_spread[i]),
                    liquidity=liquidity,
                    volume=volume,
                    open_interest=open_interest,
                    max_trade_spread=max_trade_spread,
                    min_trade_liquidity=min_trade_liquidity,
                    min_trade_volume=min_trade_volume,
                    min_trade_open_interest=min_trade_open_interest,
                )
                if trade_allowed:
                    side, stake, pnl = _bet_and_pnl(
                        p_model=float(p[i]),
                        market_prob=float(fm_test.market_prob[i]),
                        spread=float(fm_test.market_spread[i]),
                        y=int(fm_test.y[i]),
                        kelly_fraction=kelly_fraction,
                        min_edge=min_edge,
                        max_position=max_position,
                        fee_bps=fee_bps,
                    )
                else:
                    side, stake, pnl = ("none", 0.0, 0.0)
                trade_edge = edge_yes if side == "YES" else edge_no if side == "NO" else 0.0
                rows.append(
                    {
                        "event_id": test_row["event_id"],
                        "category": str(fm_test.category[i]),
                        "open_ts": int(test_row["open_ts"]),
                        "resolve_ts": int(test_row["resolve_ts"]),
                        "time_to_resolution": int(test_row["resolve_ts"] - test_row["open_ts"]),
                        "market_prob": float(fm_test.market_prob[i]),
                        "model_prob": float(p[i]),
                        "spread": float(fm_test.market_spread[i]),
                        "resolved": int(fm_test.y[i]),
                        "edge_yes": edge_yes,
                        "edge_no": edge_no,
                        "best_side": best_side,
                        "best_edge": best_edge,
                        "trade_edge": float(trade_edge),
                        "trade_allowed": bool(trade_allowed),
                        "trade_block_reason": block_reason,
                        "side": side,
                        "stake": float(stake),
                        "pnl": float(pnl),
                        "raw_volume": volume,
                        "raw_open_interest": open_interest,
                        "feat_liquidity": (
                            liquidity
                        ),
                    }
                )

    y_all = np.concatenate(ys) if ys else np.array([], dtype=int)

    result = BacktestResult(n_folds=n_folds, n_events=int(len(y_all)))

    for name, plist in preds.items():
        p_all = np.concatenate(plist)
        bets_df = pd.DataFrame(bets_by_model[name]).sort_values("open_ts", kind="mergesort").reset_index(drop=True)
        traded = bets_df[bets_df["side"] != "none"].reset_index(drop=True)
        trade_summary = _summarize_trade_frame(traded, n_events=len(bets_df))
        n_tradeable = int(bets_df["trade_allowed"].sum()) if "trade_allowed" in bets_df.columns else len(bets_df)

        per_cat = (
            bets_df.groupby("category")
            .agg(
                n_events=("event_id", "count"),
                n_bets=("side", lambda s: int((s != "none").sum())),
                pnl=("pnl", "sum"),
                avg_market_prob=("market_prob", "mean"),
                avg_model_prob=("model_prob", "mean"),
                avg_best_edge=("best_edge", "mean"),
                turnover=("stake", "sum"),
            )
            .reset_index()
            .sort_values("pnl", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
        per_cat["coverage"] = per_cat["n_bets"] / per_cat["n_events"].clip(lower=1)

        slices = _slice_metrics(bets_df)

        result.results[name] = ModelResult(
            name=name,
            brier=brier_score(p_all, y_all),
            log_loss=log_loss(p_all, y_all),
            ece=expected_calibration_error(p_all, y_all),
            roc_auc=_roc_auc(p_all, y_all),
            average_precision=_average_precision(p_all, y_all),
            n_tradeable_events=n_tradeable,
            tradeable_event_rate=float(n_tradeable / len(bets_df)) if len(bets_df) else float("nan"),
            n_bets=int(trade_summary["n_bets"]),
            coverage=float(trade_summary["coverage"]),
            hit_rate=float(trade_summary["hit_rate"]),
            gross_pnl=float(trade_summary["gross_pnl"]),
            final_bankroll=float(trade_summary["final_bankroll"]),
            sharpe=float(trade_summary["sharpe"]),
            sortino=float(trade_summary["sortino"]),
            max_drawdown=float(trade_summary["max_drawdown"]),
            turnover=float(trade_summary["turnover"]),
            avg_stake=float(trade_summary["avg_stake"]),
            avg_trade_edge=float(trade_summary["avg_trade_edge"]),
            median_trade_edge=float(trade_summary["median_trade_edge"]),
            profit_factor=float(trade_summary["profit_factor"]),
            avg_win=float(trade_summary["avg_win"]),
            avg_loss=float(trade_summary["avg_loss"]),
            n_yes_bets=int(trade_summary["n_yes_bets"]),
            n_no_bets=int(trade_summary["n_no_bets"]),
            calibration=calibration_bins(p_all, y_all),
            per_category_pnl=per_cat,
            slices=slices,
            bets=bets_df,
        )

    return result


def _rescore_bets_for_rules(
    bets_df: pd.DataFrame,
    *,
    kelly_fraction: float,
    min_edge: float,
    max_position: float,
    fee_bps: float,
    max_trade_spread: float | None = None,
    min_trade_liquidity: float | None = None,
    min_trade_volume: float | None = None,
    min_trade_open_interest: float | None = None,
) -> dict[str, float | int]:
    rows: list[dict[str, float | str | bool]] = []
    for row in bets_df.itertuples(index=False):
        spread = float(getattr(row, "spread"))
        liquidity = float(getattr(row, "feat_liquidity", float("nan")))
        volume = float(getattr(row, "raw_volume", float("nan")))
        open_interest = float(getattr(row, "raw_open_interest", float("nan")))
        trade_allowed, _ = _trade_eligibility(
            spread=spread,
            liquidity=liquidity,
            volume=volume,
            open_interest=open_interest,
            max_trade_spread=max_trade_spread,
            min_trade_liquidity=min_trade_liquidity,
            min_trade_volume=min_trade_volume,
            min_trade_open_interest=min_trade_open_interest,
        )
        edge_yes, edge_no, _, _ = _edge_snapshot(
            p_model=float(getattr(row, "model_prob")),
            market_prob=float(getattr(row, "market_prob")),
            spread=spread,
        )
        if trade_allowed:
            side, stake, pnl = _bet_and_pnl(
                p_model=float(getattr(row, "model_prob")),
                market_prob=float(getattr(row, "market_prob")),
                spread=spread,
                y=int(getattr(row, "resolved")),
                kelly_fraction=kelly_fraction,
                min_edge=min_edge,
                max_position=max_position,
                fee_bps=fee_bps,
            )
        else:
            side, stake, pnl = ("none", 0.0, 0.0)
        trade_edge = edge_yes if side == "YES" else edge_no if side == "NO" else 0.0
        rows.append({"side": side, "stake": stake, "pnl": pnl, "trade_edge": trade_edge})

    rescored = pd.DataFrame(rows)
    traded = rescored[rescored["side"] != "none"].reset_index(drop=True)
    return _summarize_trade_frame(traded, n_events=len(bets_df))


def sweep_trading_rules(
    result: BacktestResult,
    *,
    kelly_fractions: tuple[float, ...] = (0.05, 0.10, 0.25),
    min_edges: tuple[float, ...] = (0.02, 0.05, 0.10),
    max_positions: tuple[float, ...] = (0.01, 0.02, 0.05),
    max_trade_spreads: tuple[float | None, ...] = (None,),
    min_trade_liquidities: tuple[float | None, ...] = (None,),
    min_trade_volumes: tuple[float | None, ...] = (None,),
    min_trade_open_interests: tuple[float | None, ...] = (None,),
    fee_bps: float = 20.0,
) -> pd.DataFrame:
    """Rescore stored out-of-fold predictions across trading-rule settings.

    This intentionally reuses the predictions in ``BacktestResult``. It lets
    risk settings be compared without repeatedly refitting every model, so the
    sweep measures execution/sizing sensitivity rather than training noise.
    """
    rows: list[dict[str, float | int | str | None]] = []
    for model_name, model_result in result.results.items():
        for kelly, edge, position, spread_gate, liq_gate, vol_gate, oi_gate in product(
            kelly_fractions,
            min_edges,
            max_positions,
            max_trade_spreads,
            min_trade_liquidities,
            min_trade_volumes,
            min_trade_open_interests,
        ):
            summary = _rescore_bets_for_rules(
                model_result.bets,
                kelly_fraction=kelly,
                min_edge=edge,
                max_position=position,
                fee_bps=fee_bps,
                max_trade_spread=spread_gate,
                min_trade_liquidity=liq_gate,
                min_trade_volume=vol_gate,
                min_trade_open_interest=oi_gate,
            )
            rows.append(
                {
                    "model": model_name,
                    "kelly_fraction": kelly,
                    "min_edge": edge,
                    "max_position": position,
                    "max_trade_spread": spread_gate,
                    "min_trade_liquidity": liq_gate,
                    "min_trade_volume": vol_gate,
                    "min_trade_open_interest": oi_gate,
                    **summary,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["sharpe", "final_bankroll", "max_drawdown"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def default_model_factory(fm: FeatureMatrix) -> list[Model]:
    """Thin wrapper that matches the signature expected by walk_forward_backtest."""
    from .models import train_model_zoo
    return train_model_zoo(fm)
