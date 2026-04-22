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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

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


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    name: str
    brier: float
    log_loss: float
    ece: float
    n_bets: int
    hit_rate: float
    gross_pnl: float
    final_bankroll: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calibration: pd.DataFrame
    per_category_pnl: pd.DataFrame
    bets: pd.DataFrame


@dataclass
class BacktestResult:
    n_folds: int
    n_events: int
    results: dict[str, ModelResult] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for r in self.results.values():
            rows.append(
                {
                    "model": r.name,
                    "brier": r.brier,
                    "log_loss": r.log_loss,
                    "ece": r.ece,
                    "n_bets": r.n_bets,
                    "hit_rate": r.hit_rate,
                    "gross_pnl": r.gross_pnl,
                    "final_bankroll": r.final_bankroll,
                    "sharpe": r.sharpe,
                    "sortino": r.sortino,
                    "max_drawdown": r.max_drawdown,
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
) -> BacktestResult:
    """Run a walk-forward backtest.

    Parameters
    ----------
    df : DataFrame
        The full market history, sorted by ``open_ts``. Must match the
        schema produced by ``data.generate_synthetic_markets``.
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
    """

    df = df.sort_values("open_ts", kind="mergesort").reset_index(drop=True)
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
                rows.append(
                    {
                        "event_id": test_df.iloc[i]["event_id"],
                        "category": str(fm_test.category[i]),
                        "open_ts": int(test_df.iloc[i]["open_ts"]),
                        "market_prob": float(fm_test.market_prob[i]),
                        "model_prob": float(p[i]),
                        "spread": float(fm_test.market_spread[i]),
                        "resolved": int(fm_test.y[i]),
                        "side": side,
                        "stake": float(stake),
                        "pnl": float(pnl),
                    }
                )

    y_all = np.concatenate(ys) if ys else np.array([], dtype=int)

    result = BacktestResult(n_folds=n_folds, n_events=int(len(y_all)))

    for name, plist in preds.items():
        p_all = np.concatenate(plist)
        bets_df = pd.DataFrame(bets_by_model[name]).sort_values("open_ts", kind="mergesort").reset_index(drop=True)
        traded = bets_df[bets_df["side"] != "none"].reset_index(drop=True)
        returns = traded["pnl"].to_numpy() if len(traded) else np.array([0.0])
        bankroll = np.cumprod(1.0 + returns) if len(traded) else np.array([1.0])

        per_cat = (
            bets_df.groupby("category")
            .agg(
                n_events=("event_id", "count"),
                n_bets=("side", lambda s: int((s != "none").sum())),
                pnl=("pnl", "sum"),
            )
            .reset_index()
            .sort_values("pnl", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )

        hit = float((traded["pnl"] > 0).mean()) if len(traded) else float("nan")

        result.results[name] = ModelResult(
            name=name,
            brier=brier_score(p_all, y_all),
            log_loss=log_loss(p_all, y_all),
            ece=expected_calibration_error(p_all, y_all),
            n_bets=int(len(traded)),
            hit_rate=hit,
            gross_pnl=float(traded["pnl"].sum()) if len(traded) else 0.0,
            final_bankroll=float(bankroll[-1]) if len(bankroll) else 1.0,
            sharpe=_sharpe(returns),
            sortino=_sortino(returns),
            max_drawdown=_max_drawdown(bankroll) if len(bankroll) else 0.0,
            calibration=calibration_bins(p_all, y_all),
            per_category_pnl=per_cat,
            bets=bets_df,
        )

    return result


def default_model_factory(fm: FeatureMatrix) -> list[Model]:
    """Thin wrapper that matches the signature expected by walk_forward_backtest."""
    from .models import train_model_zoo
    return train_model_zoo(fm)
