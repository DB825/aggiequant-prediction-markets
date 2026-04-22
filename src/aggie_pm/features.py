"""Feature engineering for the prediction-markets pipeline.

The market's own price is, by far, the strongest single feature: in a
roughly-efficient market the market logit already absorbs most of the
predictable signal (see Manski 2006 on interpreting market prices as
probabilities). So we take the market logit as the spine of every feature
vector and *augment* it with signals a research team might credibly have:

- Raw market price and its logit.
- Category one-hot (shrinks the per-category bias we saw in data.py).
- A rolling base-rate for the event's category over the training window.
- Time-to-resolution (short-dated markets drift differently).
- Book width (``market_spread``) as a liquidity / disagreement proxy.
- The engineered latent features (``feat_signal``, ``feat_momentum``,
  ``feat_dispersion``) that real research teams would derive from public
  data (economic surprises, polling, Elo, implied vol, etc.).
- Interactions: market_logit × category and market_logit × time_to_resolve,
  which let the model learn "the bias is biggest in crypto near resolution,"
  etc., without forcing a separate model per category.

All features are computed deterministically per row. The one stateful
piece - the rolling category base rate - is computed on the *training*
split only and frozen before it is applied to the test split, to prevent
lookahead bias (Bailey et al. 2014 on backtest overfitting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CATEGORIES


@dataclass(frozen=True)
class FeatureMatrix:
    """A feature matrix plus everything needed to score it honestly.

    Attributes
    ----------
    X : np.ndarray, shape (n_rows, n_features)
    y : np.ndarray, shape (n_rows,)
        Binary resolution outcomes.
    market_prob : np.ndarray, shape (n_rows,)
    market_spread : np.ndarray, shape (n_rows,)
        Round-trip spread (used as a transaction cost in the backtest).
    category : np.ndarray of str, shape (n_rows,)
    feature_names : tuple[str, ...]
        Column names in ``X``.
    """

    X: np.ndarray
    y: np.ndarray
    market_prob: np.ndarray
    market_spread: np.ndarray
    category: np.ndarray
    feature_names: tuple[str, ...]


def _logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def build_features(
    df: pd.DataFrame,
    *,
    category_base_rates: dict[str, float] | None = None,
) -> tuple[FeatureMatrix, dict[str, float]]:
    """Turn a raw market DataFrame into a model-ready ``FeatureMatrix``.

    If ``category_base_rates`` is passed in, it is used as-is (correct
    behaviour for a test split). If it is ``None``, base rates are computed
    from the given DataFrame and returned so the caller can freeze them
    before scoring the test split.
    """

    if category_base_rates is None:
        category_base_rates = (
            df.groupby("category")["resolved"].mean().to_dict()
            if "resolved" in df.columns
            else {c: 0.5 for c in CATEGORIES}
        )
        # Smooth toward 0.5 with a pseudo-count so tiny categories don't blow up.
        n_pseudo = 20
        for c in CATEGORIES:
            n = int((df["category"] == c).sum())
            raw = category_base_rates.get(c, 0.5)
            category_base_rates[c] = (raw * n + 0.5 * n_pseudo) / (n + n_pseudo)

    mkt_p = df["market_prob"].to_numpy(dtype=float)
    mkt_l = _logit(mkt_p)
    spread = df["market_spread"].to_numpy(dtype=float)
    ttr = (df["resolve_ts"] - df["open_ts"]).to_numpy(dtype=float)
    ttr_n = ttr / max(ttr.max(), 1.0)

    cat = df["category"].to_numpy()
    cat_rate = np.array([category_base_rates.get(c, 0.5) for c in cat])

    # Category one-hot, deterministic column order from CATEGORIES tuple.
    one_hot = np.zeros((len(df), len(CATEGORIES)), dtype=float)
    for j, c in enumerate(CATEGORIES):
        one_hot[:, j] = (cat == c).astype(float)

    feat_signal = df["feat_signal"].to_numpy(dtype=float)
    feat_mom = df["feat_momentum"].to_numpy(dtype=float)
    feat_disp = df["feat_dispersion"].to_numpy(dtype=float)

    # Interactions
    mkt_l_x_ttr = mkt_l * ttr_n
    mkt_l_x_spread = mkt_l * spread

    base_cols = {
        "market_prob": mkt_p,
        "market_logit": mkt_l,
        "market_spread": spread,
        "ttr_norm": ttr_n,
        "category_base_rate": cat_rate,
        "feat_signal": feat_signal,
        "feat_momentum": feat_mom,
        "feat_dispersion": feat_disp,
        "mkt_logit_x_ttr": mkt_l_x_ttr,
        "mkt_logit_x_spread": mkt_l_x_spread,
    }

    X_cols = list(base_cols.values()) + [one_hot[:, j] for j in range(len(CATEGORIES))]
    names = tuple(list(base_cols.keys()) + [f"cat_{c}" for c in CATEGORIES])
    X = np.column_stack(X_cols)

    y = df["resolved"].to_numpy(dtype=int) if "resolved" in df.columns else np.zeros(len(df), dtype=int)

    fm = FeatureMatrix(
        X=X,
        y=y,
        market_prob=mkt_p,
        market_spread=spread,
        category=cat,
        feature_names=names,
    )
    return fm, category_base_rates
