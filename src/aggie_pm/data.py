"""Synthetic market dataset generator.

We want a dataset where:

  1. There is a ground-truth probability for each event that is a smooth
     function of observable features (otherwise there is nothing to learn).
  2. The observed market price is a noisy, slightly biased estimate of that
     probability (otherwise the market is unbeatable and nothing is learned).
  3. Realizations ``y`` are Bernoulli draws from the true probability
     (so a perfectly calibrated model still has irreducible Brier noise equal
     to ``p*(1-p)``, which is the honest ceiling).

The bias term is deliberate: real prediction markets exhibit documented
favourite-longshot bias (Wolfers & Zitzewitz 2004) and persistent category
biases. The whole point of the pod is to let members discover and exploit
those biases out-of-sample.

We expose both ``generate_synthetic_markets`` (returns a pandas DataFrame
with engineered fields pre-populated) and ``load_markets_csv`` (reads a
real-data CSV in the same schema so the pipeline works identically against
Polymarket/Kalshi exports once a member wires them in).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CATEGORIES = (
    "macro",
    "earnings",
    "policy",
    "sports",
    "weather",
    "crypto",
    "geopolitics",
    "entertainment",
)

# Per-category "market bias": how much the market over- or under-prices YES.
# Positive = market systematically overprices YES (favourite bias etc.).
# These are the alpha to be discovered out-of-sample.
_CATEGORY_BIAS: dict[str, float] = {
    "macro": -0.015,
    "earnings": +0.020,
    "policy": -0.040,
    "sports": +0.055,     # favourite-longshot
    "weather": -0.010,
    "crypto": +0.070,     # retail over-enthusiasm
    "geopolitics": -0.030,
    "entertainment": +0.045,
}

# Per-category base rate for YES resolution.
_CATEGORY_BASE_RATE: dict[str, float] = {
    "macro": 0.46,
    "earnings": 0.58,
    "policy": 0.72,
    "sports": 0.50,
    "weather": 0.38,
    "crypto": 0.44,
    "geopolitics": 0.31,
    "entertainment": 0.55,
}


@dataclass(frozen=True)
class MarketEvent:
    """A single resolved prediction-market event.

    All probabilities are on [0, 1]. ``resolved`` is 0/1 and only known after
    settlement. ``open_ts`` and ``resolve_ts`` are integer day indices so the
    backtester can walk forward strictly in time order.
    """

    event_id: str
    category: str
    question: str
    market_prob: float
    market_spread: float
    open_ts: int
    resolve_ts: int
    true_prob: float
    resolved: int
    # Latent features that drive the true probability. In a real system these
    # would be things like recent economic surprise index, implied vol,
    # polling spread, Elo differential, etc.
    feat_signal: float
    feat_momentum: float
    feat_dispersion: float


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def generate_synthetic_markets(
    n_events: int = 2000,
    *,
    seed: int = 20260421,
    n_days: int = 520,
    categories: Iterable[str] = CATEGORIES,
) -> pd.DataFrame:
    """Generate ``n_events`` resolved markets spanning ``n_days`` days.

    The DGP is:

        logit(p_true) = alpha_category
                      + 1.5 * feat_signal
                      + 0.7 * feat_momentum
                      - 0.4 * feat_dispersion
                      + eps_true         (eps_true ~ N(0, 0.25))

        market_prob   = sigmoid( logit(p_true) + bias_category + eps_mkt )
                        (eps_mkt ~ N(0, 0.45))   # informative but noisy

        y             ~ Bernoulli(p_true)

    The market therefore has real information (its logit correlates
    strongly with the true logit) but is biased per category and noisy
    per event. That leaves measurable, exploitable edge for a model
    that conditions on the features.
    """

    rng = np.random.default_rng(seed)
    cats = list(categories)

    # Per-category intercept on the logit scale.
    alpha = {c: _logit(_CATEGORY_BASE_RATE.get(c, 0.5)) for c in cats}

    cat_idx = rng.integers(0, len(cats), size=n_events)
    cat_col = np.array([cats[i] for i in cat_idx])

    feat_signal = rng.normal(0.0, 1.0, size=n_events)
    feat_momentum = 0.5 * feat_signal + rng.normal(0.0, 0.8, size=n_events)
    feat_dispersion = np.abs(rng.normal(0.0, 1.0, size=n_events))

    alpha_vec = np.array([alpha[c] for c in cat_col])
    logit_true = (
        alpha_vec
        + 1.5 * feat_signal
        + 0.7 * feat_momentum
        - 0.4 * feat_dispersion
        + rng.normal(0.0, 0.25, size=n_events)
    )
    true_prob = _logistic(logit_true)

    bias_vec = np.array([_logit(0.5 + _CATEGORY_BIAS[c]) for c in cat_col])
    logit_market = logit_true + bias_vec + rng.normal(0.0, 0.45, size=n_events)
    market_prob = _logistic(logit_market)

    # Spread widens with dispersion and at price extremes (mirrors real books).
    market_spread = np.clip(
        0.015
        + 0.020 * feat_dispersion
        + 0.030 * (np.abs(market_prob - 0.5) > 0.35).astype(float)
        + rng.normal(0.0, 0.005, size=n_events),
        0.005,
        0.10,
    )

    # Strict time order: open_ts uniformly across [0, n_days-30], resolve
    # 3–30 days later. This lets the backtester train on the past only.
    open_ts = rng.integers(0, max(n_days - 30, 1), size=n_events)
    hold = rng.integers(3, 30, size=n_events)
    resolve_ts = open_ts + hold

    y = rng.binomial(1, true_prob)

    df = pd.DataFrame(
        {
            "event_id": [f"pm-{i:05d}" for i in range(n_events)],
            "category": cat_col,
            "question": [
                f"[{c}] Will event #{i} resolve YES?" for i, c in enumerate(cat_col)
            ],
            "market_prob": market_prob,
            "market_spread": market_spread,
            "open_ts": open_ts,
            "resolve_ts": resolve_ts,
            "true_prob": true_prob,          # held out: only used for DGP audit
            "resolved": y.astype(int),
            "feat_signal": feat_signal,
            "feat_momentum": feat_momentum,
            "feat_dispersion": feat_dispersion,
        }
    )
    return df.sort_values("open_ts", kind="mergesort").reset_index(drop=True)


def load_markets_csv(path: str | Path) -> pd.DataFrame:
    """Load a real-data CSV that matches the synthetic schema.

    Required columns: ``event_id, category, question, market_prob,
    market_spread, open_ts, resolve_ts, resolved``. Feature columns
    (``feat_signal``, ``feat_momentum``, ``feat_dispersion``) are optional;
    if missing they will be synthesized as zeros so the market-only models
    still work.
    """

    df = pd.read_csv(path)
    required = {
        "event_id",
        "category",
        "question",
        "market_prob",
        "market_spread",
        "open_ts",
        "resolve_ts",
        "resolved",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    for col in ("feat_signal", "feat_momentum", "feat_dispersion"):
        if col not in df.columns:
            df[col] = 0.0
    if "true_prob" not in df.columns:
        df["true_prob"] = np.nan
    return df.sort_values("open_ts", kind="mergesort").reset_index(drop=True)
