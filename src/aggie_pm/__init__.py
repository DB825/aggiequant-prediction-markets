"""AggieQuant prediction-markets algorithm.

A research-grade reference implementation that members can study, extend,
and point to as an example of a full quant pipeline: synthetic data
generation, feature engineering, a model zoo (logistic regression, KNN,
gradient boosting, isotonic calibration, stacked ensemble, Bayesian
shrinkage), a walk-forward backtest with fractional Kelly sizing, and a
full performance report (Brier, log loss, calibration, Sharpe, Sortino,
max drawdown, per-category PnL).

The pure-stdlib teaching script lives next to this package at
`src/prediction_markets.py` and is intentionally preserved. This package
is the "advanced track" that builds on top of it.

See `REFERENCES.md` for papers behind every choice made in this code.
"""

from .data import MarketEvent, generate_synthetic_markets, load_markets_csv
from .features import FeatureMatrix, build_features
from .models import (
    BaseRateModel,
    BayesianShrinkageModel,
    GradientBoostingModel,
    IsotonicCalibratedModel,
    KNNModel,
    LogisticModel,
    MarketPriorModel,
    StackedEnsemble,
    train_model_zoo,
)
from .backtest import BacktestResult, walk_forward_backtest
from .kalshi import build_orderbook_features, kalshi_markets_to_dataframe, load_kalshi_resolved
from .pareto import pareto_front, pareto_mask, rank_by_domination_count
from .report import format_report, save_report

__all__ = [
    "MarketEvent",
    "generate_synthetic_markets",
    "load_markets_csv",
    "FeatureMatrix",
    "build_features",
    "BaseRateModel",
    "BayesianShrinkageModel",
    "GradientBoostingModel",
    "IsotonicCalibratedModel",
    "KNNModel",
    "LogisticModel",
    "MarketPriorModel",
    "StackedEnsemble",
    "train_model_zoo",
    "BacktestResult",
    "walk_forward_backtest",
    "build_orderbook_features",
    "kalshi_markets_to_dataframe",
    "load_kalshi_resolved",
    "pareto_front",
    "pareto_mask",
    "rank_by_domination_count",
    "format_report",
    "save_report",
]
