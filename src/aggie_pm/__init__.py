"""AggieQuant prediction-markets algorithm.

A research-grade reference implementation that members can study, extend,
and point to as an example of a full quant pipeline: synthetic data
generation, feature engineering, a model zoo (logistic regression, KNN,
gradient boosting, isotonic calibration, stacked ensemble, Bayesian
shrinkage), a walk-forward backtest with fractional Kelly sizing, and a
full performance report (Brier, log loss, calibration, AUC, Sharpe, Sortino,
max drawdown, market-relative edge, slice diagnostics, per-category PnL).

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
    MicrostructureGBMModel,
    MicrostructureResidualModel,
    StackedEnsemble,
    train_model_zoo,
)
from .backtest import BacktestResult, sweep_trading_rules, walk_forward_backtest
from .case_study import build_snapshot_dataset, candlesticks_to_frame, run_kalshi_macro_case_study
from .diagnostics import format_dataset_profile, profile_market_dataset, save_dataset_profile
from .kalshi import build_orderbook_features, kalshi_markets_to_dataframe, load_kalshi_resolved
from .pareto import pareto_front, pareto_mask, rank_by_domination_count
from .relative_value import (
    backtest_pair_trades,
    build_monotonic_pair_opportunities,
    extract_threshold,
    repair_ladder_probabilities,
    run_ladder_study,
)
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
    "MicrostructureGBMModel",
    "MicrostructureResidualModel",
    "StackedEnsemble",
    "train_model_zoo",
    "BacktestResult",
    "sweep_trading_rules",
    "walk_forward_backtest",
    "build_snapshot_dataset",
    "candlesticks_to_frame",
    "run_kalshi_macro_case_study",
    "format_dataset_profile",
    "profile_market_dataset",
    "save_dataset_profile",
    "build_orderbook_features",
    "kalshi_markets_to_dataframe",
    "load_kalshi_resolved",
    "pareto_front",
    "pareto_mask",
    "rank_by_domination_count",
    "extract_threshold",
    "repair_ladder_probabilities",
    "build_monotonic_pair_opportunities",
    "backtest_pair_trades",
    "run_ladder_study",
    "format_report",
    "save_report",
]
