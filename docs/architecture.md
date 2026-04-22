# Architecture

This project is organized as a reproducible research pipeline rather than a
single modeling script. The core contract is: every dataset, synthetic or real,
is normalized to the same market schema before feature engineering, modeling,
backtesting, and reporting.

## Pipeline Overview

```text
synthetic DGP      Kalshi API/cache       CSV export
     |                  |                    |
     v                  v                    v
  data.py          kalshi.py            load_markets_csv
     |                  |                    |
     +------------------+--------------------+
                        |
                        v
                 diagnostics.py
         dataset profile and slice audit
                        |
                        v
                  features.py
          leak-safe feature matrix
                        |
                        v
                   models.py
       baselines, ML models, wrappers
                        |
                        v
                  backtest.py
      walk-forward scoring and Kelly PnL
                        |
                        v
                  pareto.py
       multi-objective model selection
                        |
                        v
                  report.py
       console report and CSV artifacts
```

## `data.py` - Synthetic Teaching DGP

The synthetic generator gives the test suite a controlled world where:

- a true probability exists for every event
- the market price is informative but biased and noisy
- resolutions are Bernoulli draws from the true probability
- known category biases can be recovered only out-of-sample

This lets the project prove that the pipeline can learn real signal before it
touches real exchange data, where ground truth is noisier and latent.

## `kalshi.py` - Real Resolved Markets

The Kalshi adapter is the real-data bridge. It:

- reads public live/recent markets from `GET /markets`
- reads archived settled markets from `GET /historical/markets`
- caches raw JSON under `data/kalshi_cache/`
- maps YES/NO results to `1/0`
- derives midpoint probability and spread from YES bid/ask/last price
- maps timestamps to integer `open_ts` and `resolve_ts`
- canonicalizes exchange categories into the stable model families
- adds Prosperity-style microstructure features:
  `feat_signal`, `feat_momentum`, `feat_dispersion`, and `feat_liquidity`

The CLI exposes this as both `run --kalshi` and `extract-kalshi`. The
extraction command writes a normalized CSV without fitting models, which is the
preferred path for larger pulls because the same dataset can be profiled,
reviewed, versioned, and reused across backtests.

The current adapter uses one market snapshot per row. The next research upgrade
is fixed-horizon candlesticks, for example prices 7 days, 3 days, and 1 day
before settlement, so the backtest measures information available before the
market was resolved.

## `diagnostics.py` - Data Audit

Large exchange pulls need to be audited before they are modeled. The
diagnostics layer computes:

- row counts, unique IDs, duplicate IDs, date ranges, and missingness
- market-prior Brier, log-loss, ROC AUC, and average precision
- category, price-bucket, spread-bucket, tenor-bucket, and liquidity-quartile
  slices

The profile command writes `dataset_summary.csv`, `missingness.csv`, and
`slices.csv`. These files answer "do we have enough data, where is it
concentrated, and how strong is the market baseline?" before a model is allowed
to claim edge.

## `features.py` - Spine Plus Signals

The market logit is the feature spine because a roughly efficient market price
already contains much of the public signal. The model augments it with:

- raw market probability and logit
- spread and normalized time-to-resolution
- category one-hot with an `other` bucket for real exchange taxonomies
- smoothed category base rate computed on the training fold only
- engineered signals: `feat_signal`, `feat_momentum`, `feat_dispersion`
- optional real-data columns: `feat_liquidity`, `feat_sentiment`,
  `feat_news_volume`
- interactions between market logit and time/spread

The key anti-leakage rule: category base rates are fit on the training window
and passed unchanged into the test window.

## `models.py` - Model Zoo

Every model implements `fit(FeatureMatrix)` and `predict(FeatureMatrix)`.

| model | role |
| --- | --- |
| `MarketPriorModel` | pass the market price through; the benchmark to beat |
| `BaseRateModel` | smoothed category historical YES rate |
| `LogisticModel` | interpretable L2 logistic regression |
| `KNNModel` | local nonparametric baseline |
| `GradientBoostingModel` | strong tabular ML baseline with isotonic calibration |
| `IsotonicCalibratedModel` | calibration wrapper for any base model |
| `BayesianShrinkageModel` | logit-scale shrinkage toward the market |
| `StackedEnsemble` | logistic meta-learner over base model probabilities |

The lineup is intentionally diverse. The point is not to over-tune one model;
it is to compare modeling assumptions under the same leak-safe backtest.

## `backtest.py` - Walk-Forward And Kelly

The backtest sorts events by `open_ts`, reserves an initial training window,
and evaluates contiguous forward folds. Each fold:

1. fits feature transforms and category rates on past data only
2. trains fresh models
3. predicts the test window
4. scores Brier, log loss, and ECE
5. simulates YES/NO bets only when model edge clears spread and `min_edge`
6. compounds bankroll with fractional Kelly, fees, and max-position caps

Reported metrics include PnL, final bankroll, hit rate, Sharpe, Sortino, max
drawdown, reliability bins, AUC, average precision, bet coverage, turnover,
profit factor, average traded edge, per-category PnL, model slices, and one row
per simulated bet.

## `pareto.py` - Model Selection

A single leaderboard sort hides real tradeoffs. `pareto.py` marks models as
non-dominated across objectives such as:

- log loss, Brier, and ECE to minimize
- Sharpe, final bankroll, and max drawdown to maximize

This also applies to risk settings. A Kelly fraction with slightly lower final
bankroll but much shallower drawdown may be preferable to the raw best return.

## `report.py` - Human And Machine Outputs

The CLI prints a compact leaderboard plus per-model detail. With `--out`, it
writes:

- `summary.txt`
- `leaderboard.csv`
- `normalized_dataset.csv`
- `dataset_summary.csv`
- `missingness.csv`
- `slices.csv`
- `calibration_<model>.csv`
- `pnl_by_category_<model>.csv`
- `slices_<model>.csv`
- `bets_<model>.csv`

These artifacts are deliberately flat so they can feed a notebook, dashboard,
or later data warehouse without reverse-engineering object state.

## What This Deliberately Does Not Do Yet

- It does not execute live trades.
- It does not model queue position, partial fills, slippage, latency, or
  exchange position limits.
- It does not claim that synthetic alpha transfers to real markets.
- It does not yet pull fixed-horizon candlestick snapshots before settlement.
- It does not perform large hyperparameter sweeps or deflated-Sharpe
  corrections, though the Pareto and reporting pieces make those natural next
  additions.
