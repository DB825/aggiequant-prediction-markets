# aggiequant-prediction-markets

A research-grade prediction-market probability pipeline: synthetic data
generation, Kalshi resolved-market ingestion, leak-safe feature engineering,
a heterogeneous model zoo, walk-forward backtesting, fractional Kelly sizing,
proper scoring rules, calibration diagnostics, per-category PnL, and
Pareto-front model selection.

**Status:** personal project by [Dylan Bago](https://github.com/DB825),
originally built as the showcase deliverable for the AggieQuant
prediction-markets project pod.

> The goal is not "fit a model and show a high accuracy." The goal is to
> beat a base-rate model first, then beat the market after spread and fees,
> using an evaluation path an auditor could replay.

## Highlights

- **Synthetic-to-real bridge.** `data.py` creates an honest latent-probability
  data-generating process for teaching and tests. `kalshi.py` maps resolved
  Kalshi market snapshots into the same schema, so the exact same model and
  backtest code can run on public exchange data.
- **Kalshi dataset support.** The adapter reads public market data from
  Kalshi's live `GET /markets` and archived `GET /historical/markets`
  endpoints, caches raw JSON under `data/kalshi_cache/`, canonicalizes
  exchange categories into stable model families, and derives orderbook-style
  features from bid/ask, last price, previous price, volume, and open interest.
- **Leak-safe features.** Category base rates are estimated on the training
  fold only and frozen before scoring the test fold. Optional real-data
  signals such as `feat_liquidity`, `feat_sentiment`, and `feat_news_volume`
  are picked up automatically when present.
- **Model zoo.** MarketPrior, BaseRate, Logistic, MicrostructureResidual,
  MicrostructureGBM, k-NN, gradient boosting, isotonic calibration, Bayesian
  shrinkage toward the market, and stacked ensembles all share the same
  `Model` protocol.
- **Trading-aware backtest.** Walk-forward folds, fractional Kelly sizing,
  spread and fee costs, Brier score, log loss, ECE, Sharpe, Sortino, max
  drawdown, bankroll path, and per-category PnL are reported together.
- **Large-dataset workflow.** `extract-kalshi` can build a reusable normalized
  CSV from many Kalshi pages, and `profile` audits category coverage,
  price/spread/liquidity buckets, duplicate IDs, missingness, and market-prior
  baseline metrics before any model is trained.
- **Pareto model selection.** `pareto.py` identifies non-dominated models
  across calibration, log loss, Sharpe, drawdown, and bankroll instead of
  over-selling one leaderboard metric.

## Install

```powershell
pip install -e ".[dev]"
```

Requires Python 3.10+. Core dependencies are NumPy, pandas, SciPy, and
scikit-learn. The Kalshi adapter uses the standard library for HTTP reads,
so no API key or extra package is required for public market-data pulls.

## Quickstart

Run the synthetic benchmark:

```powershell
aggie-pm run
```

Run against a CSV that follows the pipeline schema:

```powershell
aggie-pm run --csv path/to/markets.csv --out reports/csv_run
```

Fetch resolved Kalshi markets and run the same backtest:

```powershell
aggie-pm run --kalshi --kalshi-source historical --kalshi-pages 50 --out reports/kalshi
```

Build and profile a larger reusable Kalshi dataset first:

```powershell
aggie-pm extract-kalshi --kalshi-source historical --kalshi-pages 50 --kalshi-page-limit 1000 --out data/kalshi_resolved.csv
aggie-pm profile --csv data/kalshi_resolved.csv --out reports/kalshi_profile
aggie-pm run --csv data/kalshi_resolved.csv --out reports/kalshi_large
```

Build the leak-safe multi-year macro case study from recurring Kalshi CPI/Fed
markets. This uses settled market rows only for labels, fetches historical
candlesticks for point-in-time 30/14/7/3/1-day snapshots, runs walk-forward
models, and rescoring sweeps for Kelly/edge/position/spread settings:

```powershell
aggie-pm case-study --series KXCPI,KXFED --out reports/kalshi_macro_case_study --out-data reports/kalshi_macro_snapshots.csv
```

The default candle fetcher is intentionally single-threaded to avoid Kalshi
rate limits. Use `--workers 2` only after the cache has warmed.

Scan the generated macro snapshots for threshold-ladder relative-value trades:

```powershell
aggie-pm ladder-study --csv reports/kalshi_macro_snapshots.csv --out reports/kalshi_ladder_study
```

This parses CPI/Fed thresholds, repairs each probability ladder with isotonic
regression, and tests the dominance package `YES(lower threshold) + NO(higher
threshold)`. That package should pay at least $1, so it is an arbitrage only
when the all-in package cost clears $1 after spread and fees.

Useful knobs:

| flag | default | meaning |
| --- | ---: | --- |
| `--n-events` | 2000 | synthetic dataset size |
| `--seed` | 20260421 | synthetic DGP seed |
| `--csv` | none | load a local real-data CSV |
| `--kalshi` | false | fetch resolved Kalshi markets |
| `--kalshi-source` | `historical` | `historical` for archived settled markets, `live` for recent markets |
| `--kalshi-series` | none | optional Kalshi `series_ticker` filter |
| `--kalshi-event` | none | optional Kalshi `event_ticker` filter |
| `--kalshi-pages` | 50 | max Kalshi pages to fetch |
| `--kalshi-page-limit` | 1000 | Kalshi page size |
| `--folds` | 6 | walk-forward folds |
| `--kelly` | 0.25 | Kelly fraction |
| `--min-edge` | 0.02 | min model-vs-market edge required to place a bet |
| `--max-position` | 0.05 | bankroll fraction cap per bet |
| `--fee-bps` | 20 | round-trip fee in basis points |
| `--max-trade-spread` | none | optional gate that blocks trades in markets wider than this spread |
| `--min-trade-liquidity` | none | optional gate on `feat_liquidity` before a bet can fire |
| `--sweep` | false | rescore out-of-fold predictions across Kelly/edge/position settings |
| `--out` | none | directory to write artifacts |
| `--save-dataset` | none | write the normalized dataset used in a run to CSV |

Case-study specific knobs:

| flag | default | meaning |
| --- | ---: | --- |
| `--series` | `KXCPI,KXFED` | recurring Kalshi series tickers for the case study |
| `--horizons-days` | `30,14,7,3,1` | fixed pre-close snapshot horizons |
| `--period-interval` | `1440` | candle period in minutes: `1`, `60`, or `1440` |
| `--cache-dir` | `data/kalshi_cache/case_study` | local cache for historical candles |
| `--workers` | `1` | parallel candle fetch workers |
| `--out-data` | `reports/kalshi_macro_snapshots.csv` | point-in-time snapshot CSV |

Ladder-study knobs:

| flag | default | meaning |
| --- | ---: | --- |
| `--csv` | `reports/kalshi_macro_snapshots.csv` | point-in-time snapshot CSV |
| `--fee-bps` | `20` | fee estimate applied to dominance packages |
| `--min-edge` | `0` | required after-fee guaranteed edge |
| `--stake` | `0.01` | bankroll fraction per selected pair trade |

## Dataset Contract

The core schema is intentionally narrow:

| column | meaning |
| --- | --- |
| `event_id` | stable market or question id |
| `category` | canonical category family |
| `question` | human-readable market title |
| `market_prob` | YES-side market-implied probability |
| `market_spread` | round-trip bid/ask spread |
| `open_ts` | integer time index for market open |
| `resolve_ts` | integer time index for settlement |
| `resolved` | binary outcome, 1 for YES and 0 for NO |
| `feat_signal` | external directional signal |
| `feat_momentum` | short-horizon momentum signal |
| `feat_dispersion` | uncertainty, disagreement, or spread signal |

Optional columns are included when available:

| optional column | intended source |
| --- | --- |
| `feat_liquidity` | Kalshi volume/open-interest or orderbook depth |
| `feat_sentiment` | news/social/LLM sentiment around the question or event |
| `feat_news_volume` | article count, post count, or search intensity |

This gives the project a clean data-engineering split:

1. **Bronze:** raw Kalshi JSON cached exactly as received.
2. **Silver:** exchange-specific fields normalized to the core schema.
3. **Gold:** leak-safe feature matrix, model predictions, bets, slice metrics,
   and report CSVs.

For large pulls, the profiling layer writes:

- `dataset_summary.csv` - row counts, unique IDs, date range, spread/liquidity
  summary, and market-prior Brier/log-loss/AUC.
- `missingness.csv` - missing counts and rates by column.
- `slices.csv` - market-prior metrics by category, price bucket, spread bucket,
  time-to-resolution bucket, and liquidity quartile when available.

## Kalshi Integration

Kalshi's docs currently describe public, unauthenticated market-data reads
against `https://api.elections.kalshi.com/trade-api/v2`, including live
market reads and historical endpoints for older settled markets. The adapter
uses that split directly:

- `KalshiClient.fetch_markets(source="live")` calls `GET /markets`, usually
  with `status="settled"`.
- `KalshiClient.fetch_markets(source="historical")` calls
  `GET /historical/markets` for archived settled markets.
- `load_kalshi_resolved()` fetches or reads the cache, drops unresolved rows,
  maps YES/NO results to `1/0`, derives a midpoint price, computes spread,
  maps timestamps to walk-forward indices, and attaches microstructure
  features.
- `aggie-pm extract-kalshi` runs that same normalization path without fitting
  models, so a large pull can be cached, profiled, reviewed, and reused.

See the official Kalshi docs for
[public market data](https://docs.kalshi.com/getting_started/quick_start_market_data),
[historical data](https://docs.kalshi.com/getting_started/historical_data),
and the [`GET /historical/markets` reference](https://docs.kalshi.com/api-reference/historical/get-historical-markets).

## Modeling Approach

The model is built around the idea that the market price is the strongest
single predictor, but not the only useful predictor. Every model sees the
market probability/logit and then adds:

- category one-hot and smoothed category base rate
- time-to-resolution and spread
- Kalshi microstructure features inspired by market-making work:
  signed last-trade pressure, spread-normalized momentum, dispersion, and
  liquidity
- optional sentiment and news-volume features
- interactions between market logit, time-to-resolution, and spread

The backtest asks three questions in order:

1. **Forecasting:** Does the model beat the market on log loss or Brier?
2. **Calibration:** Is the model reliable enough for sizing?
3. **Trading:** Does the edge survive spread, fees, drawdown, and Kelly sizing?

`pareto.py` then asks which models are non-dominated across those objectives.
This is the right framing for a portfolio operator: the most calibrated model,
the highest Sharpe model, and the lowest drawdown model may be different.

The most explicit microstructure thesis is the residual family:
`MicrostructureResidualModel` and `MicrostructureGBMModel` use the market price
as the prior, learn a correction from public book-quality signals, and tune how
much to trust that correction on a walk-forward calibration slice. If
microstructure does not improve the late training window, the model shrinks
back toward the market prior.

The backtest also emits market-relative Brier/log-loss improvement, ROC AUC,
average precision, bet coverage, turnover, average traded edge, profit factor,
YES/NO bet counts, and slice-level diagnostics. Those extra columns matter
most on large datasets, where a headline Sharpe can hide that all the signal
came from one category, one spread regime, or one tenor bucket.

Trading rules are intentionally separated from forecast scoring. Optional
tradability gates such as `--max-trade-spread` and `--min-trade-liquidity`
can block a row from trading while still leaving it in Brier/log-loss/AUC
evaluation. With `--sweep`, the CLI reuses the same out-of-fold predictions
to produce `trading_sweep.csv` across Kelly fractions, edge thresholds,
position caps, and spread gates without retraining the model zoo.

## Related Portfolio Work

This repo can act as the hub that ties several strands of your work together:

| prior work theme | how it improves this project |
| --- | --- |
| Prosperity / market-making bots | orderbook pressure, spread-normalized momentum, inventory-aware sizing intuition |
| Pareto realizations | non-dominated model and Kelly-fraction selection instead of single-metric leaderboard chasing |
| Data engineering pipelines | bronze/silver/gold exchange-data flow, cached raw pulls, schema contracts, reproducible reports |
| Sentiment work | optional `feat_sentiment` and `feat_news_volume` hooks for news, social, or LLM-scored text features |
| Machine learning practice | interpretable logistic baseline, nonparametric k-NN, GBDT, calibration wrappers, shrinkage, stacking |

## Notebooks

- `notebooks/01_walkthrough.py` - synthetic end-to-end pipeline walkthrough.
- `notebooks/02_kalshi_research_notebook.py` - real-data research notebook:
  Kalshi pull/cache, EDA, feature audit, walk-forward backtest, Pareto front,
  and project-positioning notes. The file uses `# %%` cells, so it runs as a
  plain Python script or opens naturally in VS Code/Jupyter.

## Repo Layout

```text
.
|-- pyproject.toml
|-- README.md
|-- REFERENCES.md
|-- data/
|   `-- sample_markets.csv
|-- docs/
|   |-- architecture.md
|   |-- handoff.md
|   `-- results.md
|-- notebooks/
|   |-- 01_walkthrough.py
|   `-- 02_kalshi_research_notebook.py
|-- src/
|   |-- prediction_markets.py
|   `-- aggie_pm/
|       |-- data.py
|       |-- diagnostics.py
|       |-- features.py
|       |-- kalshi.py
|       |-- pareto.py
|       |-- models.py
|       |-- backtest.py
|       |-- report.py
|       `-- cli.py
`-- tests/
```

## Running Tests

```powershell
pytest
```

The suite covers deterministic data generation, CSV loading, feature
engineering, optional real-data features, every model's fit/predict contract,
scoring-rule properties, Kelly sizing, walk-forward correctness, Kalshi schema
mapping, Pareto-front selection, and the CLI smoke path.

The strongest portfolio version is not "I built a prediction-market model."
It is: "I built a reproducible research pipeline that tests whether public
signals improve prediction-market probabilities after calibration, spread,
fees, and drawdown constraints."

## Caveats

- Synthetic results are a systems test, not evidence of real alpha.
- A single Kalshi snapshot is weaker than a timestamped pre-close dataset.
  The next upgrade is to pull candlesticks at fixed horizons before settlement.
- Kelly assumes probabilities are known. Here they are estimated, so fractional
  Kelly and max-position caps are risk controls, not guarantees.
- Live trading, slippage, order depth, latency, and exchange position limits are
  intentionally out of scope.

## License

MIT. See [`LICENSE`](LICENSE).
