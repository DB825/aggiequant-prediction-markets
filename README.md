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
- **Model zoo.** MarketPrior, BaseRate, Logistic, k-NN, gradient boosting,
  isotonic calibration, Bayesian shrinkage toward the market, and stacked
  ensembles all share the same `Model` protocol.
- **Trading-aware backtest.** Walk-forward folds, fractional Kelly sizing,
  spread and fee costs, Brier score, log loss, ECE, Sharpe, Sortino, max
  drawdown, bankroll path, and per-category PnL are reported together.
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
aggie-pm run --kalshi --kalshi-source historical --kalshi-pages 5 --out reports/kalshi
```

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
| `--kalshi-pages` | 5 | max Kalshi pages to fetch |
| `--folds` | 6 | walk-forward folds |
| `--kelly` | 0.25 | Kelly fraction |
| `--min-edge` | 0.02 | min model-vs-market edge required to place a bet |
| `--max-position` | 0.05 | bankroll fraction cap per bet |
| `--fee-bps` | 20 | round-trip fee in basis points |
| `--out` | none | directory to write artifacts |

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
3. **Gold:** leak-safe feature matrix, model predictions, bets, and report CSVs.

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
