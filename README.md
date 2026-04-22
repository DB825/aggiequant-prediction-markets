# aggiequant-prediction-markets

A research-grade prediction-market probability pipeline: synthetic data
generator, feature engineering with leak-safe category priors, a model
zoo (logistic regression, k-NN, gradient boosting, isotonic calibration,
Bayesian shrinkage toward the market, stacked ensemble), and a
walk-forward backtest with fractional Kelly sizing, Brier / log-loss /
ECE / Sharpe / Sortino / max-drawdown / per-category PnL.

**Status:** personal project by [Dylan Bago](https://github.com/DB825),
originally built as the showcase deliverable for the
[AggieQuant](https://github.com/) *prediction-markets* project pod.

> The point is to beat a base-rate model first, then beat the market
> *after* spread and fees, with a backtest an auditor would sign off on.

## Highlights

- **Honest DGP.** Synthetic events are drawn from a latent-logit model
  where the market price is informative but biased per category -
  mirroring documented favourite-longshot bias (Wolfers & Zitzewitz
  2004). A model that conditions on features can measurably beat the
  market; a model that ignores features cannot.
- **Leak-safe features.** Category base rates are estimated on the
  training fold only and frozen before being applied to the test fold
  (Bailey, Borwein, Lopez de Prado, Zhu 2014, *Probability of Backtest
  Overfitting*).
- **Model zoo.** MarketPrior, BaseRate, Logistic, KNN, GBM, plus
  wrappers for isotonic calibration, Bayesian shrinkage, and stacking
  - all fit/predict-compatible under the same `Model` protocol.
- **Walk-forward backtest.** Contiguous time-ordered test folds,
  fractional Kelly sizing, spread + basis-point fees, hit rate, Sharpe,
  Sortino, max drawdown, per-category PnL, ASCII reliability diagram.
- **51 passing tests** covering the DGP, features, every model,
  scoring-rule properties, Kelly sizing, and the CLI.

See [`REFERENCES.md`](REFERENCES.md) for the papers behind every choice
and [`docs/architecture.md`](docs/architecture.md) for the full design
walkthrough.

## Install

```powershell
pip install -e ".[dev]"
```

Requires Python 3.10+. The core dependencies are numpy, pandas, scipy,
and scikit-learn.

## Quickstart

```powershell
aggie-pm run
```

Generates a 2000-event synthetic dataset, runs six-fold walk-forward
cross-validation, trains every model in the zoo, and prints a
leaderboard plus per-model detail block.

On your own CSV (same schema as the synthetic generator):

```powershell
aggie-pm run --csv path/to/markets.csv --out reports/
```

Useful knobs (see `aggie-pm run --help` for the full list):

| flag | default | meaning |
| --- | ---: | --- |
| `--n-events` | 2000 | synthetic dataset size |
| `--seed` | 20260421 | DGP seed |
| `--folds` | 6 | walk-forward folds |
| `--kelly` | 0.25 | Kelly fraction (quarter-Kelly default, Thorp 2006) |
| `--min-edge` | 0.02 | min model-vs-market edge required to place a bet |
| `--max-position` | 0.05 | cap on bankroll fraction per bet |
| `--fee-bps` | 20 | round-trip fee in basis points |
| `--out` | *(none)* | directory to write artifacts |

## Example output

```
========================================================================
  AggieQuant prediction-markets walk-forward backtest
========================================================================
  folds: 5    test events: 900

Leaderboard (sorted by log-loss, lower is better):

                               model  brier  log_loss    ece  n_bets  hit_rate  gross_pnl  final_bankroll  sharpe  sortino  max_drawdown
                        market_prior 0.1318    0.4144 0.0231       0       NaN     0.0000          1.0000     NaN      NaN        0.0000
                            logistic 0.1337    0.4178 0.0137     395    0.5620     0.3289          1.3164  0.8018   1.1655       -0.1772
stack(logistic,knn,gbm,market_prior) 0.1340    0.4253 0.0297     350    0.6429     0.0932          1.0460  0.2560   0.3031       -0.2620
                         shrink(gbm) 0.1351    0.4420 0.0332     393    0.5038     0.0450          1.0048  0.1272   0.1749       -0.3015
                                 knn 0.1432    0.4808 0.0345     584    0.3339    -0.3066          0.6886 -0.5522  -0.9123       -0.5539
                       iso(logistic) 0.1405    0.4993 0.0405     542    0.5923    -0.0569          0.8739 -0.0986  -0.1344       -0.3252
                                 gbm 0.1578    0.5572 0.0433     644    0.4394    -0.2626          0.6996 -0.3774  -0.6000       -0.5383
                           base_rate 0.2461    0.6855 0.0322     854    0.1756    -1.8185          0.1452 -2.0989  -6.4541       -0.8944
```

Reading the table: the market is the hardest benchmark to beat on pure
calibration (Brier 0.1318). The logistic model comes within a hair of
market Brier (0.1337) but translates that marginal edge into a
**Sharpe of 0.80 and +31.6% PnL** after 20 bps fees, because Kelly
sizing concentrates capital on the spots where the edge is clearest.
The base-rate model is calibration-competitive on ECE but gets
destroyed on PnL - the point Gneiting & Raftery (2007) make about
why ECE alone is not a sufficient diagnostic.

## Repo layout

```
.
├── pyproject.toml
├── LICENSE                     # MIT, Dylan Bago 2026
├── README.md
├── REFERENCES.md               # every paper this pipeline leans on
├── .github/workflows/check.yml # pytest + CLI smoke on 3.11/3.12
├── data/
│   └── sample_markets.csv      # 10-row teaching CSV
├── docs/
│   ├── architecture.md         # design walkthrough
│   ├── results.md              # how to read the leaderboard
│   └── handoff.md              # AggieQuant pod handoff
├── notebooks/
│   └── 01_walkthrough.py       # step-by-step pipeline demo
├── src/
│   ├── prediction_markets.py   # original stdlib-only teaching script
│   └── aggie_pm/
│       ├── __init__.py
│       ├── __main__.py         # `python -m aggie_pm`
│       ├── cli.py              # `aggie-pm run`
│       ├── data.py             # synthetic DGP + CSV loader
│       ├── features.py         # feature engineering
│       ├── models.py           # the model zoo
│       ├── backtest.py         # walk-forward + Kelly + metrics
│       └── report.py           # console + CSV artifacts
└── tests/
    ├── conftest.py
    ├── test_data.py
    ├── test_features.py
    ├── test_models.py
    ├── test_backtest.py
    └── test_cli.py
```

## Running the tests

```powershell
pytest
```

51 tests covering DGP determinism, leak-safe feature engineering,
model fit/predict contract, scoring-rule properties (Brier hits zero on
perfect forecasts, log loss is a proper scoring rule, ECE vanishes for
calibrated forecasts), Kelly sizing corner cases, walk-forward
correctness, and the CLI.

## Related AggieQuant work

- `aggiequant-pod-starters` (the monorepo this was extracted from).
- The stdlib-only teaching script `src/prediction_markets.py` is
  preserved verbatim so incoming members can read it first before
  touching the full pipeline.

## License

MIT. See [`LICENSE`](LICENSE).
