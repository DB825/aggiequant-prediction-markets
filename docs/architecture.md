# Architecture

This document walks through every module in `aggie_pm` and explains why
it exists, what it promises to callers, and which paper or practitioner
text backs the design. If you are reading the code for the first time,
read this file alongside it.

## Pipeline overview

```
          ┌──────────────┐   ┌─────────────┐   ┌────────────────┐
 dataset  │    data.py   │──▶│ features.py │──▶│   models.py    │──▶ predictions
          │ synthetic or │   │ leak-safe   │   │ zoo w/ wrappers│
          │ real CSV     │   │ features    │   │ + ensembles    │
          └──────────────┘   └─────────────┘   └────────────────┘
                                                      │
                                                      ▼
                                               ┌────────────────┐
                                               │  backtest.py   │
                                               │ walk-forward + │
                                               │ Kelly + PnL    │
                                               └────────────────┘
                                                      │
                                                      ▼
                                               ┌────────────────┐
                                               │   report.py    │
                                               │ leaderboard +  │
                                               │ CSV artifacts  │
                                               └────────────────┘
```

## `data.py` - the honest DGP

The point of the synthetic generator is to give the pipeline a dataset
where *there is something to learn but it is not trivial*:

- A ground-truth probability exists for every event and is a smooth
  function of observable features.
- The market price is a noisy, systematically biased observation of
  that true probability. The biases match the literature on real
  markets (favourite-longshot bias in sports; retail over-enthusiasm
  in crypto; under-pricing of status-quo outcomes in macro/policy -
  see Wolfers & Zitzewitz 2004).
- Resolutions `y` are Bernoulli draws from the true probability, which
  sets an irreducible floor on Brier score equal to `E[p(1-p)]`.

Why build a synthetic DGP at all? Because without ground truth we can
only score a model against the market; with it we can *also* audit
whether the pipeline is recovering the real signal. That is the
difference between "our model beats the market by 0.002 Brier, maybe"
and "we can show exactly which biases our model picked up and by how
much."

## `features.py` - spine + signals

The market logit is by far the strongest single feature in a roughly
efficient market (Manski 2006). Every feature vector starts from the
market logit and augments it with things a research team could
credibly assemble:

- raw price and logit
- category one-hot (to absorb per-category bias)
- smoothed category base rate, **computed on the training fold only**
  and passed into the test fold unchanged (Bailey et al. 2014)
- time-to-resolution, normalised
- book spread as a liquidity / disagreement proxy
- three engineered latent features (`feat_signal`, `feat_momentum`,
  `feat_dispersion`) that stand in for things like economic-surprise
  indices, polling spreads, Elo differentials, implied vol
- two interactions: `market_logit × time_to_resolve` and
  `market_logit × spread`, so the model can learn "the bias is biggest
  in crypto near resolution" without a separate model per category

The `FeatureMatrix` dataclass carries everything the backtest needs:
design matrix, outcomes, market price, spread, category, and column
names.

## `models.py` - the zoo

Every model implements a minimal `Model` protocol: `fit(FeatureMatrix)
-> Model` and `predict(FeatureMatrix) -> np.ndarray[float]` returning
probabilities in (0, 1). This lets the backtest treat them
uniformly.

| model | why it's in the zoo |
| --- | --- |
| `MarketPriorModel` | the benchmark to beat; passes the market price through untouched |
| `BaseRateModel` | per-category smoothed YES rate; Gneiting & Raftery (2007) reference forecast |
| `LogisticModel` | L2 logistic regression on standardised features; maximally interpretable |
| `KNNModel` | distance-weighted k-NN; catches local non-linearities the linear model misses |
| `GradientBoostingModel` | `HistGradientBoostingClassifier` with isotonic post-hoc calibration on a held-out fold (Niculescu-Mizil & Caruana 2005) |
| `IsotonicCalibratedModel` | wraps any base model and calibrates it with isotonic regression |
| `BayesianShrinkageModel` | precision-weighted logit combination with the market (Clemen & Winkler 1999) |
| `StackedEnsemble` | logistic meta-learner on out-of-fold member predictions (Wolpert 1992) |

All wrappers re-fit their base model on the full training window after
freezing the calibrator / stacker, so prediction time uses every
available training row.

## `backtest.py` - walk-forward + Kelly

Time order is the only honest way to evaluate a market model. We
reserve the first `min_train_frac` of events as the initial training
window and split the remainder into `n_folds` contiguous test windows.
At each fold the pipeline fits a fresh model zoo on all data that
closed before the fold opened, predicts on the fold, and records
scores + simulated trades.

### Bet sizing

For a binary market priced at `q`:

```
YES: edge = p_model - q_ask,    kelly_yes = edge / (1 - q_ask)
NO : edge = (1 - q_bid) - (1-p), kelly_no  = edge / (1 - q_no_ask)
```

We scale by `kelly_fraction` (default 0.25 - quarter Kelly, Thorp 2006)
and cap each bet at `max_position` of bankroll. Bankroll compounds
across events. Round-trip fees (`fee_bps`) are subtracted from payoff.

### Reported metrics

- Brier score, log loss, expected calibration error (ECE)
- n_bets, hit rate, gross PnL, final bankroll
- Sharpe and Sortino on per-event returns, annualised to 252 bets
- Max drawdown of the bankroll path
- Per-category PnL breakdown
- Reliability-diagram bin data

## `report.py` - human + machine outputs

The console report is a leaderboard sorted by log-loss plus a per-model
detail block with the metrics above and a small ASCII reliability
diagram. The `--out` flag also writes:

- `summary.txt` - the human report
- `leaderboard.csv` - all models, all metrics
- `calibration_<model>.csv` - bins, n, avg pred, empirical rate
- `pnl_by_category_<model>.csv` - PnL per category
- `bets_<model>.csv` - one row per event with side, stake, PnL

Those CSVs are intentionally small and flat so they are trivial to
drop into a dashboard or a Jupyter notebook.

## What this pipeline deliberately does not do

- It does not talk to a real exchange. Kalshi / Polymarket loaders
  belong next to `load_markets_csv`, not next to the Kelly sizer.
- It does not claim live tradability. Slippage, latency, position
  limits, and book depth are out of scope; the "tradable edge" the
  README claims is a simulation with conservative fees, not an
  investable backtest.
- It does not hyper-optimise any model. The point is to show a
  *lineup* of techniques, not to beat any one of them into the ground.
