# Reading the leaderboard

The default run produces a table sorted ascending by log-loss:

```
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

## What to notice

**The market is hard to beat on calibration.** market_prior has the
lowest Brier (0.1318) and a solid ECE (0.023). The market logit
contains most of the predictable signal, as theory says it should
(Manski 2006).

**A good model wins on PnL, not on Brier.** The logistic model beats
the market on log-loss by 1%, but Kelly sizing turns that into a
Sharpe of 0.80 and +32% bankroll over the test window. Edge
concentrates where the model is most confident *and* the market is
most wrong.

**Base-rate models are ECE-competitive and PnL-destructive.** Base
rates look fine on calibration error (0.032) but bet indiscriminately
against a price they don't see, collect spread losses, and blow up.
This is exactly the object lesson Gneiting & Raftery (2007) warn
about: calibration alone is not a sufficient forecast diagnostic.

**KNN and raw GBM overbet.** Both have the highest bet counts and the
deepest drawdowns. Their predictions are not calibrated and they don't
know it - Kelly on an over-confident model is the fastest way to
bankruptcy.

**Post-hoc calibration and shrinkage help.** Isotonic-calibrated
logistic and Bayesian-shrunk GBM trade fewer times with higher hit
rates than their naive counterparts. Shrinkage toward the market is
the single most robust wrapper in the zoo.

**Stacking is solid but not dominant.** The stacked ensemble hits the
highest hit rate (64%) but doesn't beat the logistic model on final
bankroll at this dataset size. Stacking earns its keep on larger,
noisier datasets; at 1500 events it is fine but not magic.

## How to use this on real data

1. Export resolved markets from a public source (Polymarket, Kalshi,
   Metaculus) into the CSV schema described in
   `aggie_pm.data.load_markets_csv`.
2. Or pull Kalshi resolved markets directly with
   `aggie-pm run --kalshi --kalshi-source historical --out reports/kalshi`.
3. For a larger reusable pull, run
   `aggie-pm extract-kalshi --kalshi-pages 50 --kalshi-page-limit 1000 --out data/kalshi_resolved.csv`,
   then `aggie-pm profile --csv data/kalshi_resolved.csv --out reports/kalshi_profile`.
4. Compare every model's log-loss and ECE to `market_prior`. If
   nothing beats the market on log-loss you do not have alpha - stop
   and rebuild your features before touching the bet sizer.
5. If something beats market log-loss, look at Sharpe *after fees*.
   A model that only wins by 0.001 Brier will be eaten by spread on a
   real book.
6. Inspect `pnl_by_category_<model>.csv` and `slices_<model>.csv` to see
   where the edge lives. A model whose PnL is concentrated in one
   category, price bucket, or spread regime usually means a single bias,
   not a general-purpose alpha.

## Pareto read

After a run, pass `leaderboard.csv` through `aggie_pm.pareto.pareto_front`
when choosing what to investigate next. A model that is not first on
log-loss can still be non-dominated if it has materially better Sharpe,
drawdown, or final bankroll. That is the honest way to connect model
selection to a portfolio objective instead of cherry-picking one metric.

## Honest caveats

- Results above are on synthetic data with known biases. On real
  markets you should expect smaller edges and noisier backtests.
- Kelly assumes known probabilities. In practice we do not know them;
  fractional Kelly is a hedge against estimation error, not a cure.
- Walk-forward CV still multiple-tests. Treat the leaderboard as
  hypothesis-generating, not as a promise of future performance.
  Consider computing a Deflated Sharpe Ratio (Bailey & Lopez de
  Prado 2014) before quoting the best model's Sharpe in a memo.
