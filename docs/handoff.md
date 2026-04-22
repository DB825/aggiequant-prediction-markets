# [Prediction Markets] Handoff - Spring 2026

## Identity

- Pod name: Prediction Markets
- Pod type: sprint
- Semester: Spring 2026
- Pod lead: Dylan Bago
- Members: Dylan Bago
- Date written: 2026-04-21

## One-paragraph summary

The pod shipped `aggiequant-prediction-markets`, a pip-installable
Python package that generates a realistic synthetic prediction-market
dataset, can ingest resolved Kalshi markets, engineers leak-safe features,
trains a zoo of eight models
(market prior, base rate, logistic regression, k-NN, gradient boosting,
isotonic-calibrated wrapper, Bayesian shrinkage, stacked ensemble),
and runs a walk-forward backtest with fractional Kelly sizing that
reports Brier, log loss, ECE, AUC, Sharpe, Sortino, max drawdown,
market-relative improvement, bet coverage, turnover, profit factor,
per-category PnL, and slice-level diagnostics. The package is MIT-licensed,
has a pytest suite and CI on Python 3.11/3.12, and is the archive-citable
deliverable for this semester.

## What shipped

- Deliverable: `aggiequant-prediction-markets` repository.
- Location in repo: standalone; extracted from
  `aggiequant-pod-starters/prediction-markets/`.
- Demo / presentation: `aggie-pm run` prints the full leaderboard;
  see `docs/results.md` for commentary.

## What we started from (sprint only)

- Starting question: "Are our probability forecasts better calibrated
  than the market prices we collected, and does that calibration
  translate into positive risk-adjusted PnL after spread and fees?"
- Why this question was picked: the pod brief asked for a deliverable
  that combined proper-scoring-rule discipline with something that
  looked enough like trading to be legible to the institutional track.
- Prior work reviewed: the stdlib-only teaching script at
  `src/prediction_markets.py` (preserved in this repo verbatim);
  `aggiequant-pod-starters/docs/datasets-and-papers.md`; Gneiting &
  Raftery 2007; Wolfers & Zitzewitz 2004; Bailey et al. 2014.

## Timeline of attempts

1. Week 1: reviewed the stdlib starter and confirmed it was the right
   teaching entry point but could not demonstrate a full pipeline on
   its own. Decision: keep it untouched, build an `aggie_pm` package
   next to it.
2. Week 2: built the synthetic DGP. First draft had no per-category
   bias, which meant the market was unbeatable in expectation; added
   category biases matching the Wolfers & Zitzewitz catalogue.
3. Week 3: wrote `features.py`. Initial version leaked test-set category
   rates into training by accident; fixed by making base-rate
   computation explicit and freezing it across folds.
4. Week 4: built the model zoo. Isotonic calibration and Bayesian
   shrinkage were the two wrappers that moved metrics the most.
5. Week 5: wrote the walk-forward backtester and Kelly sizer. First
   version compounded wrong (used PnL on stake, not on bankroll); fixed.
6. Week 6: wrote report, CSV dump, CLI, README, tests. Locked the
   initial pytest suite + CI.
7. Follow-up: added Kalshi historical/live resolved-market ingestion,
   Prosperity-style microstructure features, optional sentiment hooks,
   Pareto-front model selection, bulk Kalshi extraction, dataset profiling,
   and richer slice metrics for larger datasets.

## Current state of the artifact

- Runs end-to-end with `aggie-pm run` after `pip install -e ".[dev]"`.
- Tests: pytest suite on Python 3.11/3.12.
- Documentation: `README.md`, `docs/architecture.md`,
  `docs/results.md`, `REFERENCES.md`.
- CI: `.github/workflows/check.yml` runs pytest + a CLI smoke run.
- Known fragile spots:
  - The synthetic DGP assumes a stationary bias per category. Real
    markets don't.
  - The reliability diagram in `report.py` is ASCII sparkline, not a
    real plot.

## Failed experiments (worth keeping)

- **Equal-weight ensemble.** Tried averaging model probabilities
  before building the stacked logistic. Result: worse than any
  component on log-loss. Why: unequal model quality means equal weights
  is a bad prior; logit-logistic stacking dominates.
- **XGBoost instead of HistGradientBoosting.** Considered for speed,
  rejected for dependency weight - sklearn's histogram booster is
  competitive on tabular data of this size (Shwartz-Ziv & Armon 2022).
- **Random-fold CV.** Early version used `sklearn.KFold`. Caught before
  merge: shuffling leaks future market prices into training windows.
  Replaced with contiguous walk-forward.
- **Full Kelly.** Tried `kelly_fraction=1.0`. Resulted in a handful of
  80% bankroll bets and catastrophic drawdowns. Quarter Kelly gave the
  best Sharpe per unit of drawdown.

## Known broken things

- ASCII reliability diagram is decorative; replace with matplotlib for
  a real memo.
- Kalshi support currently uses a single market snapshot per row. The
  stronger research version should pull fixed-horizon candlesticks
  before settlement.
- Deflated Sharpe Ratio is not computed; the README warns about this.

## Open questions for next cohort

1. Does the stacked ensemble beat the logistic model on a larger
   dataset (>10k events), or does the extra variance from the
   meta-learner still cost it?
2. Does the per-category bias the DGP encodes survive replacement with
   real Kalshi or Polymarket resolved-event data?
3. What is the out-of-fold Deflated Sharpe of the leaderboard's top
   model after correcting for the number of models tested?
4. Can a simple recurrent feature (momentum of recent same-category
   resolutions) beat the engineered `feat_momentum` stand-in?
5. Where does this model fail most badly - is it a specific category,
   a specific spread regime, or short-dated markets near resolution?

## First-week playbook for next cohort

1. `git clone` the repo, `pip install -e ".[dev]"`, run `pytest`, then
   `aggie-pm run --out reports/` and read `reports/summary.txt`.
2. Read `docs/architecture.md` once end-to-end. Then read
   `docs/results.md` with the generated leaderboard open in another
   window.
3. Pick one open question from the list above and write a one-page
   research plan for how you would answer it with this codebase.
4. Run `aggie-pm extract-kalshi --kalshi-pages 50 --kalshi-page-limit 1000 --out data/kalshi_resolved.csv`
   and inspect `reports/kalshi_profile` after profiling it with
   `aggie-pm profile --csv data/kalshi_resolved.csv --out reports/kalshi_profile`.
5. Run the full pipeline on that real data with
   `aggie-pm run --csv data/kalshi_resolved.csv --out reports/kalshi_large`.
   If no model beats
   `market_prior` on log-loss, stop and diagnose features before
   adding anything else.

## What I would have done differently

Would have built the real-data loader in week 2 instead of after the
synthetic pipeline. Synthetic data is excellent for unit tests, but
real exchange data should shape feature design earlier.

## Files, links, and pointers

- Research log: commit history on `main`
- Data: `data/sample_markets.csv` (10 rows, teaching only)
- Deliverable: this repository
- Slides / writeup: `docs/results.md`
- External papers: see `REFERENCES.md`

## Specialization tie-in (if any)

Low direct tie-in to commodities/energy specialization. The closest
link is that the weather and macro categories in the synthetic DGP
mirror the kind of event contracts weather-trading desks (Kalshi
hurricane markets, CME HDD/CDD futures) care about. A specialization
extension would replace the synthetic weather category with real
NOAA-settled event contracts.

## Archivist checklist

- [ ] Template sections all filled (not left blank)
- [ ] Failed experiments documented with enough detail to be useful
- [ ] First-week playbook is specific, not generic
- [ ] Deliverable linked and reachable
- [ ] Citable per `deliverable-rubric.md`

Archivist sign-off: [pending]
