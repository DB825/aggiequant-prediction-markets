# References

Every design choice in the `aggie_pm` package traces back to a paper or
practitioner text. This file collects them so members can go deeper.

## Proper scoring rules and calibration

- Brier, G. W. (1950). *Verification of Forecasts Expressed in Terms of
  Probability.* Monthly Weather Review. The original Brier score.
- Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules,
  Prediction, and Estimation.* JASA 102(477). The canonical reference
  for log loss, Brier score, and calibration.
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting good
  probabilities with supervised learning.* ICML. Isotonic vs Platt
  calibration, with held-out fit.
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). *On
  Calibration of Modern Neural Networks.* ICML. Expected Calibration
  Error and why miscalibration hides inside accurate models. arXiv:
  [1706.04599](https://arxiv.org/abs/1706.04599).

## Combining forecasts

- Clemen, R. T. & Winkler, R. L. (1999). *Combining probability
  distributions from experts in risk analysis.* Risk Analysis 19(2).
  Basis for `BayesianShrinkageModel`.
- Wolpert, D. H. (1992). *Stacked Generalization.* Neural Networks 5(2).
  Basis for `StackedEnsemble`.
- Satopaa, V. A. et al. (2014). *Combining multiple probability
  predictions using a simple logit model.* International Journal of
  Forecasting 30(2). Why logit-averaging beats naive averaging.

## Prediction-market structure

- Manski, C. F. (2006). *Interpreting the predictions of prediction
  markets.* Economics Letters 91(3). Why market prices approximate
  probabilities but not perfectly.
- Wolfers, J. & Zitzewitz, E. (2004). *Prediction markets.* Journal of
  Economic Perspectives 18(2). Favourite-longshot bias and empirical
  performance of real markets.
- Snowberg, E. & Wolfers, J. (2010). *Explaining the Favorite-Longshot
  Bias: Is it Risk-Love or Misperceptions?* JPE 118(4).

## Tabular ML

- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient
  Boosting Machine.* Annals of Statistics 29(5). Foundation of GBDT.
- Ke, G. et al. (2017). *LightGBM.* NeurIPS. The histogram-boosting
  algorithm that scikit-learn's `HistGradientBoostingClassifier`
  follows.
- Shwartz-Ziv, R. & Armon, A. (2022). *Tabular Data: Deep Learning is
  Not All You Need.* Information Fusion 81. arXiv:
  [2106.03253](https://arxiv.org/abs/2106.03253). Why GBDTs remain the
  default for tabular problems.

## Bet sizing

- Kelly, J. L. (1956). *A New Interpretation of Information Rate.* Bell
  System Technical Journal 35(4).
- Thorp, E. O. (2006). *The Kelly Criterion in Blackjack, Sports
  Betting, and the Stock Market.* Handbook of Asset and Liability
  Management Vol. 1. Ch. on fractional Kelly.
- MacLean, L. C., Thorp, E. O. & Ziemba, W. T. (2010). *Good and bad
  properties of the Kelly criterion.* Quantitative Finance 10(7).

## Backtesting and overfitting

- Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* FAJ 58(4).
- White, H. (2000). *A Reality Check for Data Snooping.* Econometrica
  68(5).
- Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2014).
  *The Probability of Backtest Overfitting.* JCF.
  [SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253).
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.*
  Wiley. Walk-forward CV, purged k-fold, deflated Sharpe.

## Prediction-market research and datasets

- Kalshi API docs: <https://docs.kalshi.com/>
- Polymarket API docs: <https://docs.polymarket.com/api-reference>
- Metaculus API: <https://www.metaculus.com/notebooks/15141/officially-launching-the-metaculus-api/>
