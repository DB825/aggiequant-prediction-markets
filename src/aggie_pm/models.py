"""Model zoo for the prediction-markets pipeline.

Every model takes a ``FeatureMatrix`` for ``fit`` and ``predict`` and
returns calibrated probabilities on [0, 1]. The zoo is deliberately
heterogeneous so the club's leaderboard shows a real lineup of
techniques, not one hyper-tuned black box:

- ``MarketPriorModel``: return the market price unchanged. The honest
  baseline. If you can't beat this you haven't beaten the market.
- ``BaseRateModel``: per-category historical YES rate. The dumbest
  non-market model. Gneiting & Raftery (2007) use this as the reference
  forecast for proper scoring rules.
- ``LogisticModel``: L2-regularised logistic regression on standardised
  features. Linear in the logit; maximally interpretable.
- ``KNNModel``: k-NN on standardised features with distance weighting.
  Nonparametric, catches local non-linearities the logistic model can't.
- ``GradientBoostingModel``: scikit-learn HistGradientBoostingClassifier
  with isotonic post-hoc calibration. Strong baseline for tabular data
  (Shwartz-Ziv & Armon 2022, "Tabular data: Deep learning is not all
  you need").
- ``IsotonicCalibratedModel``: wraps any base model with sklearn's
  isotonic regression - a proper scoring-rule-consistent calibrator
  (Niculescu-Mizil & Caruana 2005).
- ``StackedEnsemble``: logistic-regression stacker over base-model
  predictions. Mirrors Netflix-prize / Kaggle blending practice.
- ``BayesianShrinkageModel``: shrink a model's prediction toward the
  market price with precision-weighted posterior combination. Lets an
  over-confident model defer to market where it's uncertain (Clemen &
  Winkler 1999 on combining forecasts).

All models are deterministic given a seed. None of them see future data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .features import FeatureMatrix


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Model(Protocol):
    name: str

    def fit(self, fm: FeatureMatrix) -> "Model": ...

    def predict(self, fm: FeatureMatrix) -> np.ndarray: ...


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@dataclass
class MarketPriorModel:
    """Return the market's own price. The baseline to beat."""

    name: str = "market_prior"

    def fit(self, fm: FeatureMatrix) -> "MarketPriorModel":
        return self

    def predict(self, fm: FeatureMatrix) -> np.ndarray:
        return _clip(fm.market_prob.copy())


@dataclass
class BaseRateModel:
    """Per-category historical YES rate, smoothed toward 0.5."""

    name: str = "base_rate"
    pseudo_count: int = 20
    _rates: dict[str, float] = field(default_factory=dict)

    def fit(self, fm: FeatureMatrix) -> "BaseRateModel":
        cats = np.unique(fm.category)
        rates: dict[str, float] = {}
        for c in cats:
            mask = fm.category == c
            n = int(mask.sum())
            raw = float(fm.y[mask].mean()) if n > 0 else 0.5
            rates[c] = (raw * n + 0.5 * self.pseudo_count) / (n + self.pseudo_count)
        self._rates = rates
        return self

    def predict(self, fm: FeatureMatrix) -> np.ndarray:
        return _clip(
            np.array([self._rates.get(c, 0.5) for c in fm.category], dtype=float)
        )


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------


@dataclass
class LogisticModel:
    """L2-regularised logistic regression with StandardScaler pre-processing."""

    name: str = "logistic"
    C: float = 1.0
    _scaler: StandardScaler | None = None
    _clf: LogisticRegression | None = None

    def fit(self, fm: FeatureMatrix) -> "LogisticModel":
        self._scaler = StandardScaler().fit(fm.X)
        Xs = self._scaler.transform(fm.X)
        self._clf = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=500, random_state=0
        ).fit(Xs, fm.y)
        return self

    def predict(self, fm: FeatureMatrix) -> np.ndarray:
        assert self._scaler is not None and self._clf is not None
        Xs = self._scaler.transform(fm.X)
        return _clip(self._clf.predict_proba(Xs)[:, 1])


@dataclass
class KNNModel:
    """Distance-weighted k-nearest-neighbours classifier."""

    name: str = "knn"
    k: int = 25
    _scaler: StandardScaler | None = None
    _clf: KNeighborsClassifier | None = None

    def fit(self, fm: FeatureMatrix) -> "KNNModel":
        self._scaler = StandardScaler().fit(fm.X)
        Xs = self._scaler.transform(fm.X)
        k_eff = max(3, min(self.k, max(len(fm.y) // 2, 3)))
        self._clf = KNeighborsClassifier(
            n_neighbors=k_eff, weights="distance", algorithm="auto"
        ).fit(Xs, fm.y)
        return self

    def predict(self, fm: FeatureMatrix) -> np.ndarray:
        assert self._scaler is not None and self._clf is not None
        Xs = self._scaler.transform(fm.X)
        return _clip(self._clf.predict_proba(Xs)[:, 1])


@dataclass
class GradientBoostingModel:
    """HistGradientBoostingClassifier with isotonic post-hoc calibration.

    We reserve the last 20 percent of the training window as a calibration
    fold so the isotonic regression doesn't memorise the training set.
    This is the standard recipe from Niculescu-Mizil & Caruana (2005).
    """

    name: str = "gbm"
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    cal_frac: float = 0.2
    _clf: HistGradientBoostingClassifier | None = None
    _iso: IsotonicRegression | None = None

    def fit(self, fm: FeatureMatrix) -> "GradientBoostingModel":
        n = len(fm.y)
        cut = max(int(n * (1 - self.cal_frac)), 2)
        X_fit, y_fit = fm.X[:cut], fm.y[:cut]
        X_cal, y_cal = fm.X[cut:], fm.y[cut:]
        self._clf = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=0,
        ).fit(X_fit, y_fit)
        if len(y_cal) >= 20 and len(np.unique(y_cal)) == 2:
            raw = self._clf.predict_proba(X_cal)[:, 1]
            self._iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y_cal)
        else:
            self._iso = None
        return self

    def predict(self, fm: FeatureMatrix) -> np.ndarray:
        assert self._clf is not None
        raw = self._clf.predict_proba(fm.X)[:, 1]
        if self._iso is not None:
            raw = self._iso.predict(raw)
        return _clip(raw)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


@dataclass
class IsotonicCalibratedModel:
    """Wrap any base ``Model`` and calibrate its output with isotonic regression."""

    base: Model
    cal_frac: float = 0.25
    name: str = "iso_cal"
    _iso: IsotonicRegression | None = None

    def __post_init__(self) -> None:
        self.name = f"iso({self.base.name})"

    def fit(self, fm: FeatureMatrix) -> "IsotonicCalibratedModel":
        n = len(fm.y)
        cut = max(int(n * (1 - self.cal_frac)), 2)
        fm_fit = FeatureMatrix(
            X=fm.X[:cut],
            y=fm.y[:cut],
            market_prob=fm.market_prob[:cut],
            market_spread=fm.market_spread[:cut],
            category=fm.category[:cut],
            feature_names=fm.feature_names,
        )
        fm_cal = FeatureMatrix(
            X=fm.X[cut:],
            y=fm.y[cut:],
            market_prob=fm.market_prob[cut:],
            market_spread=fm.market_spread[cut:],
            category=fm.category[cut:],
            feature_names=fm.feature_names,
        )
        self.base.fit(fm_fit)
        if len(fm_cal.y) >= 20 and len(np.unique(fm_cal.y)) == 2:
            raw = self.base.predict(fm_cal)
            self._iso = IsotonicRegression(out_of_bounds="clip").fit(raw, fm_cal.y)
        else:
            self._iso = None
        # Re-fit base on the full training window now that isotonic is frozen.
        self.base.fit(fm)
        return self

    def predict(self, fm: FeatureMatrix) -> np.ndarray:
        raw = self.base.predict(fm)
        if self._iso is not None:
            raw = self._iso.predict(raw)
        return _clip(raw)


@dataclass
class BayesianShrinkageModel:
    """Precision-weighted combination of a base model and the market price.

    If the base model's in-sample log-loss is ``L_m`` and the market's is
    ``L_k``, both on the same training fold, we treat each forecast as a
    noisy observation of the true logit. On the logit scale we combine:

        l_combined = (w_m * l_model + w_k * l_market) / (w_m + w_k)

    where ``w_m = 1 / var(model logit errors)`` and analogously for the
    market. This is the natural Gaussian posterior-combination rule
    (Clemen & Winkler 1999 review "Combining probability distributions").

    The shrinkage is strongest where the model is uncertain - exactly when
    you want to defer to the market - and weakest where the model has
    reliably beaten the market in-sample.
    """

    base: Model
    name: str = "shrink"
    _w_model: float = 1.0
    _w_market: float = 1.0

    def __post_init__(self) -> None:
        self.name = f"shrink({self.base.name})"

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def fit(self, fm: FeatureMatrix) -> "BayesianShrinkageModel":
        self.base.fit(fm)
        # In-sample residuals on the logit scale, vs resolved outcomes treated
        # as the empirical truth via a thin Bayesian smoother.
        pm = self.base.predict(fm)
        pk = fm.market_prob
        # Turn binary y into a soft target on the logit scale using Laplace smoothing.
        y_soft = (fm.y * 0.98 + 0.01)
        lm_err = self._logit(pm) - self._logit(y_soft)
        lk_err = self._logit(pk) - self._logit(y_soft)
        var_m = max(float(np.var(lm_err)), 1e-3)
        var_k = max(float(np.var(lk_err)), 1e-3)
        self._w_model = 1.0 / var_m
        self._w_market = 1.0 / var_k
        return self

    def predict(self, fm: FeatureMatrix) -> np.ndarray:
        lm = self._logit(self.base.predict(fm))
        lk = self._logit(fm.market_prob)
        lc = (self._w_model * lm + self._w_market * lk) / (self._w_model + self._w_market)
        return _clip(1.0 / (1.0 + np.exp(-lc)))


@dataclass
class StackedEnsemble:
    """Logistic-regression stacker over base models.

    Members are fit on the early training fold; the stacker is fit on their
    out-of-fold predictions on the late training fold. This avoids target
    leakage into the meta-learner (Wolpert 1992, "Stacked generalization").
    """

    members: list[Model]
    name: str = "stack"
    cal_frac: float = 0.3
    _meta: LogisticRegression | None = None

    def __post_init__(self) -> None:
        self.name = "stack(" + ",".join(m.name for m in self.members) + ")"

    def fit(self, fm: FeatureMatrix) -> "StackedEnsemble":
        n = len(fm.y)
        cut = max(int(n * (1 - self.cal_frac)), 2)
        fm_fit = FeatureMatrix(
            X=fm.X[:cut],
            y=fm.y[:cut],
            market_prob=fm.market_prob[:cut],
            market_spread=fm.market_spread[:cut],
            category=fm.category[:cut],
            feature_names=fm.feature_names,
        )
        fm_meta = FeatureMatrix(
            X=fm.X[cut:],
            y=fm.y[cut:],
            market_prob=fm.market_prob[cut:],
            market_spread=fm.market_spread[cut:],
            category=fm.category[cut:],
            feature_names=fm.feature_names,
        )
        for m in self.members:
            m.fit(fm_fit)
        Z = np.column_stack([m.predict(fm_meta) for m in self.members])
        # Meta-learner: logistic regression on logits of member predictions
        Zl = np.log(np.clip(Z, 1e-6, 1 - 1e-6) / (1 - np.clip(Z, 1e-6, 1 - 1e-6)))
        self._meta = LogisticRegression(C=1.0, max_iter=500, random_state=0).fit(
            Zl, fm_meta.y
        )
        # Re-fit members on the full training window so prediction time uses
        # all available training data.
        for m in self.members:
            m.fit(fm)
        return self

    def predict(self, fm: FeatureMatrix) -> np.ndarray:
        assert self._meta is not None
        Z = np.column_stack([m.predict(fm) for m in self.members])
        Zl = np.log(np.clip(Z, 1e-6, 1 - 1e-6) / (1 - np.clip(Z, 1e-6, 1 - 1e-6)))
        return _clip(self._meta.predict_proba(Zl)[:, 1])


# ---------------------------------------------------------------------------
# Zoo factory
# ---------------------------------------------------------------------------


def train_model_zoo(fm: FeatureMatrix) -> list[Model]:
    """Return a list of fitted models. Order is stable for reporting."""

    base_logistic = LogisticModel()
    base_knn = KNNModel()
    base_gbm = GradientBoostingModel()

    market = MarketPriorModel()
    base_rate = BaseRateModel().fit(fm)
    logistic = LogisticModel().fit(fm)
    knn = KNNModel().fit(fm)
    gbm = GradientBoostingModel().fit(fm)
    iso_logistic = IsotonicCalibratedModel(base=LogisticModel()).fit(fm)
    shrink_gbm = BayesianShrinkageModel(base=GradientBoostingModel()).fit(fm)
    stack = StackedEnsemble(
        members=[LogisticModel(), KNNModel(), GradientBoostingModel(), MarketPriorModel()]
    ).fit(fm)

    # market doesn't need fitting
    market.fit(fm)

    return [market, base_rate, logistic, knn, gbm, iso_logistic, shrink_gbm, stack]
