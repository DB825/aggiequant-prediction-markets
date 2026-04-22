"""Model-zoo tests."""

from __future__ import annotations

import numpy as np
import pytest

from aggie_pm.data import generate_synthetic_markets
from aggie_pm.features import build_features
from aggie_pm.models import (
    BaseRateModel,
    BayesianShrinkageModel,
    GradientBoostingModel,
    IsotonicCalibratedModel,
    KNNModel,
    LogisticModel,
    MarketPriorModel,
    StackedEnsemble,
    train_model_zoo,
)


@pytest.fixture(scope="module")
def split():
    df = generate_synthetic_markets(n_events=800, seed=101)
    fm_train, cat_rates = build_features(df.iloc[:600])
    fm_test, _ = build_features(df.iloc[600:], category_base_rates=cat_rates)
    return fm_train, fm_test


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MarketPriorModel(),
        lambda: BaseRateModel(),
        lambda: LogisticModel(),
        lambda: KNNModel(),
        lambda: GradientBoostingModel(),
        lambda: IsotonicCalibratedModel(base=LogisticModel()),
        lambda: BayesianShrinkageModel(base=LogisticModel()),
        lambda: StackedEnsemble(
            members=[LogisticModel(), KNNModel(), MarketPriorModel()]
        ),
    ],
    ids=[
        "market_prior",
        "base_rate",
        "logistic",
        "knn",
        "gbm",
        "iso_logistic",
        "shrink_logistic",
        "stacked",
    ],
)
def test_models_fit_predict_shape_and_range(split, factory):
    fm_train, fm_test = split
    m = factory().fit(fm_train)
    p = m.predict(fm_test)
    assert p.shape == fm_test.y.shape
    assert (p > 0).all() and (p < 1).all()


def test_market_prior_returns_market(split):
    _, fm_test = split
    p = MarketPriorModel().fit(fm_test).predict(fm_test)
    np.testing.assert_allclose(p, np.clip(fm_test.market_prob, 1e-6, 1 - 1e-6))


def test_logistic_beats_base_rate_out_of_sample(split):
    """A linear model with market-logit as a feature must beat per-category
    base rates. If this test fails, feature engineering is broken."""
    from aggie_pm.backtest import log_loss

    fm_train, fm_test = split
    p_log = LogisticModel().fit(fm_train).predict(fm_test)
    p_base = BaseRateModel().fit(fm_train).predict(fm_test)
    assert log_loss(p_log, fm_test.y) < log_loss(p_base, fm_test.y)


def test_shrinkage_lies_between_base_and_market(split):
    """Bayesian shrinkage predictions should sit between the base model and
    the market on the logit scale (weighted average, both weights > 0)."""
    fm_train, fm_test = split
    base = LogisticModel().fit(fm_train)
    shrink = BayesianShrinkageModel(base=LogisticModel()).fit(fm_train)

    def lg(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    l_base = lg(base.predict(fm_test))
    l_mkt = lg(fm_test.market_prob)
    l_shr = lg(shrink.predict(fm_test))

    lo = np.minimum(l_base, l_mkt)
    hi = np.maximum(l_base, l_mkt)
    # Allow tiny numerical slack
    assert (l_shr >= lo - 1e-6).all()
    assert (l_shr <= hi + 1e-6).all()


def test_train_model_zoo_returns_all_models(split):
    fm_train, _ = split
    zoo = train_model_zoo(fm_train)
    names = {m.name for m in zoo}
    for expected in ("market_prior", "base_rate", "logistic", "knn", "gbm"):
        assert expected in names
