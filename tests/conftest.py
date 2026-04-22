"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from aggie_pm.data import generate_synthetic_markets
from aggie_pm.features import build_features


@pytest.fixture(scope="session")
def small_df():
    return generate_synthetic_markets(n_events=400, seed=42)


@pytest.fixture(scope="session")
def medium_df():
    return generate_synthetic_markets(n_events=1200, seed=7)


@pytest.fixture(scope="session")
def fm_small(small_df):
    fm, _ = build_features(small_df)
    return fm
