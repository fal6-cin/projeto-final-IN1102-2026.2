"""Aceitação de src/pipeline/bootstrap.py: intervalo percentil por bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.pipeline.bootstrap import bootstrap_ci


def _predictions(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(["a", "b"] * (n // 2))
    proba = np.column_stack([y == "a", y == "b"]) * 0.3 + rng.uniform(0, 0.7, (n, 2))
    return y, proba / proba.sum(axis=1, keepdims=True), np.array(["a", "b"])


def test_bootstrap_is_deterministic_and_contains_estimate():
    y, proba, classes = _predictions()
    a = bootstrap_ci(y, proba, classes, "auc_ovo", n_boot=300, seed=1)
    b = bootstrap_ci(y, proba, classes, "auc_ovo", n_boot=300, seed=1)
    assert a == b
    assert a["ci_low"] <= a["estimate"] <= a["ci_high"]
    assert set(a) >= {"estimate", "ci_low", "ci_high", "n_boot"}


def test_bootstrap_accuracy_close_to_scipy_percentile():
    y, proba, classes = _predictions()
    hits = (classes[np.argmax(proba, axis=1)] == y).astype(float)
    ref = stats.bootstrap((hits,), np.mean, n_resamples=2000, method="percentile", random_state=0)
    ours = bootstrap_ci(y, proba, classes, "accuracy", n_boot=2000, seed=0)
    assert ours["ci_low"] == pytest.approx(ref.confidence_interval.low, abs=0.01)
    assert ours["ci_high"] == pytest.approx(ref.confidence_interval.high, abs=0.01)


def test_interval_narrows_with_more_test_instances():
    small = bootstrap_ci(*_predictions(n=100), "auc_ovo", n_boot=300)
    large = bootstrap_ci(*_predictions(n=2000), "auc_ovo", n_boot=300)
    assert (large["ci_high"] - large["ci_low"]) < (small["ci_high"] - small["ci_low"])
