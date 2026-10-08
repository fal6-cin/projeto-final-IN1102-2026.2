"""Métricas exigidas pelo enunciado, calculadas a partir de predict_proba."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.evaluate import compute_metrics, g_mean_score


def test_metrics_follow_class_order_with_text_labels():
    classes = np.array(["neg", "pos"])
    y_true = np.array(["pos", "neg", "pos", "neg"])
    proba = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    m = compute_metrics(y_true, proba, classes)
    assert m["auc_ovo"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)


def test_multiclass_metrics_with_missing_class_in_test():
    classes = np.array(["a", "b", "c"])
    y_true = np.array(["a", "b", "a", "b"])
    proba = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.3, 0.6, 0.1]])
    m = compute_metrics(y_true, proba, classes)
    assert 0.0 <= m["auc_ovo"] <= 1.0
    assert np.isfinite(m["cross_entropy"])


def test_g_mean_is_zero_when_a_class_is_never_recalled():
    assert g_mean_score(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0])) == 0.0
    assert g_mean_score(np.array([0, 1, 2]), np.array([0, 1, 2])) == pytest.approx(1.0)
