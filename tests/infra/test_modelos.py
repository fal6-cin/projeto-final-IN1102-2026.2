"""Baselines GBDT e imputação de valores ausentes."""

from __future__ import annotations

import pytest

from src.pipeline.evaluate import fit_predict_evaluate
from tests.dados_sinteticos import make_dataset, split


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_baselines_run(model_name, n_classes):
    from src.models.baselines import BASELINE_FACTORIES

    result = fit_predict_evaluate(
        BASELINE_FACTORIES[model_name](seed=42), *split(*make_dataset(n_classes=n_classes))
    )
    assert 0.0 <= result.auc_ovo <= 1.0
    assert result.y_proba.shape[1] == n_classes
    assert set(result.classes) == {f"classe_{i}" for i in range(n_classes)}


def test_imputed_classifier_handles_missing_values():
    from sklearn.linear_model import LogisticRegression

    from src.models.wrappers import ImputedClassifier

    X, y = make_dataset()
    X = X[["a", "b"]]  # LogisticRegression não aceita NaN nem categorias
    result = fit_predict_evaluate(ImputedClassifier(LogisticRegression()), *split(X, y))
    assert 0.0 <= result.auc_ovo <= 1.0
