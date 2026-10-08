"""Contrato do modelo tema do grupo (src/models/group_model.py).

Verifica a compatibilidade com scikit-learn e com o protocolo do projeto:
rótulos de texto, atributos categóricos, valores ausentes, multiclasse,
probabilidades válidas alinhadas a `classes_`, reprodutibilidade e espaço de
busca. Requer o extra do modelo tema instalado.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.base import clone

from src.models.group_model import build_group_model, search_space
from src.pipeline.tune import tune
from tests.dados_sinteticos import make_dataset, split


@pytest.mark.parametrize("n_classes", [2, 3])
def test_group_model_contract(n_classes):
    X_train, y_train, X_test, _ = split(*make_dataset(n=300, n_classes=n_classes))
    X_before = X_test.copy()
    model = build_group_model(seed=42)
    clone(model)  # compatível com scikit-learn (get_params)
    assert model.fit(X_train, y_train) is model
    proba = np.asarray(model.predict_proba(X_test))
    assert proba.shape == (len(X_test), n_classes)
    assert np.all(proba >= 0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert sorted(map(str, model.classes_)) == sorted(np.unique(y_train))
    pd.testing.assert_frame_equal(X_test, X_before)  # não altera a entrada


def test_group_model_unseen_category_and_single_row():
    X_train, y_train, X_test, _ = split(*make_dataset(n=300, n_classes=3))
    X_test = X_test.copy()
    X_test["c"] = pd.Categorical(
        X_test["c"].astype(str).where(X_test.index % 7 != 0, "nova"),
        categories=["x", "y", "z", "nova"],
    )
    model = build_group_model(seed=1).fit(X_train, y_train)
    proba = np.asarray(model.predict_proba(X_test))
    assert np.all(np.isfinite(proba)) and proba.shape == (len(X_test), 3)
    assert np.asarray(model.predict_proba(X_test.iloc[:1])).shape == (1, 3)


def test_group_model_is_reproducible():
    X_train, y_train, X_test, _ = split(*make_dataset(n=200))
    p1 = build_group_model(seed=7).fit(X_train, y_train).predict_proba(X_test)
    p2 = build_group_model(seed=7).fit(X_train, y_train).predict_proba(X_test)
    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_search_space_builds_valid_models():
    params = search_space(optuna.create_study().ask())
    assert isinstance(params, dict) and params
    build_group_model(seed=42, **params)
    X, y = make_dataset(n=200)
    result = tune(build_group_model, search_space, X, y, n_trials=2, cv_folds=2, timeout_s=None)
    assert isinstance(result.best_params, dict)
