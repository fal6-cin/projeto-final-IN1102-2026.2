"""Motor de busca de hiperparâmetros (restrito ao treino)."""

from __future__ import annotations

import optuna

from src.pipeline.tune import tune
from tests.dados_sinteticos import make_dataset


def test_tune_reconstructs_nested_params():
    from sklearn.linear_model import LogisticRegression

    X, y = make_dataset()
    X = X[["b"]]

    def factory(seed, config=None):
        return LogisticRegression(C=config["C"], random_state=seed)

    def space(trial: optuna.Trial):
        return {"config": {"C": trial.suggest_float("C", 0.01, 10, log=True)}}

    result = tune(factory, space, X, y, n_trials=3, cv_folds=2, timeout_s=None)
    assert set(result.best_params) == {"config"}
    assert result.tune_time_s > 0
