"""Busca de hiperparâmetros com Optuna, restrita ao treino de cada fold externo.

Protocolo da pós-graduação (IN1102): os 3 folds oficiais (repetição 0) formam a
validação cruzada EXTERNA. A busca usa validação cruzada estratificada INTERNA
sobre os dados de treino de cada fold externo; o teste do fold externo nunca é
visto durante a busca e só é usado uma vez, em `src.pipeline.final_test`, após
o congelamento do protocolo (`src.pipeline.freeze`). O espaço de busca deve ser
justificado a priori (literatura e documentação do modelo). O tempo gasto é
medido e somado ao custo total do modelo.

A subamostra e as partições internas são as mesmas de `src.pipeline.cv_grid`
(`stratified_subsample` e `inner_cv_splits`), de modo que estudos e busca
compartilham a CV interna.

Orçamento padrão do projeto (ver enunciado): 20 tentativas, CV interna de 3
folds, limite de 10 minutos por fold externo e subamostra estratificada de até
10.000 instâncias para a busca nos datasets grandes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.pipeline.cv_grid import inner_cv_splits, stratified_subsample

DEFAULT_N_TRIALS = 20
DEFAULT_CV_FOLDS = 3
DEFAULT_TIMEOUT_S = 600
DEFAULT_MAX_SAMPLES = 10_000


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_score: float
    tune_time_s: float
    study: optuna.Study


def _auc(estimator: Any, X: pd.DataFrame, y: np.ndarray) -> float:
    proba = np.asarray(estimator.predict_proba(X))
    classes = np.asarray(estimator.classes_)
    if classes.size == 2:
        return float(roc_auc_score(y == classes[1], proba[:, 1]))
    return float(roc_auc_score(y, proba, multi_class="ovo", labels=classes))


def tune(
    estimator_factory: Callable[..., Any],
    search_space: Callable[[optuna.Trial], dict[str, Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42,
    n_trials: int = DEFAULT_N_TRIALS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    timeout_s: float | None = DEFAULT_TIMEOUT_S,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> TuningResult:
    """Executa a busca e retorna os melhores hiperparâmetros (maximizando AUC OvO).

    Args:
        estimator_factory: função `factory(seed, **params)` que devolve um
            estimador compatível com scikit-learn.
        search_space: função que recebe um `optuna.Trial` e devolve o dict de
            hiperparâmetros amostrados.
        X, y: dados de TREINO do fold externo (nunca o teste).
        timeout_s: limite de tempo da busca (None desativa). O Optuna verifica o
            limite entre tentativas: a tentativa em curso termina antes da parada,
            por isso o tempo efetivo (`tune_time_s`) pode exceder o limite e deve
            ser relatado.
        max_samples: tamanho máximo da subamostra estratificada usada na busca.
    """
    X, y = stratified_subsample(X, np.asarray(y), max_samples, seed)
    splits = inner_cv_splits(y, cv_folds, seed)

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        scores = []
        for train_idx, val_idx in splits:
            estimator = estimator_factory(seed, **params)
            estimator.fit(X.iloc[train_idx], y[train_idx])
            scores.append(_auc(estimator, X.iloc[val_idx], y[val_idx]))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
    tune_time_s = time.perf_counter() - t0
    # Reconstrói o dicionário (possivelmente aninhado) a partir da melhor tentativa.
    best_params = search_space(optuna.trial.FixedTrial(study.best_params))
    return TuningResult(
        best_params=best_params,
        best_score=float(study.best_value),
        tune_time_s=tune_time_s,
        study=study,
    )
