"""Validação cruzada interna sobre o TREINO de um fold externo (infraestrutura pronta).

Protocolo da pós-graduação (IN1102): os 3 folds oficiais do TabArena (repetição 0) são
a validação cruzada EXTERNA. Toda decisão (busca de hiperparâmetros, estudos de
sensibilidade, estudo específico do modelo, curvas e análise de overfitting) é
tomada por validação cruzada INTERNA dentro do treino de cada fold externo. O
teste de cada fold externo só é usado uma vez, em `src.pipeline.final_test`,
depois do congelamento (`src.pipeline.freeze`).

Este módulo oferece:

- `stratified_subsample`: subamostra estratificada (mesma regra da busca em
  `tune.py`), usada nos datasets grandes;
- `inner_cv_splits`: partições estratificadas da CV interna;
- `cv_grid`: estudo de sensibilidade de um hiperparâmetro por CV interna, com
  métricas de treino e de validação; prepara subamostra e partições e delega o
  laço a `src.pipeline.overfitting.validation_curve` (trabalho do grupo).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.pipeline.evaluate import METRICS

DEFAULT_CV_FOLDS = 3
DEFAULT_MAX_SAMPLES = 10_000
DEFAULT_SEED = 42


def stratified_subsample(
    X: pd.DataFrame, y: np.ndarray, max_samples: int | None, seed: int = DEFAULT_SEED
) -> tuple[pd.DataFrame, np.ndarray]:
    """Subamostra estratificada de até `max_samples` instâncias (None mantém tudo).

    Os rótulos do índice de `X` são preservados, o que permite rastrear quais
    instâncias foram usadas.
    """
    y = np.asarray(y)
    if max_samples is None or len(y) <= max_samples:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=max_samples, stratify=y, random_state=seed
    )
    return X_sub, np.asarray(y_sub)


def inner_cv_splits(
    y: np.ndarray, n_folds: int = DEFAULT_CV_FOLDS, seed: int = DEFAULT_SEED
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Posições (treino interno, validação interna) de uma CV estratificada com embaralhamento."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), np.asarray(y)))


def cv_grid(
    factory: Callable[..., Any],
    param: str,
    values: Iterable[Any],
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int = DEFAULT_CV_FOLDS,
    seed: int = DEFAULT_SEED,
    metric: str = "auc_ovo",
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> pd.DataFrame:
    """Estudo de um hiperparâmetro por CV interna estratificada sobre dados de TREINO.

    Prepara a subamostra estratificada (`stratified_subsample`) e as partições
    internas (`inner_cv_splits`), as mesmas da busca, e DELEGA o laço de ajuste e
    medição a `src.pipeline.overfitting.validation_curve` (trabalho do grupo).
    Nunca recebe nem acessa o teste do fold externo: passe apenas `X_train`, `y_train`.

    Args:
        factory: função `factory(seed, **params)` que devolve um estimador com
            `fit`, `predict_proba` e `classes_`.
        param: nome do hiperparâmetro variado.
        values: valores avaliados (na ordem desejada para o gráfico).
        X, y: dados de TREINO de um fold externo.
        cv_folds: número de folds da CV interna (pelo menos 2).
        metric: métrica primária do estudo (informativa; todas as métricas de
            METRICS são calculadas). Deve pertencer a METRICS.
        max_samples: subamostra estratificada máxima (None desativa), como na busca.

    Returns:
        A saída de `validation_curve`: uma linha por (valor, fold interno), na
        ordem de `values` e dos folds, e colunas `param`, `value`, `fold`,
        `n_train`, `n_val` e, para cada m em METRICS, `<m>_train` e `<m>_val`.
    """
    from src.pipeline import overfitting

    if metric not in METRICS:
        raise ValueError(f"Métrica desconhecida: {metric}. Use uma de {METRICS}.")
    X, y = stratified_subsample(X, np.asarray(y), max_samples, seed)
    splits = inner_cv_splits(y, cv_folds, seed)
    return overfitting.validation_curve(factory, param, list(values), X, y, splits, seed)
