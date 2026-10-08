"""Partições estratificadas para dados locais e testes automatizados.

Protocolo da pós-graduação (IN1102): o experimento usa as partições oficiais das tasks
do TabArena-v0.1 (`data.load_tabarena.get_folds`: repetição 0, 3 folds
estratificados) como validação cruzada EXTERNA. O teste de cada fold externo só
é usado em `src.pipeline.final_test`, uma única vez, após o congelamento.

Esta função gera partições equivalentes para dados sintéticos e testes. Para a
CV INTERNA sobre o treino de um fold externo (busca, sensibilidade, curvas e
overfitting), use `src.pipeline.cv_grid.inner_cv_splits`.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

DEFAULT_SEED = 42
DEFAULT_N_FOLDS = 3


def stratified_folds(
    y: np.ndarray, n_folds: int = DEFAULT_N_FOLDS, seed: int = DEFAULT_SEED
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Índices (treino, teste) de uma validação cruzada estratificada com embaralhamento."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))
