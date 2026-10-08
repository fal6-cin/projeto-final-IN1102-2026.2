"""Erros por classe: matriz de confusão e recall por classe (trabalho do grupo).

Entradas no mesmo formato das predições de validação interna gravadas por
`run_validation` em `results/preds_val/` (o estudo usa somente validação, nunca
predições de teste): rótulos verdadeiros, matriz de probabilidades `y_proba`
(n x n_classes) e `classes` na ordem das colunas.
A classe prevista é `classes[argmax(y_proba)]` (empate: a primeira coluna).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def confusion_table(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray, normalize: bool = False
) -> pd.DataFrame:
    """Matriz de confusão como DataFrame.

    Linhas: classe verdadeira (índice com nome "verdadeira"); colunas: classe
    prevista (nome "prevista"); ambas na ordem de `classes`, incluindo classes
    sem instâncias. Sem normalização, contagens inteiras. Com `normalize=True`,
    cada linha é dividida pelo seu total (a diagonal passa a ser o recall da
    classe); linhas sem instâncias ficam com zeros.
    """
    raise NotImplementedError("TODO(grupo): derive a classe prevista pelo argmax e conte os pares (verdadeira, prevista) na ordem de classes")


def recall_per_class(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    """Recall de cada classe: acertos da classe / instâncias da classe.

    Returns:
        DataFrame com colunas class, support (instâncias verdadeiras da classe) e
        recall (NaN quando support = 0), uma linha por classe na ordem de `classes`.
    """
    raise NotImplementedError("TODO(grupo): use a diagonal da matriz de confusão e o total de cada linha")
