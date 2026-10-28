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
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    classes = np.asarray(classes)
    if y_true.ndim != 1 or classes.ndim != 1:
        raise ValueError("Rótulos e classes devem ser vetores unidimensionais.")
    if y_proba.shape != (len(y_true), len(classes)):
        raise ValueError("Probabilidades devem ter formato (n_instâncias, n_classes).")
    class_index = pd.Index(classes)
    true_positions = class_index.get_indexer(y_true)
    if (true_positions < 0).any():
        raise ValueError("Todos os rótulos verdadeiros devem pertencer a classes.")
    predicted_positions = np.argmax(y_proba, axis=1)
    counts = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(counts, (true_positions, predicted_positions), 1)
    table = pd.DataFrame(
        counts,
        index=pd.Index(classes, name="verdadeira"),
        columns=pd.Index(classes, name="prevista"),
    )
    if normalize:
        support = table.sum(axis=1)
        table = table.div(support.where(support > 0), axis=0).fillna(0.0)
    return table


def recall_per_class(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    """Recall de cada classe: acertos da classe / instâncias da classe.

    Returns:
        DataFrame com colunas class, support (instâncias verdadeiras da classe) e
        recall (NaN quando support = 0), uma linha por classe na ordem de `classes`.
    """
    counts = confusion_table(y_true, y_proba, classes)
    support = counts.sum(axis=1).to_numpy()
    recall = np.divide(
        np.diag(counts.to_numpy()),
        support,
        out=np.full(len(support), np.nan),
        where=support > 0,
    )
    return pd.DataFrame({"class": np.asarray(classes), "support": support, "recall": recall})
