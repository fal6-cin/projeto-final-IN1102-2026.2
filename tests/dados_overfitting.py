"""Dados sintéticos dos estudos de overfitting (CV interna no treino)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_numeric(n: int = 400, seed: int = 0, noise: float = 1.0):
    """Problema binário só com atributos numéricos e ruído no rótulo (árvores profundas sobreajustam)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["x1", "x2", "x3", "x4"])
    score = X["x1"] + 0.5 * X["x2"] + rng.normal(scale=noise, size=n)
    y = np.where(score > 0, "pos", "neg")
    return X, y


def validation_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arquivos de validação e de treino no formato de `run_validation`, com valores à mão.

    Sistema a: treino (0.9, 0.8), validação (0.7, 0.7): gaps 0.2 e 0.1.
    Sistema b: treino 0.6, validação 0.65: gap -0.05 (um único par).
    """
    keys = {
        "task_id": [1, 1, 1],
        "dataset": ["ds1", "ds1", "ds1"],
        "fold": [0, 0, 0],
        "inner_fold": [0, 1, 0],
        "model": ["a", "a", "b"],
    }
    val = pd.DataFrame(
        {
            **keys,
            "auc_ovo": [0.7, 0.7, 0.65],
            "cross_entropy": [0.5, 0.6, 0.4],
            "status": "ok",
            "error": "",
        }
    )
    train = pd.DataFrame(
        {
            **keys,
            "n_train_eval": [100, 100, 100],
            "auc_ovo": [0.9, 0.8, 0.6],
            "cross_entropy": [0.2, 0.4, 0.5],
            "status": "ok",
            "error": "",
        }
    )
    return val, train


def complexity_curve(metric: str = "auc_ovo", lower: bool = False) -> pd.DataFrame:
    """Curva com 4 valores e 3 folds; médias de validação 0.80, 0.876, 0.88, 0.877.

    Desvio entre folds 0.01 em todos os valores (EP = 0.01 / raiz de 3). Com
    `lower=True` a métrica é 1 - média (menor é melhor).
    """
    means = {1: 0.80, 2: 0.876, 3: 0.88, 4: 0.877}
    rows = []
    for value, m in means.items():
        for fold, delta in enumerate((-0.01, 0.0, 0.01)):
            v = m + delta
            rows.append(
                {
                    "param": "max_depth",
                    "value": value,
                    "fold": fold,
                    f"{metric}_train": 1.0,
                    f"{metric}_val": 1.0 - v if lower else v,
                }
            )
    return pd.DataFrame(rows)
