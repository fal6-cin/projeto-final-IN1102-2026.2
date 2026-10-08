"""Gráficos da análise de overfitting (trabalho do grupo).

Entradas produzidas apenas com dados de desenvolvimento (CV interna no treino):
`src.pipeline.overfitting.validation_curve` (ou `src.pipeline.cv_grid.cv_grid`),
`src.pipeline.overfitting.generalization_gap` e
`src.pipeline.class_errors.confusion_table` (predições de validação interna em
`results/preds_val/`, nunca predições de teste).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_validation_curve(
    curve: pd.DataFrame,
    metric: str = "auc_ovo",
    log_x: bool = False,
    output: Path | None = None,
) -> plt.Figure:
    """Curvas de treino e de validação (média e faixa de média ± 1 EP entre folds) por valor.

    O erro padrão (EP) de cada valor é o desvio amostral (ddof = 1) entre folds
    dividido pela raiz do número de folds, o mesmo da regra de 1 EP em
    `select_complexity` (coerente com a Figura 2 do enunciado).

    Args:
        curve: saída de `validation_curve` ou `cv_grid` (colunas `param`, `value`,
            `fold`, `<metric>_train`, `<metric>_val`).
        log_x: eixo x em escala logarítmica.
        output: caminho opcional para salvar a figura.

    Returns:
        A `matplotlib.figure.Figure`, com duas linhas (treino e validação) no eixo.
    """
    raise NotImplementedError("TODO(grupo): desenhe treino e validação com faixas de média ± 1 EP por valor")


def plot_generalization_gap(gap_table: pd.DataFrame, output: Path | None = None) -> plt.Figure:
    """Barras da lacuna média treino-validação por sistema, com barras de erro (gap_std).

    Args:
        gap_table: saída de `generalization_gap` (colunas `model`, `gap_mean`, `gap_std`).
        output: caminho opcional para salvar a figura.

    Returns:
        A `matplotlib.figure.Figure`, com uma barra por sistema.
    """
    raise NotImplementedError("TODO(grupo): uma barra por sistema com gap_mean e erro gap_std")


def plot_confusion(table: pd.DataFrame, output: Path | None = None):
    """Mapa de calor da matriz de confusão (formato de `confusion_table`), com os valores anotados."""
    raise NotImplementedError("TODO(grupo): desenhe a matriz com imshow e anote cada célula")
