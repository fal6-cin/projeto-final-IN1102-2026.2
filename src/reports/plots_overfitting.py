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
    grouped = curve.groupby("value", sort=False)
    if curve.empty or (grouped["fold"].nunique() < 2).any():
        raise ValueError("Cada valor da curva precisa de pelo menos 2 folds.")
    values = list(grouped.groups)
    try:
        x = np.asarray(values, dtype=float)
        categorical = False
    except (TypeError, ValueError):
        x = np.arange(len(values))
        categorical = True
    if log_x and (categorical or np.any(x <= 0)):
        raise ValueError("Escala log exige valores numéricos positivos.")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for suffix, label, color in (("train", "Treino", "C0"), ("val", "Validação", "C1")):
        stats = grouped[f"{metric}_{suffix}"].agg(["mean", "std", "count"])
        mean = stats["mean"].to_numpy()
        se = (stats["std"] / np.sqrt(stats["count"])).to_numpy()
        ax.plot(x, mean, marker="o", label=label, color=color)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2)
    if categorical:
        ax.set_xticks(x, [str(value) for value in values], rotation=25, ha="right")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(str(curve["param"].iloc[0]))
    ax.set_ylabel(metric)
    ax.set_title("Curva de validação — média ± 1 erro padrão")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def plot_generalization_gap(gap_table: pd.DataFrame, output: Path | None = None) -> plt.Figure:
    """Barras da lacuna média treino-validação por sistema, com barras de erro (gap_std).

    Args:
        gap_table: saída de `generalization_gap` (colunas `model`, `gap_mean`, `gap_std`).
        output: caminho opcional para salvar a figura.

    Returns:
        A `matplotlib.figure.Figure`, com uma barra por sistema.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(gap_table))
    # Um único par não define desvio amostral: omitir sua barra de erro.
    errors = gap_table["gap_std"].fillna(0.0).to_numpy()
    ax.bar(x, gap_table["gap_mean"].to_numpy(), yerr=errors, capsize=4, color="C0")
    ax.set_xticks(x, gap_table["model"].astype(str), rotation=25, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Lacuna orientada (média ± desvio padrão)")
    ax.set_title("Lacuna de generalização por sistema")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def plot_confusion(table: pd.DataFrame, output: Path | None = None):
    """Mapa de calor da matriz de confusão (formato de `confusion_table`), com os valores anotados."""
    values = table.to_numpy()
    fig, ax = plt.subplots(figsize=(max(5, 0.65 * len(table.columns)),
                                    max(4.5, 0.65 * len(table.index))))
    heatmap = ax.imshow(values, cmap="Blues", vmin=0)
    fig.colorbar(heatmap, ax=ax)
    ax.set_xticks(np.arange(len(table.columns)), table.columns.astype(str),
                  rotation=35, ha="right")
    ax.set_yticks(np.arange(len(table.index)), table.index.astype(str))
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title("Matriz de confusão")
    threshold = heatmap.norm.vmin + (heatmap.norm.vmax - heatmap.norm.vmin) / 2
    integer_counts = all(pd.api.types.is_integer_dtype(dtype) for dtype in table.dtypes)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = str(int(value)) if integer_counts else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center",
                    color="white" if value > threshold else "black")
    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig
