"""Gráficos para o relatório: distribuição por sistema e custo versus desempenho.

O diagrama de diferença crítica está em `src.pipeline.demsar.plot_cd_diagram`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.pipeline.evaluate import HIGHER_IS_BETTER
from src.pipeline.results import ok_results


def boxplot_by_model(
    table: pd.DataFrame, metric: str = "auc_ovo", output: Path | None = None
) -> plt.Figure:
    """Boxplot da métrica por sistema (um ponto por dataset), ordenado pela mediana."""
    ascending = not HIGHER_IS_BETTER.get(metric, True)
    order = list(table.median(axis=0).sort_values(ascending=ascending).index)
    long_df = table.melt(var_name="model", value_name=metric)

    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = [table[m].dropna().to_numpy() for m in order]
    ax.boxplot(
        box_data,
        positions=range(len(order)),
        tick_labels=order,
        widths=0.5,
        showfliers=False,
        orientation="vertical",
    )
    sns.stripplot(
        data=long_df,
        x="model",
        y=metric,
        order=order,
        ax=ax,
        color="black",
        alpha=0.6,
        jitter=True,
    )
    ax.set_xlabel("Sistema")
    ax.set_ylabel(metric)
    ax.set_title(f"Distribuição de {metric} por sistema")
    fig.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")

    return fig


def cost_vs_performance(
    raw: pd.DataFrame,
    metric: str = "auc_ovo",
    time_col: str = "total_time_s",
    output: Path | None = None,
) -> plt.Figure:
    """Tempo mediano por fold (escala log) versus métrica média, um ponto por sistema.

    Usa a mediana do tempo entre datasets, pois os tempos variam em ordens de
    grandeza conforme o tamanho do dataset.
    """
    df = ok_results(raw)
    by_ds = df.groupby(["model", "task_id"], as_index=False).agg(
        metric_val=(metric, "mean"),
        time_val=(time_col, "median"),
    )
    summary = by_ds.groupby("model", as_index=False).agg(
        metric_mean=("metric_val", "mean"),
        time_median=("time_val", "median"),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=summary,
        x="time_median",
        y="metric_mean",
        hue="model",
        style="model",
        s=100,
        ax=ax,
    )
    for _, row in summary.iterrows():
        ax.annotate(
            str(row["model"]),
            (float(row["time_median"]), float(row["metric_mean"])),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_xlabel(f"Tempo mediano ({time_col}, escala log)")
    ax.set_ylabel(f"Média de {metric}")
    ax.set_title(f"Custo computacional vs. desempenho ({metric})")
    fig.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")

    return fig
