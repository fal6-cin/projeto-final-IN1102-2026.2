"""Tabelas-resumo a partir de `results/test.csv` (avaliação final, `final_test`) para o relatório."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline.results import ok_results

DEFAULT_METRICS = [
    "auc_ovo",
    "accuracy",
    "g_mean",
    "cross_entropy",
    "tune_time_s",
    "fit_time_s",
    "predict_time_s",
    "total_time_s",
]


def summary_by_model(raw: pd.DataFrame, metrics: list[str] = DEFAULT_METRICS) -> pd.DataFrame:
    """Média e desvio padrão entre datasets, por sistema.

    Cada dataset é primeiro resumido pela média dos seus folds; só então se
    calculam média e desvio entre datasets. Colunas de saída: model e
    `<métrica>_mean`, `<métrica>_std` para cada métrica.
    """
    df = ok_results(raw)
    by_dataset = df.groupby(["model", "task_id"], as_index=False)[metrics].mean()
    grouped = by_dataset.groupby("model", sort=False)

    rows = []
    for model, group in grouped:
        row: dict[str, object] = {"model": model}
        for m in metrics:
            row[f"{m}_mean"] = float(group[m].mean())
            row[f"{m}_std"] = float(group[m].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def per_dataset_table(raw: pd.DataFrame, metric: str = "auc_ovo") -> pd.DataFrame:
    """Tabela `dataset x sistema` com texto 'média ± desvio' da métrica entre os folds."""
    df = ok_results(raw)
    agg = (
        df.groupby(["dataset", "model"], sort=False)[metric]
        .agg(mean="mean", std=lambda s: s.std(ddof=1))
        .reset_index()
    )
    agg["formatted"] = agg.apply(
        lambda r: f"{r['mean']:.4f} ± {0.0 if pd.isna(r['std']) else r['std']:.4f}",
        axis=1,
    )
    table = agg.pivot(index="dataset", columns="model", values="formatted")
    return table


def export_markdown(df: pd.DataFrame, output_path: Path, floatfmt: str = ".4f") -> None:
    """Exporta uma tabela em Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(df.to_markdown(index=False, floatfmt=floatfmt), encoding="utf-8")
