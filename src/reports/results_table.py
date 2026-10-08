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
    raise NotImplementedError("TODO(grupo): agregue primeiro por (model, task_id) e depois por model")


def per_dataset_table(raw: pd.DataFrame, metric: str = "auc_ovo") -> pd.DataFrame:
    """Tabela `dataset x sistema` com texto 'média ± desvio' da métrica entre os folds."""
    raise NotImplementedError("TODO(grupo): agrupe por (dataset, model), calcule média e desvio e formate o texto")


def export_markdown(df: pd.DataFrame, output_path: Path, floatfmt: str = ".4f") -> None:
    """Exporta uma tabela em Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(df.to_markdown(index=False, floatfmt=floatfmt), encoding="utf-8")
