"""Análise dos resultados por regime de dataset.

Regimes considerados (metadados em `data.load_tabarena.catalog()`):
    - regime_size: pequeno (n < 2.500), médio (2.500 <= n < 10.000), grande (n >= 10.000)
    - problem: binário vs. multiclasse
    - regime_cat_share: baixa (< 25% de atributos categóricos) vs. alta (>= 25%)
    - regime_missing: com vs. sem valores ausentes

A análise por regime usa a tabela da avaliação final (`results/test.csv`,
gerada por `final_test` após o congelamento) e é descritiva (poucos datasets
por célula); não deve retroalimentar escolhas do modelo. O limiar de 25%
para a proporção categórica é uma convenção da disciplina;
discuta no relatório a sensibilidade das conclusões a esse limiar. Células com
poucos datasets (coluna `n_datasets`) devem ser interpretadas como evidência
descritiva, não como generalização.
"""

from __future__ import annotations

import pandas as pd

from data.load_tabarena import catalog
from src.pipeline.demsar import rank_matrix

CAT_SHARE_THRESHOLD = 0.25
REGIME_COLUMNS = ["regime_size", "problem", "regime_cat_share", "regime_missing"]


def regime_metadata(task_ids: list[int] | None = None) -> pd.DataFrame:
    """Metadados com as colunas de regime para os task IDs indicados."""
    df = catalog()
    if task_ids is not None:
        df = df[df["task_id"].isin(task_ids)]
    df = df.copy()
    df["regime_cat_share"] = (df["cat_share"] >= CAT_SHARE_THRESHOLD).map(
        {True: "high", False: "low"}
    )
    df["regime_missing"] = df["has_missing"].map({True: "yes", False: "no"})
    return df[["task_id", "name", "n_samples", *REGIME_COLUMNS]]


def aggregate_by_regime(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    regime_col: str,
) -> pd.DataFrame:
    """Média, desvio padrão amostral (ddof = 1) e número de datasets por (valor do regime, sistema).

    Colunas de saída: <regime_col>, model, mean, std e n_datasets. Células com um
    único dataset têm std = NaN.

    Args:
        table: tabela `dataset x sistema` (índice task_id), por exemplo a saída
            de `src.pipeline.results.per_dataset`.
        metadata: saída de `regime_metadata`.
        regime_col: uma das colunas de REGIME_COLUMNS.
    """
    long_df = (
        table.rename_axis("task_id")
        .reset_index()
        .melt(id_vars="task_id", var_name="model", value_name="value")
    )
    merged = long_df.merge(metadata[["task_id", regime_col]], on="task_id", how="inner")
    agg = (
        merged.groupby([regime_col, "model"], as_index=False)["value"]
        .agg(
            mean="mean",
            std=lambda s: s.std(ddof=1),
            n_datasets="count",
        )
    )
    return agg[[regime_col, "model", "mean", "std", "n_datasets"]]


def rank_by_regime(
    table: pd.DataFrame, metadata: pd.DataFrame, regime_col: str, higher_is_better: bool = True
) -> pd.DataFrame:
    """Posto médio dos sistemas dentro de cada valor do regime.

    Ranqueia cada dataset (1 = melhor, empates com posto médio) e agrega como em
    `aggregate_by_regime`. Colunas de saída: <regime_col>, model, mean_rank,
    std_rank e n_datasets.
    """
    ranks = rank_matrix(table, higher_is_better=higher_is_better)
    agg = aggregate_by_regime(ranks, metadata, regime_col)
    return agg.rename(columns={"mean": "mean_rank", "std": "std_rank"})
