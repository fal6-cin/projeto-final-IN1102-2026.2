"""Leitura e organização dos resultados (infraestrutura comum a todos os grupos).

Dois arquivos, duas fases do protocolo:
    - `results/validation.csv` (e `validation_train.csv`): desenvolvimento por CV
      interna no treino de cada fold externo (`run_validation`); apenas diagnóstico
      (não são base de decisões) e base da análise de overfitting;
    - `results/test.csv`: avaliação final única no teste de cada fold externo
      (`final_test`), gerada após o congelamento; usada só no relatório final
      (Demšar, bootstrap, regimes), nunca para voltar a ajustar o modelo.

A regra de completude abaixo define o número de datasets N usado nos testes
estatísticos e deve ser a mesma para todos os grupos: entram apenas datasets em
que todos os sistemas comparados concluíram todos os folds externos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_RAW = Path("results/test.csv")
DEFAULT_VALIDATION = Path("results/validation.csv")
DEFAULT_VALIDATION_TRAIN = Path("results/validation_train.csv")


def load_raw(path: Path = DEFAULT_RAW) -> pd.DataFrame:
    """Lê o CSV da avaliação final gerado por `final_test` (uma linha por dataset, fold e sistema)."""
    return pd.read_csv(path)


def load_validation(path: Path = DEFAULT_VALIDATION, train: bool = False) -> pd.DataFrame:
    """Lê o CSV da fase de desenvolvimento gerado por `run_validation`.

    Com `train=True` lê `validation_train.csv` (métricas no treino interno) do
    mesmo diretório. Linhas de `group_tuned` têm `optimistic = True`.
    """
    path = Path(path)
    if train:
        path = path.with_name(DEFAULT_VALIDATION_TRAIN.name)
    return pd.read_csv(path)


def ok_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Linhas concluídas com sucesso."""
    return raw[raw["status"] == "ok"] if "status" in raw.columns else raw


def per_dataset(
    raw: pd.DataFrame, metric: str = "auc_ovo", models: list[str] | None = None
) -> pd.DataFrame:
    """Tabela `dataset x sistema` com a média da métrica nos folds externos (uma medida por dataset).

    Mantém apenas datasets em que todos os sistemas indicados concluíram todos os folds externos.
    """
    df = ok_results(raw)
    if models is not None:
        df = df[df["model"].isin(models)]
    counts = df.groupby(["task_id", "model"])["fold"].nunique().unstack()
    n_folds = int(counts.max().max())
    complete = counts.index[(counts == n_folds).all(axis=1)]
    table = df[df["task_id"].isin(complete)].pivot_table(
        index="task_id", columns="model", values=metric, aggfunc="mean"
    )
    return table if models is None else table[models]
