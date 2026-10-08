"""Catálogo, validação da seleção e carregamento dos datasets do TabArena-v0.1.

A seleção dos 30 datasets é atividade de cada grupo. Este módulo oferece:

- `catalog()`: os 38 datasets de classificação da suíte OpenML 457
  (TabArena-v0.1), com metadados e regime de tamanho calculado;
- `validate_selection()`: confere se a seleção do grupo atende às regras do
  enunciado (pertencimento à suíte, 10 datasets por regime, presença de
  problemas binários e multiclasse);
- `load_task()`, `get_folds()` e `get_train_indices()`: carregamento via
  OpenML, com cache local, e partições oficiais de treino e teste das tasks do
  TabArena (validação cruzada externa; o teste é usado só em `final_test`).

O catálogo é lido de `data/tabarena_v01_classificacao.csv` (instantâneo da
suíte); `refresh_catalog()` regenera esse arquivo a partir do OpenML.

Referências:
    https://tabarena.ai
    https://www.openml.org/search?type=study&study_type=task&id=457
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SUITE_ID = 457
DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
CATALOG_PATH = DATA_DIR / "tabarena_v01_classificacao.csv"
SELECTION_PATH = DATA_DIR / "selecao_datasets.txt"
DEFAULT_CACHE_DIR = REPO_ROOT / "cache" / "openml"

# Regimes de tamanho (n = número total de instâncias do dataset):
#   pequeno: n < 2.500; médio: 2.500 <= n < 10.000; grande: n >= 10.000.
REGIME_THRESHOLDS = {"small": 2_500, "medium": 10_000}
REGIMES = ("small", "medium", "large")
REQUIRED_PER_REGIME = {"small": 10, "medium": 10, "large": 10}

# Partições oficiais: as tasks do TabArena-v0.1 definem 10 repetições de
# validação cruzada estratificada com 3 folds. O projeto usa a repetição 0.
OFFICIAL_REPEAT = 0
OFFICIAL_FOLDS = (0, 1, 2)

_cache_configured = False


def _configure_cache() -> None:
    """Configura o cache do OpenML (feito apenas no primeiro carregamento)."""
    global _cache_configured
    if _cache_configured:
        return
    import openml

    cache_dir = Path(os.environ.get("TABARENA_CACHE", DEFAULT_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(cache_dir))
    _cache_configured = True


def classify_regime(n_samples: int) -> str:
    """Retorna 'small', 'medium' ou 'large' conforme o número total de instâncias."""
    if n_samples < REGIME_THRESHOLDS["small"]:
        return "small"
    if n_samples < REGIME_THRESHOLDS["medium"]:
        return "medium"
    return "large"


def catalog() -> pd.DataFrame:
    """Catálogo dos datasets de classificação do TabArena-v0.1 com regimes calculados."""
    df = pd.read_csv(CATALOG_PATH)
    df["regime_size"] = df["n_samples"].apply(classify_regime)
    df["problem"] = np.where(df["n_classes"] == 2, "binary", "multiclass")
    df["cat_share"] = df["n_categorical"] / df["n_features"]
    df["has_missing"] = df["n_missing_values"] > 0
    df["imbalance_ratio"] = df["majority_class_size"] / df["minority_class_size"]
    return df


def refresh_catalog(path: Path = CATALOG_PATH) -> pd.DataFrame:
    """Regenera o instantâneo do catálogo a partir da suíte OpenML 457."""
    import openml

    _configure_cache()
    suite = openml.study.get_suite(SUITE_ID)
    tasks = openml.tasks.list_tasks(task_id=list(suite.tasks), output_format="dataframe")
    tasks = tasks[tasks["task_type"] == "Supervised Classification"]
    df = pd.DataFrame(
        {
            "task_id": tasks["tid"].astype(int),
            "dataset_id": tasks["did"].astype(int),
            "name": tasks["name"],
            "n_samples": tasks["NumberOfInstances"].astype(int),
            # NumberOfFeatures e NumberOfSymbolicFeatures incluem o alvo nominal.
            "n_features": tasks["NumberOfFeatures"].astype(int) - 1,
            "n_categorical": (tasks["NumberOfSymbolicFeatures"].astype(int) - 1).clip(lower=0),
            "n_classes": tasks["NumberOfClasses"].astype(int),
            "n_missing_values": tasks["NumberOfMissingValues"].astype(int),
            "minority_class_size": tasks["MinorityClassSize"].astype(int),
            "majority_class_size": tasks["MajorityClassSize"].astype(int),
        }
    ).sort_values("n_samples")
    df.to_csv(path, index=False)
    return df


def read_selection(path: Path = SELECTION_PATH) -> list[int]:
    """Lê os task IDs escolhidos pelo grupo (um por linha; '#' inicia comentário)."""
    ids: list[int] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        content = line.split("#", 1)[0].strip()
        if content:
            ids.append(int(content))
    return ids


@dataclass
class SelectionReport:
    table: pd.DataFrame
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_selection(
    task_ids: Iterable[int],
    required_per_regime: dict[str, int] = REQUIRED_PER_REGIME,
) -> SelectionReport:
    """Confere a seleção do grupo contra as regras do enunciado.

    Erros: task fora do catálogo, task repetida, contagem por regime diferente
    da exigida. Avisos: regime sem problema binário ou sem multiclasse, quando
    o catálogo oferece ambos.
    """
    ids = list(task_ids)
    cat = catalog().set_index("task_id")
    errors: list[str] = []
    warnings: list[str] = []

    duplicates = sorted({t for t in ids if ids.count(t) > 1})
    if duplicates:
        errors.append(f"Task IDs repetidos: {duplicates}")
    unknown = [t for t in ids if t not in cat.index]
    if unknown:
        errors.append(f"Task IDs fora do catálogo de classificação do TabArena-v0.1: {unknown}")

    table = cat.loc[[t for t in dict.fromkeys(ids) if t in cat.index]].reset_index()
    counts = table["regime_size"].value_counts()
    for regime, required in required_per_regime.items():
        found = int(counts.get(regime, 0))
        if found != required:
            errors.append(f"Regime '{regime}': {found} datasets selecionados, exigidos {required}.")
        offered = set(cat.loc[cat["regime_size"] == regime, "problem"])
        chosen = set(table.loc[table["regime_size"] == regime, "problem"])
        for problem in ("binary", "multiclass"):
            if problem in offered and problem not in chosen:
                warnings.append(
                    f"Regime '{regime}' sem problema '{problem}', embora o catálogo ofereça."
                )
    return SelectionReport(table=table, errors=errors, warnings=warnings)


@dataclass(frozen=True)
class TabularDataset:
    task_id: int
    name: str
    X: pd.DataFrame
    y: np.ndarray
    categorical_indicator: list[bool]
    attribute_names: list[str]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return int(pd.unique(self.y).size)

    @property
    def n_categorical(self) -> int:
        return int(sum(self.categorical_indicator))

    @property
    def has_missing(self) -> bool:
        return bool(self.X.isna().any().any())

    @property
    def regime(self) -> str:
        return classify_regime(self.n_samples)


def load_task(task_id: int) -> TabularDataset:
    """Carrega um dataset do OpenML pelo task ID, com cache local."""
    import openml

    _configure_cache()
    task = openml.tasks.get_task(task_id, download_data=True, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=task.target_name)
    return TabularDataset(
        task_id=task_id,
        name=dataset.name,
        X=X,
        y=np.asarray(y),
        categorical_indicator=list(categorical_indicator),
        attribute_names=list(attribute_names),
    )


def get_folds(
    task_id: int,
    folds: Iterable[int] = OFFICIAL_FOLDS,
    repeat: int = OFFICIAL_REPEAT,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Índices (treino, teste) das partições oficiais da task no OpenML.

    Os folds oficiais são a validação cruzada EXTERNA do projeto. O teste de
    cada fold só deve ser usado em `src.pipeline.final_test`; no
    desenvolvimento, use `get_train_indices`.
    """
    import openml

    _configure_cache()
    task = openml.tasks.get_task(task_id, download_data=False, download_splits=True)
    return [task.get_train_test_split_indices(fold=f, repeat=repeat) for f in folds]


def get_train_indices(
    task_id: int,
    folds: Iterable[int] = OFFICIAL_FOLDS,
    repeat: int = OFFICIAL_REPEAT,
) -> list[np.ndarray]:
    """Somente os índices de TREINO de cada fold externo oficial (uso no desenvolvimento).

    Busca, estudos de sensibilidade, curvas e análise de overfitting são feitos
    por CV interna sobre esses índices; os índices de teste não são devolvidos.
    """
    return [np.asarray(train_idx) for train_idx, _ in get_folds(task_id, folds, repeat)]


def iter_datasets(task_ids: Iterable[int]) -> Iterator[TabularDataset]:
    """Itera sobre os datasets indicados."""
    for task_id in task_ids:
        yield load_task(task_id)


def summarize(task_ids: Iterable[int]) -> pd.DataFrame:
    """Tabela da seleção para o relatório: nome, task, n, features, classes e regimes."""
    cols = [
        "task_id",
        "name",
        "n_samples",
        "n_features",
        "n_classes",
        "n_categorical",
        "cat_share",
        "has_missing",
        "imbalance_ratio",
        "regime_size",
        "problem",
    ]
    cat = catalog().set_index("task_id")
    return cat.loc[list(task_ids)].reset_index()[cols]
