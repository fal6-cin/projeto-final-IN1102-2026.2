"""Estimador espião, dataset sintético e repositório git temporário para verificar o protocolo.

`SpyClassifier` registra cada chamada a `fit` e a `predict_proba` (tipo,
rótulo do sistema e rótulos do índice de X). O dataset sintético usa um índice
deslocado (rótulos diferentes das posições), de modo que os rótulos identificam
as instâncias de forma inequívoca ao longo de `iloc`, subamostras e CV interna.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tests.dados_sinteticos import make_dataset

SEEN: dict[str, set] = {"fit": set(), "predict": set()}
CALLS: list[tuple[str, str, frozenset]] = []


def reset() -> None:
    SEEN["fit"].clear()
    SEEN["predict"].clear()
    CALLS.clear()


class SpyClassifier(ClassifierMixin, BaseEstimator):
    """Classificador de frequências de classe que registra as chamadas e os índices vistos."""

    def __init__(self, random_state: int = 0, max_depth: int = 1, tag: str = "spy"):
        self.random_state = random_state
        self.max_depth = max_depth
        self.tag = tag

    def fit(self, X, y):
        labels = X.index.tolist()
        SEEN["fit"].update(labels)
        CALLS.append(("fit", self.tag, frozenset(labels)))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        return self

    def predict_proba(self, X):
        labels = X.index.tolist()
        SEEN["predict"].update(labels)
        CALLS.append(("predict", self.tag, frozenset(labels)))
        return np.tile(self.prior_, (len(X), 1))


def spy_factory(seed: int = 42, tag: str = "spy", **params) -> SpyClassifier:
    return SpyClassifier(random_state=seed, tag=tag, **params)


def spy_search_space(trial) -> dict:
    return {"max_depth": trial.suggest_int("max_depth", 1, 4)}


def install(monkeypatch) -> None:
    """Substitui LightGBM e o modelo do grupo por espiões com rótulos distintos."""
    from src.models import baselines, group_model

    monkeypatch.setitem(
        baselines.BASELINE_FACTORIES, "lightgbm", partial(spy_factory, tag="lightgbm")
    )
    monkeypatch.setattr(group_model, "build_group_model", partial(spy_factory, tag="group"))
    monkeypatch.setattr(group_model, "search_space", spy_search_space)
    reset()


def calls_touching(labels: set, kind: str | None = None) -> list[tuple[str, str, frozenset]]:
    """Chamadas registradas que viram algum dos rótulos (opcionalmente de um tipo)."""
    return [c for c in CALLS if c[2] & labels and (kind is None or c[0] == kind)]


def spy_dataset(task_id: int = 9999, n: int = 240, n_classes: int = 3):
    """TabularDataset sintético com índice deslocado (rótulos 10000, 10001, ...)."""
    from data.load_tabarena import TabularDataset

    X, y = make_dataset(n=n, n_classes=n_classes)
    X.index = X.index + 10_000
    return TabularDataset(
        task_id, f"sintetico{task_id}", X, y, [False, False, True], list(X.columns)
    )


# ------------------------------------------------------------------ repositório git temporário

GIT_ID = ["-c", "user.name=Teste", "-c", "user.email=teste@example.com"]


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *GIT_ID, *args], cwd=root, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def commit_all(root: Path, message: str, *paths: str) -> str:
    """Adiciona os caminhos (todos, se vazio) e cria um commit; devolve o hash."""
    git(root, "add", *(paths or ("-A",)))
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD")


def make_repo(base: Path, task_ids: str = "9999\n") -> Path:
    """Repositório git mínimo com arquivos travados (src/, data/) e um commit inicial."""
    from data.load_tabarena import REPO_ROOT

    root = base / "repo"
    for rel in (
        "src/models/group_model.py",
        "src/models/wrappers.py",
        "src/pipeline/evaluate.py",
        "data/load_tabarena.py",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, root / rel)
    (root / "data" / "selecao_datasets.txt").write_text(task_ids, encoding="utf-8")
    git(root, "init", "-q")
    git(root, "config", "core.autocrlf", "false")
    commit_all(root, "Commit inicial")
    return root
