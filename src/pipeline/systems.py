"""Sistemas comparados no experimento e sua construção (infraestrutura comum).

Sistemas:
    lightgbm, xgboost, catboost   baselines GBDT com configuração padrão TD do pytabkit
    group_default                  modelo do grupo com hiperparâmetros padrão
    group_tuned                    modelo do grupo após busca de hiperparâmetros no treino

`build_runner` é usado tanto em `run_validation` (desenvolvimento, só treino)
quanto em `final_test` (avaliação final única). Em ambos os casos a função
devolvida recebe apenas o TREINO do fold externo; para `group_tuned`, a busca
Optuna (`tune.py`) é feita por CV interna sobre esse treino.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.tune import tune

BASELINES = ("lightgbm", "xgboost", "catboost")
GROUP = ("group_default", "group_tuned")
ALL_MODELS = BASELINES + GROUP


def build_runner(
    model: str, args: argparse.Namespace
) -> Callable[[pd.DataFrame, np.ndarray], tuple[Any, float, dict[str, Any]]]:
    """Devolve uma função `run(X_train, y_train) -> (estimador não treinado, tempo de busca, info)`.

    `info` é vazio para os sistemas sem busca; para `group_tuned`, contém
    `best_params` e `best_score` (AUC OvO média na CV interna da busca, estimativa
    otimista por ter sido usada na própria seleção).

    `args` deve ter `seed`, `n_trials`, `tune_timeout` e `tune_max_samples`.
    """
    if model in BASELINES:
        from src.models.baselines import BASELINE_FACTORIES

        factory = BASELINE_FACTORIES[model]
        return lambda X, y: (factory(args.seed), 0.0, {})

    if model == "group_default":
        from src.models.group_model import build_group_model

        return lambda X, y: (build_group_model(args.seed), 0.0, {})

    if model == "group_tuned":
        from src.models.group_model import build_group_model, search_space

        def run_tuned(X: pd.DataFrame, y: np.ndarray) -> tuple[Any, float, dict[str, Any]]:
            result = tune(
                build_group_model,
                search_space,
                X,
                y,
                seed=args.seed,
                n_trials=args.n_trials,
                timeout_s=args.tune_timeout,
                max_samples=args.tune_max_samples,
            )
            info = {"best_params": result.best_params, "best_score": result.best_score}
            return build_group_model(args.seed, **result.best_params), result.tune_time_s, info

        return run_tuned

    raise ValueError(f"Sistema desconhecido: {model}")
