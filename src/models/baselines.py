"""Baselines obrigatórios: LightGBM, XGBoost e CatBoost.

Usam a configuração padrão TD do `pytabkit`: hiperparâmetros padrão obtidos
por meta-aprendizado em uma coleção de datasets de referência (Holzmüller et
al., NeurIPS 2024). Os baselines não passam por busca de hiperparâmetros no
projeto; essa assimetria deve ser discutida no relatório.

Os wrappers do pytabkit não aceitam valores ausentes em colunas numéricas;
por isso os baselines são envolvidos em `ImputedClassifier` (mediana do treino
e indicador binário de ausência). Esse tratamento difere do suporte nativo a
ausências dos GBDTs e deve ser mencionado como decisão de protocolo.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pytabkit import CatBoost_TD_Classifier, LGBM_TD_Classifier, XGB_TD_Classifier

from src.models.wrappers import ImputedClassifier


def build_lightgbm(seed: int = 42, **params: Any) -> ImputedClassifier:
    """LightGBM com a configuração padrão TD do pytabkit.

    `params` (opcional) sobrescreve hiperparâmetros de `LGBM_TD_Classifier`; serve
    apenas aos estudos de complexidade por CV interna (por exemplo,
    `src.pipeline.cv_grid.cv_grid`). Os baselines do experimento usam o padrão.
    """
    return ImputedClassifier(LGBM_TD_Classifier(random_state=seed, **params))


def build_xgboost(seed: int = 42, **params: Any) -> ImputedClassifier:
    """XGBoost com a configuração padrão TD do pytabkit.

    `params` (opcional) sobrescreve hiperparâmetros de `XGB_TD_Classifier`; serve
    apenas aos estudos de complexidade por CV interna (por exemplo,
    `src.pipeline.cv_grid.cv_grid`). Os baselines do experimento usam o padrão.
    """
    return ImputedClassifier(XGB_TD_Classifier(random_state=seed, **params))


def build_catboost(seed: int = 42, **params: Any) -> ImputedClassifier:
    """CatBoost com a configuração padrão TD do pytabkit.

    `params` (opcional) sobrescreve hiperparâmetros de `CatBoost_TD_Classifier`; serve
    apenas aos estudos de complexidade por CV interna (por exemplo,
    `src.pipeline.cv_grid.cv_grid`). Os baselines do experimento usam o padrão.
    """
    return ImputedClassifier(CatBoost_TD_Classifier(random_state=seed, **params))


BASELINE_FACTORIES: dict[str, Callable[..., object]] = {
    "lightgbm": build_lightgbm,
    "xgboost": build_xgboost,
    "catboost": build_catboost,
}
