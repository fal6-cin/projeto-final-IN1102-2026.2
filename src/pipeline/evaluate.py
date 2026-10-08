"""Cálculo das métricas exigidas: AUC OvO, acurácia, G-Mean, entropia cruzada e tempos.

Todas as métricas são derivadas de `predict_proba`. A predição de classe é o
`argmax` das probabilidades, alinhado a `estimator.classes_`, de modo que a
inferência é executada uma única vez e os rótulos nunca ficam desalinhados das
colunas de probabilidade.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

METRICS = ["auc_ovo", "accuracy", "g_mean", "cross_entropy"]
# Direção de cada métrica: True quando maior é melhor.
HIGHER_IS_BETTER = {
    "auc_ovo": True,
    "accuracy": True,
    "g_mean": True,
    "cross_entropy": False,
    "tune_time_s": False,
    "fit_time_s": False,
    "predict_time_s": False,
    "total_time_s": False,
}


@dataclass
class EvaluationResult:
    auc_ovo: float
    accuracy: float
    g_mean: float
    cross_entropy: float
    tune_time_s: float
    fit_time_s: float
    predict_time_s: float
    classes: np.ndarray = field(repr=False, default=None)
    y_proba: np.ndarray = field(repr=False, default=None)

    def to_dict(self) -> dict[str, float]:
        return {
            "auc_ovo": self.auc_ovo,
            "accuracy": self.accuracy,
            "g_mean": self.g_mean,
            "cross_entropy": self.cross_entropy,
            "tune_time_s": self.tune_time_s,
            "fit_time_s": self.fit_time_s,
            "predict_time_s": self.predict_time_s,
            "total_time_s": self.tune_time_s + self.fit_time_s + self.predict_time_s,
        }


def g_mean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """G-Mean: média geométrica do recall por classe (0 se alguma classe tiver recall 0)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = [float((y_pred[y_true == c] == c).mean()) for c in np.unique(y_true)]
    if not recalls or min(recalls) == 0.0:
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    """Métricas a partir das probabilidades; colunas de `y_proba` seguem `classes`."""
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    y_proba = np.asarray(y_proba, dtype=float)
    # Renormaliza as linhas: alguns modelos devolvem probabilidades em float32
    # cuja soma difere de 1 por erro de arredondamento.
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    y_pred = classes[np.argmax(y_proba, axis=1)]

    # Em uma partição de teste podem faltar classes; a AUC OvO usa apenas as presentes.
    present = np.isin(classes, np.unique(y_true))
    if present.sum() < 2:
        auc = float("nan")
    elif present.sum() == 2:
        proba = y_proba[:, present]
        positive = classes[present][1]
        auc = roc_auc_score(y_true == positive, proba[:, 1] / proba.sum(axis=1))
    else:
        proba = y_proba[:, present]
        proba = proba / proba.sum(axis=1, keepdims=True)
        auc = roc_auc_score(y_true, proba, multi_class="ovo", labels=classes[present])

    return {
        "auc_ovo": float(auc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "g_mean": g_mean_score(y_true, y_pred),
        "cross_entropy": float(log_loss(y_true, y_proba, labels=classes)),
    }


def fit_predict_evaluate(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    tune_time_s: float = 0.0,
) -> EvaluationResult:
    """Treina e avalia um estimador, medindo tempos de treino e de predição.

    Args:
        tune_time_s: tempo gasto na busca de hiperparâmetros (medido fora desta
            função), somado ao tempo total reportado.
    """
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            f"{type(estimator).__name__} não implementa predict_proba; "
            "as métricas AUC OvO e entropia cruzada exigem probabilidades."
        )

    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_proba = np.asarray(estimator.predict_proba(X_test))
    predict_time_s = time.perf_counter() - t0

    classes = np.asarray(getattr(estimator, "classes_", np.unique(y_train)))
    metrics = compute_metrics(y_test, y_proba, classes)
    return EvaluationResult(
        **metrics,
        tune_time_s=float(tune_time_s),
        fit_time_s=fit_time_s,
        predict_time_s=predict_time_s,
        classes=classes,
        y_proba=y_proba,
    )
