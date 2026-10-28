"""Intervalos de confiança por bootstrap sobre as predições de teste salvas por `final_test`.

As predições em `results/preds/` são geradas uma única vez, na avaliação final
(`src.pipeline.final_test`), após o congelamento do protocolo. O intervalo é
calculado por dataset e fold externo, reamostrando com reposição as instâncias
do teste desse fold (bootstrap percentil). Ele expressa a incerteza da métrica
devida à amostra de teste; a variabilidade entre folds externos e entre
datasets é tratada em `demsar.py`. É uma análise descritiva do resultado final
e não deve retroalimentar escolhas do modelo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.evaluate import compute_metrics

DEFAULT_N_BOOT = 1000


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lê um arquivo .npz de `results/preds/` e devolve (y_true, y_proba, classes)."""
    data = np.load(path)
    return data["y_true"], data["y_proba"], data["classes"]


def bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    metric: str = "auc_ovo",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Estimativa pontual e intervalo percentil (1 - alpha) da métrica.

    Reamostra com reposição as n instâncias de teste `n_boot` vezes, calcula a
    métrica em cada reamostra e devolve os quantis alpha/2 e 1 - alpha/2.
    Reamostras com uma única classe (métrica indefinida) são descartadas e o
    número efetivo de reamostras é informado em `n_boot`.

    Returns:
        dict com chaves estimate, ci_low, ci_high e n_boot.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    classes = np.asarray(classes)
    n = len(y_true)

    orig_metrics = compute_metrics(y_true, y_proba, classes)
    estimate = float(orig_metrics[metric])

    rng = np.random.default_rng(seed)
    boot_values: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_sample = y_true[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        m = compute_metrics(y_sample, y_proba[idx], classes)[metric]
        if np.isnan(m):
            continue
        boot_values.append(float(m))

    if not boot_values:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": 0,
        }

    q_low = 100.0 * (alpha / 2.0)
    q_high = 100.0 * (1.0 - alpha / 2.0)
    ci_low = float(np.percentile(boot_values, q_low))
    ci_high = float(np.percentile(boot_values, q_high))

    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boot": len(boot_values),
    }


def bootstrap_table(
    preds_dir: Path = Path("results/preds"),
    metric: str = "auc_ovo",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
) -> pd.DataFrame:
    """Intervalos para todos os arquivos de predição: uma linha por dataset, fold externo e sistema."""
    rows = []
    for path in sorted(Path(preds_dir).glob("*.npz")):
        task_id, fold, model = path.stem.split("_", 2)
        ci = bootstrap_ci(*load_predictions(path), metric=metric, n_boot=n_boot, seed=seed)
        rows.append(
            {"task_id": int(task_id), "fold": int(fold.removeprefix("fold")), "model": model, **ci}
        )
    return pd.DataFrame(rows)
