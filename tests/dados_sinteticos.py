"""Dados sintéticos compartilhados pelos testes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_dataset(n: int = 300, n_classes: int = 2, seed: int = 0):
    """Dataset com atributo categórico, valores ausentes e rótulos de texto."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": pd.Categorical(rng.choice(["x", "y", "z"], size=n)),
        }
    )
    X.loc[rng.random(n) < 0.1, "a"] = np.nan
    score = X["b"].to_numpy() + (X["c"] == "x").to_numpy() + rng.normal(scale=0.5, size=n)
    edges = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.array([f"classe_{i}" for i in np.digitize(score, edges)])
    return X, y


def split(X, y, frac: float = 0.7):
    n = int(len(y) * frac)
    return X.iloc[:n], y[:n], X.iloc[n:], y[n:]


def raw_results(n_datasets: int = 8, seed: int = 0, task_ids=None) -> pd.DataFrame:
    """Resultados sintéticos no formato de test.csv (3 folds, 4 sistemas)."""
    rng = np.random.default_rng(seed)
    ids = task_ids if task_ids is not None else list(range(1000, 1000 + n_datasets))
    rows = []
    for t in ids:
        base = rng.uniform(0.7, 0.9)
        for fold in range(3):
            for model, delta in [
                ("group_tuned", 0.03),
                ("lightgbm", 0.0),
                ("xgboost", -0.01),
                ("catboost", 0.01),
            ]:
                rows.append(
                    {
                        "task_id": t,
                        "dataset": f"ds{t}",
                        "fold": fold,
                        "model": model,
                        "auc_ovo": base + delta + rng.normal(0, 0.003),
                        "accuracy": 0.8,
                        "g_mean": 0.7,
                        "cross_entropy": 0.5 - delta,
                        "tune_time_s": 0.0,
                        "fit_time_s": rng.uniform(1, 10),
                        "predict_time_s": 0.1,
                        "total_time_s": rng.uniform(1, 10),
                        "status": "ok",
                        "error": "",
                    }
                )
    return pd.DataFrame(rows)
