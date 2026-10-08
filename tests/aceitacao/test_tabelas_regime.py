"""Aceitação das tabelas-resumo, da análise por regime e dos gráficos."""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

from src.pipeline.regime import aggregate_by_regime, rank_by_regime
from src.reports.plots import boxplot_by_model, cost_vs_performance
from src.reports.results_table import per_dataset_table, summary_by_model
from tests.dados_sinteticos import raw_results

matplotlib.use("Agg")


def _table_and_meta():
    table = pd.DataFrame(
        {"a": [0.9, 0.8, 0.7, 0.6], "b": [0.8, 0.9, 0.6, 0.5]},
        index=pd.Index([1, 2, 3, 4], name="task_id"),
    )
    meta = pd.DataFrame(
        {"task_id": [1, 2, 3, 4], "regime_size": ["small", "small", "large", "large"]}
    )
    return table, meta


def test_aggregate_by_regime_values():
    table, meta = _table_and_meta()
    out = aggregate_by_regime(table, meta, "regime_size").set_index(["regime_size", "model"])
    assert out.loc[("small", "a"), "mean"] == pytest.approx(0.85)
    assert out.loc[("large", "b"), "mean"] == pytest.approx(0.55)
    assert out.loc[("small", "a"), "n_datasets"] == 2
    assert out.loc[("small", "a"), "std"] == pytest.approx(0.0707107, abs=1e-6)  # ddof = 1


def test_rank_by_regime_values():
    table, meta = _table_and_meta()
    out = rank_by_regime(table, meta, "regime_size").set_index(["regime_size", "model"])
    assert out.loc[("small", "a"), "mean_rank"] == pytest.approx(1.5)
    assert out.loc[("large", "a"), "mean_rank"] == pytest.approx(1.0)
    assert {"std_rank", "n_datasets"} <= set(out.columns)


def test_summary_averages_folds_before_datasets():
    raw = raw_results(n_datasets=2)
    # Dataset 1000 com um único fold: a média por dataset difere da média agrupada.
    raw = raw.drop(raw[(raw["task_id"] == 1000) & (raw["fold"] > 0)].index)
    summary = summary_by_model(raw, metrics=["auc_ovo"]).set_index("model")
    expected = raw.groupby(["model", "task_id"])["auc_ovo"].mean().groupby("model").mean()
    for model, value in expected.items():
        assert summary.loc[model, "auc_ovo_mean"] == pytest.approx(value)


def test_per_dataset_table_format():
    table = per_dataset_table(raw_results(n_datasets=2))
    assert table.shape == (2, 4)
    assert "±" in table.iloc[0, 0]


def test_plots_return_figures(tmp_path):
    raw = raw_results(n_datasets=6)
    table = raw.pivot_table(index="task_id", columns="model", values="auc_ovo")
    assert boxplot_by_model(table, output=tmp_path / "box.png") is not None
    assert cost_vs_performance(raw, output=tmp_path / "custo.png") is not None
    assert (tmp_path / "box.png").exists()
    assert (tmp_path / "custo.png").exists()
