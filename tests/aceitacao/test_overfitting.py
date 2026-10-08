"""Aceitação da análise de overfitting por CV interna (src/pipeline/overfitting.py e gráficos)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

from src.pipeline.class_errors import confusion_table, recall_per_class
from src.pipeline.cv_grid import cv_grid, inner_cv_splits
from src.pipeline.evaluate import METRICS, compute_metrics
from src.pipeline.overfitting import generalization_gap, select_complexity, validation_curve
from src.pipeline.split import stratified_folds
from src.reports.plots_overfitting import (
    plot_confusion,
    plot_generalization_gap,
    plot_validation_curve,
)
from tests import espiao
from tests.dados_overfitting import complexity_curve, make_numeric, validation_pairs

matplotlib.use("Agg")


def _tree(seed=42, max_depth=None):
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(max_depth=max_depth, random_state=seed)


def test_generalization_gap_higher_is_better():
    val, train = validation_pairs()
    out = generalization_gap(val, train, "auc_ovo", higher_is_better=True)
    assert list(out.columns) == [
        "model",
        "n_folds",
        "train_mean",
        "val_mean",
        "gap_mean",
        "gap_std",
    ]
    assert list(out["model"]) == ["a", "b"]  # ordenado por gap_mean decrescente
    a = out.set_index("model").loc["a"]
    assert a["n_folds"] == 2
    assert a["train_mean"] == pytest.approx(0.85)
    assert a["val_mean"] == pytest.approx(0.7)
    assert a["gap_mean"] == pytest.approx(0.15)
    assert a["gap_std"] == pytest.approx(np.std([0.2, 0.1], ddof=1))
    b = out.set_index("model").loc["b"]
    assert b["gap_mean"] == pytest.approx(-0.05)
    assert np.isnan(b["gap_std"])


def test_generalization_gap_lower_is_better():
    val, train = validation_pairs()
    out = generalization_gap(val, train, "cross_entropy", higher_is_better=False).set_index("model")
    # validação menos treino: a -> (0.3, 0.2), b -> -0.1
    assert out.loc["a", "gap_mean"] == pytest.approx(0.25)
    assert out.loc["b", "gap_mean"] == pytest.approx(-0.1)
    assert list(out.index) == ["a", "b"]


def test_validation_curve_matches_manual_computation():
    X, y = make_numeric(n=240)
    folds = inner_cv_splits(y, n_folds=3, seed=0)
    curve = validation_curve(_tree, "max_depth", [1, 3], X, y, folds, seed=7)
    assert len(curve) == 6
    assert list(curve["value"]) == [1, 1, 1, 3, 3, 3]
    assert list(curve["fold"]) == [0, 1, 2, 0, 1, 2]
    for m in METRICS:
        assert {f"{m}_train", f"{m}_val"} <= set(curve.columns)
    row = curve.iloc[4]  # max_depth = 3, fold 1
    tr, va = folds[1]
    est = _tree(7, max_depth=3).fit(X.iloc[tr], y[tr])
    exp_tr = compute_metrics(y[tr], est.predict_proba(X.iloc[tr]), est.classes_)
    exp_va = compute_metrics(y[va], est.predict_proba(X.iloc[va]), est.classes_)
    assert row["param"] == "max_depth"
    assert row["n_train"] == len(tr) and row["n_val"] == len(va)
    for m in METRICS:
        assert row[f"{m}_train"] == pytest.approx(exp_tr[m])
        assert row[f"{m}_val"] == pytest.approx(exp_va[m])


@pytest.mark.parametrize(
    ("rule", "simpler_is_lower", "expected"),
    [("best", True, 3), ("1se", True, 2), ("1se", False, 4), ("best", False, 3)],
)
def test_select_complexity_higher_is_better(rule, simpler_is_lower, expected):
    curve = complexity_curve()
    chosen = select_complexity(curve, "auc_ovo", rule=rule, simpler_is_lower=simpler_is_lower)
    assert chosen == expected


@pytest.mark.parametrize(
    ("rule", "simpler_is_lower", "expected"),
    [("best", True, 3), ("1se", True, 2), ("1se", False, 4)],
)
def test_select_complexity_lower_is_better(rule, simpler_is_lower, expected):
    curve = complexity_curve("cross_entropy", lower=True)
    chosen = select_complexity(
        curve,
        "cross_entropy",
        rule=rule,
        simpler_is_lower=simpler_is_lower,
        higher_is_better=False,
    )
    assert chosen == expected


def test_select_complexity_tie_goes_to_simpler():
    curve = pd.DataFrame(
        {"value": [5, 5, 2, 2], "fold": [0, 1, 0, 1], "auc_ovo_val": [0.9, 0.8, 0.8, 0.9]}
    )
    assert select_complexity(curve, rule="best") == 2
    assert select_complexity(curve, rule="best", simpler_is_lower=False) == 5


def test_validation_curve_and_select_complexity_require_two_folds():
    X, y = make_numeric(n=120)
    folds = inner_cv_splits(y, n_folds=3, seed=0)
    with pytest.raises(ValueError):
        validation_curve(_tree, "max_depth", [1, 2], X, y, folds[:1])
    curve = complexity_curve()
    with pytest.raises(ValueError):
        select_complexity(curve[curve["fold"] == 0], rule="1se")


class _EstimatorError(Exception):
    """Erro do estimador (não deriva de NotImplementedError)."""


def test_validation_curve_propagates_estimator_errors():
    X, y = make_numeric(n=120)

    def broken(seed=42, max_depth=None):
        raise _EstimatorError("estimador inválido")

    with pytest.raises(_EstimatorError):
        validation_curve(broken, "max_depth", [1], X, y, inner_cv_splits(y, 3, 0))


# ------------------------------------------------------------------ cv_grid (delega a validation_curve)


def test_cv_grid_columns_and_train_vs_validation():
    X, y = make_numeric(n=400)
    depths = [1, 2, 4, 8, 16]
    grid = cv_grid(_tree, "max_depth", depths, X, y, cv_folds=3)
    expected = ["param", "value", "fold", "n_train", "n_val"]
    expected += [f"{m}_{s}" for m in METRICS for s in ("train", "val")]
    assert set(expected) <= set(grid.columns)
    assert len(grid) == len(depths) * 3
    assert list(grid["value"].unique()) == depths
    means = grid.groupby("value", sort=False)[["auc_ovo_train", "auc_ovo_val"]].mean()
    assert np.all(np.diff(means["auc_ovo_train"].to_numpy()) >= -1e-12)
    assert (means["auc_ovo_train"] >= means["auc_ovo_val"]).all()
    assert means["auc_ovo_train"].iloc[-1] - means["auc_ovo_val"].iloc[-1] > 0.1


def test_cv_grid_subsamples_large_training_sets():
    X, y = make_numeric(n=400)
    grid = cv_grid(_tree, "max_depth", [2], X, y, cv_folds=2, max_samples=100)
    assert (grid["n_train"] + grid["n_val"] == 100).all()


def test_cv_grid_never_sees_outer_test(monkeypatch):
    espiao.install(monkeypatch)
    ds = espiao.spy_dataset()
    train_idx, test_idx = stratified_folds(ds.y, n_folds=3)[0]
    cv_grid(espiao.spy_factory, "max_depth", [1, 2], ds.X.iloc[train_idx], ds.y[train_idx])
    assert len(espiao.CALLS) == 2 * 3 * 3  # valores x folds x (ajuste, treino, validação)
    seen = espiao.SEEN["fit"] | espiao.SEEN["predict"]
    assert seen <= set(ds.X.index[train_idx])
    assert not espiao.calls_touching(set(ds.X.index[test_idx]))


# ------------------------------------------------------------------ erros por classe


def test_confusion_and_recall_by_hand():
    classes = np.array(["a", "b", "c"])
    y_true = np.array(["a", "a", "b", "b", "c"])
    proba = np.array(
        [
            [0.8, 0.1, 0.1],  # a -> a
            [0.2, 0.7, 0.1],  # a -> b
            [0.1, 0.8, 0.1],  # b -> b
            [0.1, 0.3, 0.6],  # b -> c
            [0.1, 0.1, 0.8],  # c -> c
        ]
    )
    counts = confusion_table(y_true, proba, classes)
    assert counts.to_numpy().tolist() == [[1, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert list(counts.index) == ["a", "b", "c"] and list(counts.columns) == ["a", "b", "c"]
    rates = confusion_table(y_true, proba, classes, normalize=True)
    np.testing.assert_allclose(rates.loc["a"].to_numpy(), [0.5, 0.5, 0.0])
    recall = recall_per_class(y_true, proba, classes).set_index("class")
    assert recall["support"].tolist() == [2, 2, 1]
    np.testing.assert_allclose(recall["recall"].to_numpy(), [0.5, 0.5, 1.0])


# ------------------------------------------------------------------ gráficos


def test_overfitting_figures(tmp_path):
    curve = complexity_curve()
    fig = plot_validation_curve(curve, "auc_ovo", output=tmp_path / "curva.png")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes[0].get_lines()) >= 2
    assert (tmp_path / "curva.png").exists()
    val, train = validation_pairs()
    fig = plot_generalization_gap(generalization_gap(val, train), output=tmp_path / "gap.png")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "gap.png").exists()
    classes = np.array(["a", "b"])
    table = confusion_table(np.array(["a", "b"]), np.array([[0.9, 0.1], [0.6, 0.4]]), classes)
    assert plot_confusion(table, output=tmp_path / "confusao.png") is not None
    assert (tmp_path / "confusao.png").exists()


def test_validation_curve_band_is_one_standard_error():
    """A faixa da curva de validação é média ± 1 EP (desvio / raiz do número de folds)."""
    curve = complexity_curve()
    fig = plot_validation_curve(curve, "auc_ovo")
    val_band = fig.axes[0].collections[1]  # treino, depois validação
    ys = np.concatenate([p.vertices[:, 1] for p in val_band.get_paths()])
    se = 0.01 / np.sqrt(3)
    assert ys.max() == pytest.approx(0.88 + se, abs=1e-9)
    assert ys.min() == pytest.approx(0.80 - se, abs=1e-9)
