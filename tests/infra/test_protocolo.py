"""Fase de desenvolvimento: cv_grid, tune e run_validation usam somente o treino dos folds externos."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from src.pipeline import cv_grid as cv_grid_module
from src.pipeline import run_validation
from src.pipeline.cv_grid import cv_grid, inner_cv_splits, stratified_subsample
from src.pipeline.run_validation import (
    TRAIN_COLUMNS,
    VALIDATION_COLUMNS,
    done_keys,
    validate_dataset,
)
from src.pipeline.split import stratified_folds
from src.pipeline.tune import tune
from tests import espiao
from tests.dados_overfitting import make_numeric


def _args(tmp_path, **kw):
    base = {
        "seed": 42,
        "output_dir": tmp_path,
        "n_trials": 2,
        "tune_timeout": None,
        "tune_max_samples": None,
        "inner_folds": 3,
        "train_eval_max": 2_000,
    }
    return argparse.Namespace(**{**base, **kw})


@pytest.fixture
def spy_systems(monkeypatch):
    espiao.install(monkeypatch)


def test_cv_grid_delegates_to_validation_curve(monkeypatch):
    """cv_grid prepara subamostra e partições internas e delega o laço a validation_curve."""
    from src.pipeline import overfitting

    captured = {}

    def fake_curve(factory, param, values, X, y, folds, seed=42):
        captured.update(param=param, values=list(values), X=X, y=y, folds=folds, seed=seed)
        return pd.DataFrame({"param": [param], "value": [values[0]], "fold": [0]})

    monkeypatch.setattr(overfitting, "validation_curve", fake_curve)
    X, y = make_numeric(n=400)
    out = cv_grid(object, "max_depth", (1, 2), X, y, cv_folds=4, seed=3, max_samples=100)
    assert list(out["param"]) == ["max_depth"]
    X_sub, y_sub = stratified_subsample(X, y, 100, 3)
    assert captured["X"].index.equals(X_sub.index)
    np.testing.assert_array_equal(captured["y"], y_sub)
    assert captured["values"] == [1, 2] and captured["seed"] == 3
    expected = inner_cv_splits(y_sub, 4, 3)
    assert len(captured["folds"]) == 4
    for (tr, va), (etr, eva) in zip(captured["folds"], expected):
        np.testing.assert_array_equal(tr, etr)
        np.testing.assert_array_equal(va, eva)
    with pytest.raises(ValueError):
        cv_grid(object, "max_depth", [1], X, y, metric="desconhecida")
    assert cv_grid_module.DEFAULT_CV_FOLDS == 3


def test_tune_never_sees_outer_test(spy_systems):
    ds = espiao.spy_dataset()
    train_idx, test_idx = stratified_folds(ds.y, n_folds=3)[0]
    test_labels = set(ds.X.index[test_idx])
    tune(
        espiao.spy_factory,
        espiao.spy_search_space,
        ds.X.iloc[train_idx],
        ds.y[train_idx],
        n_trials=3,
        timeout_s=None,
    )
    assert len(espiao.CALLS) == 3 * 3 * 2  # tentativas x folds internos x (fit, predict)
    assert not espiao.calls_touching(test_labels)


def test_run_validation_never_sees_outer_test(tmp_path, spy_systems):
    ds = espiao.spy_dataset()
    folds = stratified_folds(ds.y, n_folds=3)
    args = _args(tmp_path)
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        espiao.reset()
        validate_dataset(ds, [train_idx], [fold_id], ["lightgbm", "group_tuned"], args)
        seen = espiao.SEEN["fit"] | espiao.SEEN["predict"]
        assert seen and seen <= set(ds.X.index[train_idx])
        assert not espiao.calls_touching(set(ds.X.index[test_idx]))
        # lightgbm: 3 folds internos x (ajuste, validação, treino); group_tuned: + busca
        assert sum(c[1] == "lightgbm" for c in espiao.CALLS) == 3 * 3


def test_run_validation_main_with_patched_loaders(tmp_path, monkeypatch, spy_systems):
    """main() com load_task e get_folds substituídos: nenhum índice de teste é usado."""
    import data.load_tabarena as lt

    ds = espiao.spy_dataset(n=300)
    # Linhas sentinela: estão no teste de TODOS os folds externos e em nenhum treino.
    sentinel = np.arange(30)
    rest = np.arange(30, len(ds.y))
    outer = [
        (rest[tr], np.concatenate([sentinel, rest[te]]))
        for tr, te in stratified_folds(ds.y[rest], n_folds=3)
    ]
    requested = []

    def fake_get_folds(task_id, folds=(0, 1, 2), repeat=0):
        requested.append(list(folds))
        return [outer[f] for f in folds]

    monkeypatch.setattr(run_validation, "load_task", lambda task_id: ds)
    monkeypatch.setattr(lt, "get_folds", fake_get_folds)
    out = tmp_path / "results"
    argv = ["--task-ids", "9999", "--models", "lightgbm", "group_tuned", "--n-trials", "2"]
    argv += ["--output-dir", str(out), "--tune-timeout", "60"]
    run_validation.main(argv, root=tmp_path)
    assert requested == [[0, 1, 2]]
    assert espiao.CALLS
    assert not espiao.calls_touching(set(ds.X.index[sentinel]))
    train_sets = [set(ds.X.index[tr]) for tr, _ in outer]
    assert all(any(c[2] <= s for s in train_sets) for c in espiao.CALLS)
    val = pd.read_csv(out / "validation.csv")
    assert len(val) == 2 * 3 * 3 and (val["status"] == "ok").all()
    assert len(list((out / "preds_val").glob("*.npz"))) == 2 * 3 * 3
    assert not (out / "preds").exists() and not (out / "test.csv").exists()


def test_run_validation_saves_validation_predictions(tmp_path, spy_systems):
    ds = espiao.spy_dataset()
    train_idx = stratified_folds(ds.y, n_folds=3)[0][0]
    validate_dataset(ds, [train_idx], [0], ["lightgbm"], _args(tmp_path))
    files = sorted((tmp_path / "preds_val").glob("*.npz"))
    assert [f.name for f in files] == [f"9999_fold0_inner{i}_lightgbm.npz" for i in range(3)]
    total = 0
    for f in files:
        z = np.load(f)
        assert z["y_proba"].shape == (len(z["y_true"]), len(z["classes"]))
        total += len(z["y_true"])
    assert total == len(train_idx)  # cada instância do treino validada uma vez


def test_run_validation_writes_validation_and_train_metrics(tmp_path, spy_systems):
    ds = espiao.spy_dataset()
    train_folds = [tr for tr, _ in stratified_folds(ds.y, n_folds=3)]
    args = _args(tmp_path, train_eval_max=50)
    validate_dataset(ds, train_folds, [0, 1, 2], ["lightgbm", "group_tuned"], args)
    val = pd.read_csv(tmp_path / "validation.csv")
    train = pd.read_csv(tmp_path / "validation_train.csv")
    assert list(val.columns) == VALIDATION_COLUMNS
    assert list(train.columns) == TRAIN_COLUMNS
    assert len(val) == len(train) == 2 * 3 * 3  # sistemas x folds externos x folds internos
    assert (val["status"] == "ok").all() and (train["status"] == "ok").all()
    assert (train["n_train_eval"] <= 50).all()
    tuned = val[val["model"] == "group_tuned"]
    assert tuned["optimistic"].all() and tuned["tune_best_score"].notna().all()
    assert not val.loc[val["model"] == "lightgbm", "optimistic"].any()
    keys = ["task_id", "dataset", "fold", "inner_fold", "model"]
    assert val[keys].equals(train[keys])
    done = done_keys(tmp_path / "validation.csv")
    assert validate_dataset(ds, train_folds, [0, 1, 2], ["lightgbm"], args, done) == []


def test_run_validation_records_failures(tmp_path, monkeypatch):
    from src.models import group_model

    def broken(seed=42, **params):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr(group_model, "build_group_model", broken)
    ds = espiao.spy_dataset()
    train_folds = [tr for tr, _ in stratified_folds(ds.y, n_folds=3)][:1]
    validate_dataset(ds, train_folds, [0], ["group_default"], _args(tmp_path))
    val = pd.read_csv(tmp_path / "validation.csv")
    train = pd.read_csv(tmp_path / "validation_train.csv")
    assert list(val["status"]) == ["error"] and list(train["status"]) == ["error"]
    assert "falha simulada" in val["error"].iloc[0]


def test_run_validation_refuses_after_final_test(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "test.csv").write_text("task_id\n1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="avaliação final"):
        run_validation.main(["--task-ids", "9999", "--output-dir", str(out)], root=tmp_path)


def test_run_validation_with_real_baseline(tmp_path):
    ds = espiao.spy_dataset(n=200, n_classes=2)
    train_folds = [stratified_folds(ds.y, n_folds=3)[0][0]]
    validate_dataset(ds, train_folds, [0], ["lightgbm"], _args(tmp_path))
    train = pd.read_csv(tmp_path / "validation_train.csv")
    val = pd.read_csv(tmp_path / "validation.csv")
    assert (val["status"] == "ok").all()
    assert train["auc_ovo"].between(0, 1).all() and val["auc_ovo"].between(0, 1).all()
