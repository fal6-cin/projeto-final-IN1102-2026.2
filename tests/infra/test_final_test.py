"""Avaliação final: congelamento, integridade via git, guardas, retomada e isolamento do teste."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import final_test, freeze
from src.pipeline.integrity import is_locked_path, locked_hashes, sha256_bytes
from src.pipeline.results import per_dataset
from src.pipeline.split import stratified_folds
from tests import espiao
from tests.dados_sinteticos import raw_results


@pytest.fixture
def repo(tmp_path):
    """Repositório git mínimo com arquivos travados e seleção com uma task sintética."""
    return espiao.make_repo(tmp_path)


@pytest.fixture
def spy_systems(monkeypatch):
    espiao.install(monkeypatch)


def _loader():
    ds = espiao.spy_dataset()
    folds = stratified_folds(ds.y, n_folds=3)
    return ds, folds, (lambda task_id, fold_ids: (ds, [folds[f] for f in fold_ids]))


def _freeze(root, models=("lightgbm",), extra=()):
    argv = ["--output-dir", str(root / "results"), "--skip-validation", "--n-trials", "2"]
    return freeze.main(argv + ["--models", *models, *extra], root=root)


def _freeze_and_commit(root, models=("lightgbm",)):
    _freeze(root, models)
    espiao.commit_all(root, "Congela o protocolo", "results/protocolo.lock")


def _run(root, loader, *extra):
    return final_test.main(
        ["--output-dir", str(root / "results"), *extra], root=root, loader=loader
    )


def test_locked_paths_rule():
    assert is_locked_path("src/models/group_model.py")
    assert is_locked_path("src/pipeline/sub/x.py")
    assert is_locked_path("data/load_tabarena.py")
    assert is_locked_path("data/selecao_datasets.txt")
    assert not is_locked_path("data/tabarena_v01_classificacao.csv")
    assert not is_locked_path("data/sub/x.py")
    assert not is_locked_path("notebooks/x.py")
    assert sha256_bytes(b"a\r\nb\n") == sha256_bytes(b"a\nb\n")


def test_freeze_records_all_versioned_files(repo, spy_systems):
    lock = json.loads(_freeze(repo).read_text(encoding="utf-8"))
    assert set(lock["files"]) == {
        "data/load_tabarena.py",
        "data/selecao_datasets.txt",
        "src/models/group_model.py",
        "src/models/wrappers.py",
        "src/pipeline/evaluate.py",
    }
    assert lock["files"] == locked_hashes(repo)
    assert "max_depth" in lock["search_space"]["sample"]
    assert lock["budget"]["n_trials"] == 2 and lock["seed"] == 42
    assert lock["task_ids"] == [9999] and lock["models"] == ["lightgbm"]
    assert lock["git_commit"] and freeze.verify_lock(lock, repo) == []
    with pytest.raises(SystemExit):
        _freeze(repo)  # lock existente exige --force
    _freeze(repo, extra=["--force"])


def test_freeze_requires_git(tmp_path, spy_systems):
    root = tmp_path / "semgit"
    (root / "data").mkdir(parents=True)
    (root / "data" / "selecao_datasets.txt").write_text("9999\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="git"):
        _freeze(root)


def test_final_test_requires_committed_lock_and_runs_once(repo, spy_systems):
    _, _, loader = _loader()
    out = repo / "results"
    with pytest.raises(SystemExit, match="protocolo.lock"):
        _run(repo, loader)
    _freeze(repo)
    with pytest.raises(SystemExit, match="versionado"):
        _run(repo, loader)  # lock ainda não comitado
    espiao.commit_all(repo, "Congela o protocolo", "results/protocolo.lock")
    head = espiao.git(repo, "rev-parse", "HEAD")
    rows = _run(repo, loader)
    assert len(rows) == 3
    test = pd.read_csv(out / "test.csv")
    assert list(test.columns) == final_test.CSV_COLUMNS
    assert (test["status"] == "ok").all()
    assert len(list((out / "preds").glob("*.npz"))) == 3
    log = pd.read_csv(out / "test_log.csv")
    assert list(log.columns) == final_test.LOG_COLUMNS
    assert list(log["event"]) == ["recusa", "inicio", "sistema", "sistema", "sistema", "fim"]
    assert set(log.loc[log["event"] != "recusa", "git_head"]) == {head}
    manifest = json.loads((out / "test_manifest.json").read_text(encoding="utf-8"))
    assert manifest["lock_sha256"] == freeze.lock_digest(out / freeze.LOCK_FILE)
    assert [r["git_head"] for r in manifest["runs"]] == [head]
    assert set(manifest["status"].values()) == {"ok"} and len(manifest["status"]) == 3
    assert set(manifest["rows"]) == set(manifest["status"])
    assert all(len(v) == 64 for v in manifest["rows"].values())
    assert b"\r\n" not in (out / "test.csv").read_bytes()
    assert b"\r\n" not in (out / "test_manifest.json").read_bytes()

    with pytest.raises(SystemExit, match="única"):
        _run(repo, loader)
    with pytest.raises(SystemExit, match="recongelado"):
        _freeze(repo, extra=["--force"])
    with pytest.raises(SystemExit, match="nada a retomar"):
        _run(repo, loader, "--resume")
    log = pd.read_csv(out / "test_log.csv")
    assert (log["event"] == "inicio").sum() == 1  # recusas não contam como execução


@pytest.mark.parametrize("removed", ["test.csv", "test_manifest.json", "preds"])
def test_final_test_refuses_after_deleting_some_traces(repo, spy_systems, removed):
    _, _, loader = _loader()
    _freeze_and_commit(repo)
    _run(repo, loader)
    target = repo / "results" / removed
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    with pytest.raises(SystemExit, match="vestígios"):
        _run(repo, loader)


def test_final_test_refuses_when_history_has_results(repo, spy_systems):
    _, _, loader = _loader()
    _freeze_and_commit(repo)
    _run(repo, loader)
    espiao.commit_all(repo, "Avaliação final", "results")
    for name in ("test.csv", "test_log.csv", "test_manifest.json"):
        (repo / "results" / name).unlink()
    shutil.rmtree(repo / "results" / "preds")
    with pytest.raises(SystemExit, match="histórico git"):
        _run(repo, loader)


def test_final_test_refuses_uncommitted_or_changed_files(repo, spy_systems):
    _, _, loader = _loader()
    _freeze_and_commit(repo)
    path = repo / "src" / "models" / "group_model.py"
    original = path.read_text(encoding="utf-8")
    path.write_text(original + "\n# alterado\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="não comitadas"):
        _run(repo, loader)
    espiao.commit_all(repo, "Altera o modelo depois do congelamento")
    with pytest.raises(SystemExit, match="group_model.py"):
        _run(repo, loader)  # comitado, mas o hash diverge do lock
    path.write_text(original, encoding="utf-8")
    espiao.commit_all(repo, "Restaura o modelo")
    (repo / "src" / "models" / "novo.py").write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="novo.py"):
        _run(repo, loader)  # arquivo novo não versionado
    espiao.commit_all(repo, "Arquivo novo")
    with pytest.raises(SystemExit, match="novo.py"):
        _run(repo, loader)  # versionado, mas fora do congelamento
    assert not (repo / "results" / "test.csv").exists()


def test_final_test_refuses_lock_different_from_head(repo, spy_systems):
    _, _, loader = _loader()
    _freeze_and_commit(repo)
    lock_path = repo / "results" / freeze.LOCK_FILE
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["seed"] = 7
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(SystemExit, match="HEAD"):
        _run(repo, loader)


def test_final_test_records_failures_and_resumes_only_errors(repo, spy_systems, monkeypatch):
    from src.models import group_model

    def broken(seed=42, **params):
        raise RuntimeError("falha simulada")

    _, _, loader = _loader()
    _freeze_and_commit(repo, models=("lightgbm", "group_default"))
    monkeypatch.setattr(group_model, "build_group_model", broken)
    _run(repo, loader)
    out = repo / "results"
    test = pd.read_csv(out / "test.csv")
    assert len(test) == 6
    assert (test.loc[test["model"] == "group_default", "status"] == "error").all()

    monkeypatch.setattr(group_model, "build_group_model", espiao.spy_factory)
    espiao.reset()
    rows = _run(repo, loader, "--resume")
    assert {r["model"] for r in rows} == {"group_default"} and len(rows) == 3
    assert {c[1] for c in espiao.CALLS} == {"spy"}  # lightgbm (ok) não foi refeito
    assert len(final_test.done_keys(out / "test.csv")) == 6
    test = pd.read_csv(out / "test.csv")
    assert len(test) == 6 and (test["status"] == "ok").all()  # linha de erro substituída
    log = pd.read_csv(out / "test_log.csv")
    starts = log.loc[log["event"] == "inicio", "detail"].tolist()
    assert len(starts) == 2 and "resume=True" in starts[1]


def test_resume_refuses_missing_or_altered_rows(repo, spy_systems):
    _, _, loader = _loader()
    _freeze_and_commit(repo)
    _run(repo, loader)
    path = repo / "results" / "test.csv"
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    path.write_text("".join(lines[:-1]), encoding="utf-8")  # linha concluída removida
    with pytest.raises(SystemExit, match="ausente ou alterada"):
        _run(repo, loader, "--resume")
    path.write_text(original, encoding="utf-8")
    frame = pd.read_csv(path)
    frame.loc[0, "auc_ovo"] = 0.999  # linha concluída alterada
    frame.to_csv(path, index=False)
    with pytest.raises(SystemExit, match="ausente ou alterada"):
        _run(repo, loader, "--resume")
    path.write_text(original, encoding="utf-8")
    with pytest.raises(SystemExit, match="nada a retomar"):
        _run(repo, loader, "--resume")


def test_final_test_fits_on_train_and_predicts_each_test_once(tmp_path, spy_systems):
    ds, folds, loader = _loader()
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        repo = espiao.make_repo(tmp_path / f"fold{fold_id}")
        _freeze(repo, models=("lightgbm", "group_tuned"), extra=["--folds", str(fold_id)])
        espiao.commit_all(repo, "Congela o protocolo", "results/protocolo.lock")
        espiao.reset()
        _run(repo, loader)
        train_labels = set(ds.X.index[train_idx])
        test_labels = set(ds.X.index[test_idx])
        assert not espiao.calls_touching(test_labels, kind="fit")
        for tag in ("lightgbm", "group"):
            touching = [c for c in espiao.calls_touching(test_labels) if c[1] == tag]
            assert len(touching) == 1, tag  # exatamente uma predição no teste
            assert touching[0][0] == "predict" and touching[0][2] == test_labels
        fits = [c for c in espiao.CALLS if c[0] == "fit" and c[2] == train_labels]
        assert {c[1] for c in fits} == {"lightgbm", "group"}


def test_per_dataset_keeps_only_complete_datasets():
    raw = raw_results(n_datasets=5)
    raw = raw.drop(raw[(raw["task_id"] == 1000) & (raw["fold"] == 2)].index[:1])
    table = per_dataset(raw)
    assert 1000 not in table.index
    assert len(table) == 4
    assert table.shape[1] == 4


def test_results_default_paths():
    from src.pipeline import results

    assert results.DEFAULT_RAW == Path("results/test.csv")
    assert results.DEFAULT_VALIDATION == Path("results/validation.csv")
