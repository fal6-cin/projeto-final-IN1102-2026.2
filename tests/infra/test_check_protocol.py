"""check_protocol: verificação do protocolo pelo histórico git."""

from __future__ import annotations

import json

import pytest

from src.pipeline import check_protocol, final_test, freeze
from src.pipeline.split import stratified_folds
from tests import espiao


@pytest.fixture
def repo(tmp_path, monkeypatch):
    espiao.install(monkeypatch)
    return espiao.make_repo(tmp_path)


def _loader():
    ds = espiao.spy_dataset()
    folds = stratified_folds(ds.y, n_folds=3)
    return lambda task_id, fold_ids: (ds, [folds[f] for f in fold_ids])


def _freeze(root):
    argv = ["--output-dir", str(root / "results"), "--skip-validation", "--n-trials", "2"]
    freeze.main(argv + ["--models", "lightgbm"], root=root)


def _final(root):
    final_test.main(["--output-dir", str(root / "results")], root=root, loader=_loader())


def _check(root):
    report = check_protocol.check_protocol(root / "results", repo=root)
    print(report.text())
    return 0 if report.approved else 1


def _correct_flow(root):
    _freeze(root)
    espiao.commit_all(root, "Congela o protocolo", "results/protocolo.lock")
    _final(root)
    espiao.commit_all(root, "Avaliação final", "results")


def test_correct_flow_is_approved(repo, capsys, monkeypatch):
    assert _check(repo) == 0  # antes do congelamento: apenas avisos
    _correct_flow(repo)
    assert _check(repo) == 0
    out = capsys.readouterr().out
    assert "APROVADO" in out and "[FALHA]" not in out
    from src.pipeline import integrity

    monkeypatch.setattr(integrity, "repo_root", lambda: repo)  # CLI: --output-dir e código de saída
    with pytest.raises(SystemExit) as exc:
        check_protocol.main(["--output-dir", str(repo / "results")])
    assert exc.value.code == 0


def test_lock_and_results_in_same_commit_fail(repo):
    _freeze(repo)
    espiao.commit_all(repo, "Congela o protocolo", "results/protocolo.lock")
    _final(repo)
    lock_path = repo / "results" / freeze.LOCK_FILE
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["created_at"] = "outro"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    espiao.commit_all(repo, "Lock e resultados juntos", "results")
    assert _check(repo) == 1


def test_results_without_lock_commit_fail(repo):
    _freeze(repo)
    espiao.commit_all(repo, "Congela o protocolo", "results/protocolo.lock")
    _final(repo)
    espiao.git(repo, "rm", "-q", "--cached", "results/protocolo.lock")
    espiao.commit_all(repo, "Remove o lock do índice")
    espiao.git(repo, "checkout", "-q", "--orphan", "outra")
    espiao.commit_all(repo, "Histórico sem o lock", "results/test.csv", "results/test_log.csv")
    assert _check(repo) == 1


def test_locked_change_after_test_needs_justification(repo, capsys):
    _correct_flow(repo)
    path = repo / "src" / "models" / "group_model.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n# ajuste\n", encoding="utf-8")
    espiao.commit_all(repo, "Ajusta o modelo")
    assert _check(repo) == 1
    path.write_text(path.read_text(encoding="utf-8") + "# outro\n", encoding="utf-8")
    espiao.git(repo, "add", "-A")
    espiao.git(repo, "commit", "-q", "--amend", "-m", "Comentário\n\nJustificativa: documentação.")
    capsys.readouterr()
    assert _check(repo) == 0
    assert "[AVISO]" in capsys.readouterr().out


def test_second_initial_evaluation_fails(repo):
    _correct_flow(repo)
    log = repo / "results" / "test_log.csv"
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    log.write_text("".join(lines + lines[1:]), encoding="utf-8")  # segunda avaliação inicial
    espiao.commit_all(repo, "Nova avaliação", "results")
    assert _check(repo) == 1


def test_lock_updated_after_results_fails(repo):
    _correct_flow(repo)
    lock_path = repo / "results" / freeze.LOCK_FILE
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["seed"] = 7
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    espiao.commit_all(repo, "Recongela", "results/protocolo.lock")
    assert _check(repo) == 1
