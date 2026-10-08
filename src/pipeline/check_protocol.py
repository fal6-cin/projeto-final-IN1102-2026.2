"""Conferência do protocolo pelo histórico git (infraestrutura pronta, usada também na correção).

As travas de `freeze` e `final_test` são dissuasórias: arquivos locais podem
ser apagados. A evidência de que o teste foi usado uma única vez, depois do
congelamento, é o HISTÓRICO GIT. Por isso o grupo comita `results/protocolo.lock`
ANTES de `final_test` e comita `results/` IMEDIATAMENTE depois. Este comando
confere, no histórico do commit atual (HEAD):

1. o commit que grava `results/protocolo.lock` antecede o primeiro commit com
   test.csv, test_manifest.json, preds/ ou test_log.csv com evento de início, e
   o lock não é alterado nesse commit nem depois;
2. test_log.csv registra uma única avaliação inicial; retomadas (--resume) só
   avaliam combinações que estavam pendentes ou com erro; todos os eventos usam
   o lock comitado e partem de commits que contêm o lock;
3. os hashes de todos os arquivos travados (src/**/*.py, data/*.py,
   data/selecao_datasets.txt) conferem com o lock no commit do lock e em HEAD;
   test.csv confere com os hashes por linha do manifesto;
4. nenhum commit entre o lock e o teste altera arquivos travados, e nenhum
   commit posterior ao teste os altera sem uma linha "Justificativa: ..." na
   mensagem do commit (com justificativa, a alteração é listada como AVISO);
5. os resultados do teste presentes no disco estão comitados.

Imprime um relatório e termina com código 0 (aprovado) ou 1 (reprovado).

Uso:
    python -m src.pipeline.check_protocol
    python -m src.pipeline.check_protocol --output-dir results
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.pipeline import integrity
from src.pipeline.integrity import diff_hashes, sha256_bytes, show_file, try_git

OK, AVISO, FALHA = "OK", "AVISO", "FALHA"
JUSTIFICATION = "justificativa:"


@dataclass
class Report:
    items: list[tuple[str, str]] = field(default_factory=list)

    def add(self, level: str, message: str) -> None:
        self.items.append((level, message))

    @property
    def failures(self) -> list[str]:
        return [m for level, m in self.items if level == FALHA]

    @property
    def approved(self) -> bool:
        return not self.failures

    def text(self) -> str:
        lines = [f"[{level}] {message}" for level, message in self.items]
        verdict = "APROVADO" if self.approved else f"REPROVADO ({len(self.failures)} falha(s))"
        return "\n".join([*lines, f"Resultado: {verdict}"])


def _commits(repo: Path) -> list[str]:
    """Commits do histórico de HEAD, do mais antigo ao mais recente."""
    out = try_git(repo, "log", "--reverse", "--topo-order", "--format=%H", "HEAD") or ""
    return out.split()


def _touching(repo: Path, paths: list[str]) -> list[str]:
    return (try_git(repo, "log", "--format=%H", "HEAD", "--", *paths) or "").split()


def _log_rows(data: bytes | None) -> list[dict]:
    if not data:
        return []
    text = integrity.normalize_newlines(data).decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def _has_test(repo: Path, sha: str, rel: str) -> bool:
    names = try_git(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        sha,
        "--",
        f"{rel}/{integrity.TEST_FILE}",
        f"{rel}/{integrity.MANIFEST_FILE}",
        f"{rel}/{integrity.TEST_PREDS_DIR}",
    )
    if names:
        return True
    return integrity.log_has_start(show_file(sha, f"{rel}/{integrity.LOG_FILE}", repo))


def _is_ancestor(repo: Path, a: str, b: str) -> bool:
    return try_git(repo, "merge-base", "--is-ancestor", a, b) is not None


def _message(repo: Path, sha: str) -> str:
    return try_git(repo, "log", "-1", "--format=%B", sha) or ""


def _changed_locked(repo: Path, sha: str) -> list[str]:
    out = try_git(
        repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", "--relative", sha
    )
    return [f for f in (out or "").splitlines() if integrity.is_locked_path(f)]


def _is_resume(detail: str) -> bool:
    for part in detail.split(";"):
        key, _, value = part.strip().partition("=")
        if key == "resume":
            return value.strip() == "True"
    return False


def _check_log(report: Report, rows: list[dict], lock_sha: str, lock_commit: str, repo: Path):
    starts = [r for r in rows if r.get("event") == integrity.START_EVENT]
    if not starts:
        report.add(FALHA, "test_log.csv não registra nenhum evento de início da avaliação final.")
        return
    initial = [r for r in starts if not _is_resume(r.get("detail", ""))]
    if len(initial) != 1 or _is_resume(starts[0].get("detail", "")):
        report.add(FALHA, f"test_log.csv registra {len(initial)} avaliações iniciais (esperado 1).")
    else:
        report.add(
            OK,
            f"test_log.csv registra uma única avaliação inicial e {len(starts) - 1} retomada(s).",
        )
    status: dict[str, str] = {}
    redone = []
    for row in rows:
        if row.get("event") == "sistema":
            key = row.get("model", "")
            if status.get(key) == "ok":
                redone.append(key)
            status[key] = row.get("status", "")
    if redone:
        report.add(
            FALHA, f"Retomadas reavaliaram combinações já concluídas: {sorted(set(redone))[:5]}."
        )
    elif len(starts) > 1:
        report.add(OK, "As retomadas avaliaram apenas combinações pendentes ou com erro.")
    used = {
        r.get("lock_sha256")
        for r in rows
        if r.get("event") in (integrity.START_EVENT, "sistema", integrity.END_EVENT)
    }
    if used != {lock_sha}:
        report.add(
            FALHA, "Há execuções da avaliação final com um protocolo.lock diferente do comitado."
        )
    heads = {r.get("git_head") for r in starts}
    bad_heads = [h for h in heads if not h or not _is_ancestor(repo, lock_commit, h)]
    if bad_heads:
        report.add(
            FALHA, f"Execuções da avaliação final em commits sem o lock comitado: {bad_heads}."
        )
    else:
        report.add(
            OK, "Todas as execuções da avaliação final partiram de commits que contêm o lock."
        )


def _check_manifest(report: Report, repo: Path, rel: str, lock_sha: str) -> None:
    raw = show_file("HEAD", f"{rel}/{integrity.MANIFEST_FILE}", repo)
    if raw is None:
        report.add(FALHA, "test_manifest.json não está versionado em HEAD.")
        return
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("lock_sha256") != lock_sha:
        report.add(FALHA, "test_manifest.json foi gerado com outro protocolo.lock.")
    from src.pipeline.final_test import CSV_COLUMNS, unit_key

    text = integrity.normalize_newlines(
        show_file("HEAD", f"{rel}/{integrity.TEST_FILE}", repo) or b""
    )
    lines = [line for line in text.decode("utf-8").split("\n")[1:] if line]
    pos = [CSV_COLUMNS.index(c) for c in ("task_id", "fold", "model")]
    by_key: dict[str, list[str]] = {}
    for line in lines:
        fields = next(csv.reader([line]))
        by_key.setdefault(unit_key(*(fields[p] for p in pos)), []).append(line)
    repeated = sorted(k for k, v in by_key.items() if len(v) > 1)
    if repeated:
        report.add(FALHA, f"test.csv tem combinações repetidas: {repeated[:5]}.")
    bad = [
        k
        for k, sha in manifest.get("rows", {}).items()
        if k not in by_key or integrity.sha256_text(by_key[k][-1]) != sha
    ]
    if bad:
        report.add(FALHA, f"test.csv não confere com os hashes do manifesto: {bad[:5]}.")
    else:
        report.add(OK, "test.csv confere com os hashes por linha do manifesto.")


def check_protocol(results_dir: Path = Path("results"), repo: Path | None = None) -> Report:
    """Executa as verificações e devolve o relatório."""
    report = Report()
    repo = Path(repo or integrity.repo_root())
    if not integrity.is_git_repo(repo) or integrity.git_head(repo) is None:
        report.add(FALHA, f"{repo} não é um repositório git com commits.")
        return report
    results_dir = Path(results_dir)
    if not results_dir.is_absolute():
        results_dir = repo / results_dir
    rel = integrity.rel_to_repo(results_dir, repo)
    if rel is None:
        report.add(FALHA, f"{results_dir} está fora do repositório.")
        return report
    lock_rel = f"{rel}/{integrity.LOCK_FILE}"
    commits = _commits(repo)
    position = {c: i for i, c in enumerate(commits)}
    test_candidates = _touching(
        repo,
        [
            f"{rel}/{integrity.TEST_FILE}",
            f"{rel}/{integrity.LOG_FILE}",
            f"{rel}/{integrity.MANIFEST_FILE}",
            f"{rel}/{integrity.TEST_PREDS_DIR}",
        ],
    )
    test_commits = sorted(
        (s for s in test_candidates if s in position and _has_test(repo, s, rel)),
        key=position.get,
    )
    lock_commits = sorted(
        (s for s in _touching(repo, [lock_rel]) if s in position), key=position.get
    )
    local = integrity.local_test_evidence(results_dir)
    if not test_commits:
        if local:
            report.add(FALHA, f"Há resultados da avaliação final no disco não comitados: {local}.")
        if not lock_commits:
            report.add(AVISO, "O protocolo.lock ainda não foi comitado (protocolo não congelado).")
        else:
            report.add(
                OK, f"protocolo.lock comitado em {lock_commits[-1][:10]}; teste não executado."
            )
        return report
    first_test = test_commits[0]
    report.add(OK, f"Primeiro commit com a avaliação final: {first_test[:10]}.")
    before = [s for s in lock_commits if position[s] < position[first_test]]
    if not before:
        report.add(FALHA, "O protocolo.lock não foi comitado ANTES do primeiro commit com o teste.")
        return report
    lock_commit = before[-1]
    report.add(OK, f"protocolo.lock comitado antes do teste: {lock_commit[:10]}.")
    if len(before) > 1:
        report.add(
            AVISO,
            f"O lock foi comitado {len(before)} vezes antes do teste (recongelamentos); justifique no relatório.",
        )
    after = [s[:10] for s in lock_commits if position[s] >= position[first_test]]
    if after:
        report.add(FALHA, f"protocolo.lock alterado no commit do teste ou depois: {after}.")

    lock_bytes = show_file(lock_commit, lock_rel, repo) or b"{}"
    lock = json.loads(lock_bytes.decode("utf-8"))
    lock_sha = sha256_bytes(lock_bytes)
    at_lock = integrity.locked_hashes_at(lock_commit, repo)
    diffs = diff_hashes(lock.get("files", {}), at_lock)
    if diffs:
        report.add(FALHA, f"Hashes divergentes no commit do lock: {diffs}.")
    else:
        report.add(OK, f"Hashes de {len(at_lock)} arquivos travados conferem no commit do lock.")

    between = [
        c[:10]
        for c in commits[position[lock_commit] + 1 : position[first_test] + 1]
        if _changed_locked(repo, c)
    ]
    if between:
        report.add(FALHA, f"Arquivos travados alterados entre o lock e o teste: {between}.")
    justified: set[str] = set()
    unjustified = []
    for c in commits[position[first_test] + 1 :]:
        changed = _changed_locked(repo, c)
        if not changed:
            continue
        lines = _message(repo, c).splitlines()
        if any(line.strip().lower().startswith(JUSTIFICATION) for line in lines):
            justified |= set(changed)
            report.add(AVISO, f"Commit {c[:10]} posterior ao teste altera {changed} (justificado).")
        else:
            unjustified.append(c[:10])
    if unjustified:
        report.add(
            FALHA,
            f"Commits posteriores ao teste alteram arquivos travados sem justificativa: {unjustified}.",
        )
    at_head = integrity.locked_hashes_at("HEAD", repo)
    head_diffs = [f for f in diff_hashes(lock.get("files", {}), at_head) if f not in justified]
    if head_diffs:
        report.add(FALHA, f"Hashes divergentes em HEAD: {head_diffs}.")
    else:
        report.add(
            OK, "Hashes dos arquivos travados conferem em HEAD (exceto alterações justificadas)."
        )

    log_rel = f"{rel}/{integrity.LOG_FILE}"
    _check_log(report, _log_rows(show_file("HEAD", log_rel, repo)), lock_sha, lock_commit, repo)
    _check_manifest(report, repo, rel, lock_sha)

    dirty = try_git(repo, "status", "--porcelain", "--untracked-files=all", "--", rel) or ""
    pending = [
        line[3:]
        for line in dirty.splitlines()
        if any(name in line for name in (*integrity.TEST_FILES, integrity.TEST_PREDS_DIR + "/"))
    ]
    if pending:
        report.add(
            FALHA, f"Resultados da avaliação final no disco diferentes dos comitados: {pending}."
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    report = check_protocol(args.output_dir)
    print(report.text())
    sys.exit(0 if report.approved else 1)


if __name__ == "__main__":
    main()
