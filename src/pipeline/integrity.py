"""Integridade do protocolo: hashes normalizados, consultas ao git e vestígios do teste (infraestrutura).

As travas locais (`freeze`, `final_test`, `run_validation`) são dissuasórias:
quem tem acesso ao repositório pode apagar arquivos. A evidência do protocolo é
o HISTÓRICO GIT, conferido por `python -m src.pipeline.check_protocol`. Por
isso o fluxo exige comitar `results/protocolo.lock` ANTES da avaliação final e
comitar `results/` LOGO DEPOIS dela.

Convenções (as mesmas das trilhas de graduação, IF699 e CIN0209):
- arquivos travados: todos os arquivos VERSIONADOS no git em `src/**/*.py` e
  `data/*.py`, mais `data/selecao_datasets.txt`;
- hash sha256 com quebras de linha normalizadas (\\r\\n é convertido em \\n antes do hash),
  para que o mesmo conteúdo produza o mesmo hash no Windows, no Linux e no Colab;
- CSV e JSON do protocolo são gravados com terminador de linha "\\n";
- vestígio de avaliação final: test.csv, test_manifest.json, predições de teste
  (`preds/`) ou test_log.csv com evento "inicio" (registros de recusa não
  contam), no disco ou em qualquer commit do histórico git.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

from data.load_tabarena import REPO_ROOT

LOCK_FILE = "protocolo.lock"
TEST_FILE = "test.csv"
LOG_FILE = "test_log.csv"
MANIFEST_FILE = "test_manifest.json"
TEST_FILES = (TEST_FILE, LOG_FILE, MANIFEST_FILE)
TEST_PREDS_DIR = "preds"
SELECTION_FILE = "data/selecao_datasets.txt"
LOCKED_DIRS = ("src", "data")
STATUS_DIRS = ("src", "data", "config")
START_EVENT = "inicio"
END_EVENT = "fim"


class GitError(RuntimeError):
    """Falha ao executar um comando git."""


def repo_root() -> Path:
    """Raiz do repositório do grupo."""
    return REPO_ROOT


# ------------------------------------------------------------------ hashes e gravação


def normalize_newlines(data: bytes) -> bytes:
    """Normaliza as quebras de linha (CRLF para LF) antes do hash."""
    return data.replace(b"\r\n", b"\n")


def sha256_bytes(data: bytes) -> str:
    """sha256 do conteúdo com quebras de linha normalizadas."""
    return hashlib.sha256(normalize_newlines(data)).hexdigest()


def sha256_file(path: Path) -> str:
    """sha256 de um arquivo com quebras de linha normalizadas."""
    return sha256_bytes(Path(path).read_bytes())


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def write_json(path: Path, data: dict) -> None:
    """Grava JSON em UTF-8 com terminador "\\n" em qualquer sistema."""
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def csv_line(values: Iterable) -> str:
    """Uma linha CSV (sem o terminador), no formato do módulo csv."""
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerow(list(values))
    return buffer.getvalue().rstrip("\n")


def is_locked_path(rel: str) -> bool:
    """True para `src/**/*.py`, `data/*.py` e `data/selecao_datasets.txt` (caminhos com "/")."""
    parts = rel.split("/")
    if "__pycache__" in parts:
        return False
    if rel == SELECTION_FILE:
        return True
    if parts[0] == "src" and rel.endswith(".py"):
        return True
    return len(parts) == 2 and parts[0] == "data" and rel.endswith(".py")


# ------------------------------------------------------------------ git


def git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    """Executa `git <args>` em `root`; levanta GitError se o comando falhar."""
    try:
        proc = subprocess.run(
            ["git", "-c", "core.quotepath=off", *args],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitError(f"git {' '.join(args)}: {exc}") from exc
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(f"git {' '.join(args)}: {message}")
    return proc.stdout if binary else proc.stdout.decode("utf-8", errors="replace")


def try_git(root: Path, *args: str) -> str | None:
    """Como `git`, mas devolve None em caso de falha."""
    try:
        return str(git(root, *args)).strip()
    except GitError:
        return None


def is_git_repo(root: Path | None = None) -> bool:
    return try_git(Path(root or repo_root()), "rev-parse", "--is-inside-work-tree") == "true"


def git_head(root: Path | None = None) -> str | None:
    """Commit atual (HEAD) ou None se não houver repositório ou commit."""
    return try_git(Path(root or repo_root()), "rev-parse", "--verify", "HEAD")


def rel_to_repo(path: Path, root: Path | None = None) -> str | None:
    """Caminho relativo à raiz do repositório, com "/" (None se estiver fora dele)."""
    try:
        return Path(path).resolve().relative_to(Path(root or repo_root()).resolve()).as_posix()
    except ValueError:
        return None


def relative_to_root(path: Path, root: Path) -> str:
    """Como `rel_to_repo`, mas levanta SystemExit se o caminho estiver fora do repositório."""
    rel = rel_to_repo(path, root)
    if rel is None:
        raise SystemExit(
            f"{path} está fora do repositório {root}: o diretório de resultados deve ser "
            "versionado no mesmo repositório do código."
        )
    return rel


def show_file(commit: str, rel: str, root: Path | None = None) -> bytes | None:
    """Conteúdo de `rel` (relativo a `root`) no commit, ou None se não existir."""
    try:
        return git(Path(root or repo_root()), "show", f"{commit}:./{rel}", binary=True)
    except GitError:
        return None


def _split_z(output: bytes) -> list[str]:
    return [p.decode("utf-8") for p in output.split(b"\0") if p]


def tracked_locked_files(root: Path, commit: str | None = None) -> list[str]:
    """Arquivos travados versionados no índice (commit=None) ou em um commit."""
    if commit is None:
        out = git(root, "ls-files", "-z", "--", *LOCKED_DIRS, binary=True)
    else:
        out = git(
            root, "ls-tree", "-r", "-z", "--name-only", commit, "--", *LOCKED_DIRS, binary=True
        )
    return sorted(p for p in _split_z(out) if is_locked_path(p))


# ------------------------------------------------------------------ arquivos travados


def locked_hashes(root: Path | None = None) -> dict[str, str]:
    """{arquivo travado: sha256 normalizado} no estado atual do disco.

    `data/selecao_datasets.txt` é sempre incluído, mesmo que ainda não versionado.
    """
    root = Path(root or repo_root())
    files = set(tracked_locked_files(root))
    files.add(SELECTION_FILE)
    return {rel: sha256_file(root / rel) for rel in sorted(files) if (root / rel).exists()}


def locked_hashes_at(commit: str, root: Path | None = None) -> dict[str, str]:
    """{arquivo travado: sha256 normalizado} no commit informado."""
    root = Path(root or repo_root())
    out = {}
    for rel in tracked_locked_files(root, commit):
        data = show_file(commit, rel, root)
        if data is not None:
            out[rel] = sha256_bytes(data)
    return out


def diff_hashes(expected: dict[str, str], current: dict[str, str]) -> list[str]:
    """Arquivos ausentes, novos ou com conteúdo diferente."""
    return sorted(f for f in set(expected) | set(current) if expected.get(f) != current.get(f))


def describe_diff(expected: dict[str, str], current: dict[str, str]) -> list[str]:
    """Divergências entre dois mapas caminho -> sha256, em texto (vazia se idênticos)."""
    problems = []
    for rel in diff_hashes(expected, current):
        if rel not in current:
            problems.append(f"{rel}: arquivo travado ausente")
        elif rel not in expected:
            problems.append(f"{rel}: arquivo novo, não incluído no congelamento")
        else:
            problems.append(f"{rel}: conteúdo alterado após o congelamento")
    return problems


def dirty_locked_paths(results_dir: Path, root: Path | None = None) -> list[str]:
    """Alterações não comitadas em src/, data/, config/ e no lock.

    Considera arquivos versionados modificados, adicionados ou removidos e
    arquivos novos (não versionados) que seriam travados (`.py` e a seleção).
    """
    root = Path(root or repo_root())
    lock_rel = rel_to_repo(Path(results_dir) / LOCK_FILE, root)
    extra = (lock_rel,) if lock_rel else ()
    paths = [p for p in (*STATUS_DIRS, *extra) if (root / p).exists()]
    if not paths:
        return []
    out = git(
        root, "status", "--porcelain", "-z", "--untracked-files=all", "--", *paths, binary=True
    )
    prefix = try_git(root, "rev-parse", "--show-prefix") or ""  # porcelain usa a raiz do git
    problems = []
    entries = iter(_split_z(out))
    for entry in entries:
        status, rel = entry[:2], entry[3:]
        if status[0] in "RC":
            next(entries, None)  # caminho de origem de renomeação ou cópia
        rel = rel[len(prefix) :] if prefix and rel.startswith(prefix) else rel
        if status == "??" and not (is_locked_path(rel) or rel in extra):
            continue
        problems.append(f"{status.strip()} {rel}")
    return problems


# ------------------------------------------------------------------ vestígios do teste


def log_has_start(data: bytes | None) -> bool:
    """test_log.csv contém algum evento de início? (registros de recusa não contam)."""
    if not data:
        return False
    text = normalize_newlines(data).decode("utf-8", errors="replace")
    return any(row.get("event") == START_EVENT for row in csv.DictReader(io.StringIO(text)))


def local_test_evidence(results_dir: Path) -> list[str]:
    """Vestígios no disco: test.csv, test_log.csv com início, manifesto, preds/*.npz."""
    out = Path(results_dir)
    found = []
    if (out / TEST_FILE).exists():
        found.append(str(out / TEST_FILE))
    log = out / LOG_FILE
    if log.exists() and log_has_start(log.read_bytes()):
        found.append(f"{log} (evento de início)")
    if (out / MANIFEST_FILE).exists():
        found.append(str(out / MANIFEST_FILE))
    preds = out / TEST_PREDS_DIR
    if preds.is_dir() and any(preds.glob("*.npz")):
        found.append(f"{preds}/")
    return found


def git_test_evidence(results_dir: Path, root: Path | None = None) -> list[str]:
    """Vestígios no histórico git (em qualquer commit) ou no índice."""
    root = Path(root or repo_root())
    if not is_git_repo(root) or git_head(root) is None:
        return []
    rel = rel_to_repo(results_dir, root)
    if rel is None:
        return []
    found = []
    targets = [f"{rel}/{TEST_FILE}", f"{rel}/{MANIFEST_FILE}", f"{rel}/{TEST_PREDS_DIR}"]
    tracked = try_git(root, "ls-files", "--", *targets)
    history = (try_git(root, "log", "--all", "--format=%H", "--", *targets) or "").split()
    if tracked or history:
        found.append(f"{TEST_FILE}, {MANIFEST_FILE} ou {TEST_PREDS_DIR}/ no histórico git")
    log_rel = f"{rel}/{LOG_FILE}"
    commits = (try_git(root, "log", "--all", "--format=%H", "--", log_rel) or "").split()
    if any(log_has_start(show_file(c, log_rel, root)) for c in commits):
        found.append(f"{LOG_FILE} com evento de início no histórico git")
    return found


def test_evidence(results_dir: Path, root: Path | None = None) -> list[str]:
    """Todos os vestígios de avaliação final (disco e histórico git).

    Confere o diretório de saída e, se for outro, também `results/` da raiz.
    """
    root = Path(root or repo_root())
    dirs = [Path(results_dir).resolve()]
    if (root / "results").resolve() not in dirs:
        dirs.append((root / "results").resolve())
    found: list[str] = []
    for directory in dirs:
        found += local_test_evidence(directory)
        found += git_test_evidence(directory, root)
    return list(dict.fromkeys(found))
