"""Congelamento do protocolo antes da avaliação final (infraestrutura pronta).

Executado ao final da fase de desenvolvimento (`run_validation` e estudos por CV
interna), grava `results/protocolo.lock` (JSON, terminador "\\n") com:

- em `files`, o sha256 de TODOS os arquivos versionados em `src/**/*.py` e `data/*.py`,
  mais `data/selecao_datasets.txt`, com quebras de linha normalizadas;
- o resultado de `search_space` em uma tentativa fixa (repr dos hiperparâmetros
  amostrados e das distribuições), que identifica o espaço de busca;
- o orçamento da busca (n_trials, timeout, max_samples), a semente, os sistemas,
  os folds externos e os task IDs da seleção;
- o commit git atual e se havia alterações não registradas.

Exige um repositório git. Depois do congelamento, registre o lock no git ANTES
da avaliação final:

    python -m src.pipeline.freeze
    git add results/protocolo.lock && git commit -m "Congela o protocolo"
    python -m src.pipeline.final_test
    git add results && git commit -m "Avaliação final"
    python -m src.pipeline.check_protocol

`src.pipeline.final_test` recusa executar se o lock não estiver versionado e
idêntico ao de HEAD, se houver alterações não comitadas ou se qualquer hash
divergir. Existindo qualquer vestígio de avaliação final (test.csv, test_log.csv,
test_manifest.json, predições de teste ou commits com esses arquivos), o
protocolo não pode mais ser recongelado, nem com `--force`.

Exemplos:
    python -m src.pipeline.freeze
    python -m src.pipeline.freeze --force      (recongela antes do teste, se necessário)
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.load_tabarena import OFFICIAL_FOLDS, REPO_ROOT, read_selection, validate_selection
from src.pipeline.integrity import (
    LOCK_FILE,
    SELECTION_FILE,
    describe_diff,
    git_head,
    is_git_repo,
    locked_hashes,
    relative_to_root,
    sha256_file,
    test_evidence,
    try_git,
    write_json,
)
from src.pipeline.systems import ALL_MODELS
from src.pipeline.tune import DEFAULT_MAX_SAMPLES, DEFAULT_N_TRIALS, DEFAULT_TIMEOUT_S

__all__ = [
    "LOCK_FILE",
    "build_lock",
    "lock_digest",
    "main",
    "read_lock",
    "search_space_signature",
    "sha256_file",
    "verify_lock",
]

LOCK_VERSION = 2
LOCK_NAME = LOCK_FILE  # mesmo nome usado nas trilhas de graduação (IF699 e CIN0209)


def search_space_signature(seed: int = 0) -> dict[str, str]:
    """Identifica o espaço de busca: hiperparâmetros de uma tentativa fixa e distribuições.

    Usa um estudo Optuna com amostrador aleatório de semente fixa; se o espaço
    ainda não estiver implementado, registra o erro (o hash continua comparável).
    """
    import optuna

    from src.models.group_model import search_space

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    trial = study.ask()
    try:
        params = search_space(trial)
    except Exception as exc:  # noqa: BLE001 - o espaço pode não estar implementado
        return {"sample": f"{type(exc).__name__}: {exc}", "distributions": ""}
    distributions = sorted((k, repr(v)) for k, v in trial.distributions.items())
    return {"sample": repr(params), "distributions": repr(distributions)}


def git_info(root: Path) -> dict[str, Any]:
    """Commit atual e indicação de alterações não registradas (None se não houver git)."""
    status = try_git(root, "status", "--porcelain")
    return {"git_commit": git_head(root), "git_dirty": None if status is None else bool(status)}


def build_lock(
    root: Path = REPO_ROOT,
    seed: int = 42,
    n_trials: int = DEFAULT_N_TRIALS,
    tune_timeout: float | None = DEFAULT_TIMEOUT_S,
    tune_max_samples: int | None = DEFAULT_MAX_SAMPLES,
    models: list[str] | tuple[str, ...] = ALL_MODELS,
    folds: list[int] | tuple[int, ...] = OFFICIAL_FOLDS,
    task_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Conteúdo do protocolo.lock para o repositório git em `root`."""
    root = Path(root)
    if task_ids is None:
        task_ids = read_selection(root / SELECTION_FILE)
    return {
        "version": LOCK_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": locked_hashes(root),
        "search_space": search_space_signature(),
        "budget": {
            "n_trials": n_trials,
            "tune_timeout": tune_timeout,
            "tune_max_samples": tune_max_samples,
        },
        "seed": seed,
        "models": list(models),
        "folds": list(folds),
        "task_ids": list(task_ids),
        **git_info(root),
    }


def read_lock(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lock_digest(path: Path) -> str:
    """sha256 do próprio arquivo de lock (registrado no log e no manifesto do teste)."""
    return sha256_file(path)


def verify_lock(lock: dict[str, Any], root: Path = REPO_ROOT) -> list[str]:
    """Divergências entre o lock e o estado atual do repositório (vazia se íntegro).

    Compara o conjunto e o conteúdo dos arquivos travados versionados e a
    assinatura do espaço de busca.
    """
    root = Path(root)
    problems = describe_diff(lock.get("files", {}), locked_hashes(root))
    if lock.get("search_space") != search_space_signature():
        problems.append("search_space: espaço de busca alterado após o congelamento")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--tune-timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--tune-max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument("--folds", type=int, nargs="+", default=list(OFFICIAL_FOLDS))
    parser.add_argument(
        "--force", action="store_true", help="substitui um lock existente (só antes do teste)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="não exige que a seleção cumpra as regras do enunciado (uso em testes parciais)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> Path:
    args = parse_args(argv)
    root = Path(root)
    out = Path(args.output_dir)
    lock_path = out / LOCK_FILE
    if not is_git_repo(root):
        raise SystemExit(
            f"{root} não é um repositório git. O protocolo depende do histórico git: "
            "clone o template com git e registre o seu trabalho em commits."
        )
    relative_to_root(out, root)  # o lock precisa ser versionado no mesmo repositório
    traces = test_evidence(out, root)
    if traces:
        raise SystemExit(
            "Há vestígios de avaliação final; o protocolo não pode ser recongelado "
            "(nem com --force):\n  " + "\n  ".join(traces)
        )
    if lock_path.exists() and not args.force:
        raise SystemExit(f"{lock_path} já existe. Use --force para recongelar antes do teste.")
    task_ids = read_selection(root / SELECTION_FILE)
    if not task_ids:
        raise SystemExit(f"Nenhum task ID em {SELECTION_FILE}.")
    if not args.skip_validation:
        report = validate_selection(task_ids)
        if not report.ok:
            raise SystemExit("Seleção inválida:\n  " + "\n  ".join(report.errors))
    validation_path = out / "validation.csv"
    if not validation_path.exists():
        print(f"Aviso: {validation_path} não encontrado; a fase de desenvolvimento foi feita?")
    lock = build_lock(
        root,
        seed=args.seed,
        n_trials=args.n_trials,
        tune_timeout=args.tune_timeout,
        tune_max_samples=args.tune_max_samples,
        models=args.models,
        folds=args.folds,
        task_ids=task_ids,
    )
    if lock["search_space"]["distributions"] == "":
        print(f"Aviso: search_space não pôde ser avaliado ({lock['search_space']['sample']}).")
    if lock["git_dirty"]:
        print(
            "Aviso: há alterações não registradas no git; comite-as e execute "
            "freeze --force antes de comitar o lock."
        )
    out.mkdir(parents=True, exist_ok=True)
    write_json(lock_path, lock)
    print(f"Protocolo congelado em {lock_path} (sha256 {lock_digest(lock_path)[:12]}).")
    print(
        "Próximo passo: registre o lock no git ANTES da avaliação final:\n"
        f'  git add {relative_to_root(lock_path, root)} && git commit -m "Congela o protocolo"'
    )
    return lock_path


if __name__ == "__main__":
    main()
