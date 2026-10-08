"""Avaliação final ÚNICA no teste de cada fold externo, após o congelamento do protocolo.

Fluxo da pós-graduação (IN1102):
    1. desenvolvimento (só treino): `python -m src.pipeline.run_validation` e estudos
       por CV interna (`cv_grid`, `overfitting`, `class_errors`);
    2. congelamento: `python -m src.pipeline.freeze` grava `results/protocolo.lock`;
    3. registro do lock no git: `git add results/protocolo.lock && git commit`;
    4. avaliação final (este comando), uma única vez;
    5. registro imediato dos resultados: `git add results && git commit`;
    6. conferência: `python -m src.pipeline.check_protocol`.

Para cada dataset e fold externo do lock, cada sistema é ajustado no treino do
fold (com busca Optuna por CV interna nesse treino, para `group_tuned`) e
avaliado UMA vez no teste. O orçamento da busca, a semente, os sistemas, os
folds e os datasets vêm do lock, não da linha de comando.

Pré-condições (o comando recusa a execução se alguma falhar):
    (a) o diretório é um repositório git;
    (b) não há alterações não comitadas (nem arquivos `.py` novos) em src/,
        data/, config/ e em `results/protocolo.lock`;
    (c) `results/protocolo.lock` está versionado e é idêntico ao de HEAD;
    (d) os hashes do lock conferem com o repositório (arquivos travados e
        espaço de busca);
    (e) não existe vestígio de avaliação anterior: test.csv, test_manifest.json,
        predições de teste (`preds/`) ou test_log.csv com evento de início, no
        disco (no diretório de saída ou em results/) ou no histórico git.

Com `--resume`, a condição (e) é substituída: só são avaliadas as combinações
(dataset, fold externo, sistema) registradas como pendentes ou com erro no
manifesto, com o mesmo lock, e a execução é recusada se test.csv tiver linhas ok
faltando ou alteradas (o manifesto guarda o hash de cada linha).

Arquivos gerados (CSV e JSON com terminador "\\n"):
    test.csv            uma linha por (dataset, fold externo, sistema), colunas CSV_COLUMNS
    preds/              probabilidades de teste ({task}_fold{f}_{sistema}.npz), base do bootstrap
    test_log.csv        registro de cada execução e recusa (timestamp, event, lock_sha256,
                        git_head, model, status, detail); em eventos "sistema", a coluna
                        model traz a combinação "task_id|fold|sistema"
    test_manifest.json  lock usado (sha256), situação de cada combinação (status:
                        pendente, ok, error), hash de cada linha ok de test.csv (rows) e
                        execuções (runs, com git_head)

As travas locais são dissuasórias; a evidência do protocolo é o histórico git.
Comite o lock ANTES deste comando e comite `results/` LOGO DEPOIS:
    git add results && git commit -m "Avaliação final"
A conferência é feita por `python -m src.pipeline.check_protocol`.

Exemplos:
    python -m src.pipeline.final_test
    python -m src.pipeline.final_test --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
import traceback
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.load_tabarena import REPO_ROOT, TabularDataset, get_folds, load_task
from src.pipeline import integrity
from src.pipeline.evaluate import fit_predict_evaluate
from src.pipeline.freeze import LOCK_NAME, lock_digest, read_lock, verify_lock
from src.pipeline.integrity import csv_line, sha256_text
from src.pipeline.systems import ALL_MODELS, BASELINES, GROUP, build_runner

__all__ = ["ALL_MODELS", "BASELINES", "CSV_COLUMNS", "GROUP", "RESULT_KEYS", "main"]

RESULT_KEYS = ["task_id", "dataset", "fold", "model"]
CSV_COLUMNS = RESULT_KEYS + [
    "auc_ovo",
    "accuracy",
    "g_mean",
    "cross_entropy",
    "tune_time_s",
    "fit_time_s",
    "predict_time_s",
    "total_time_s",
    "status",
    "error",
]
LOG_COLUMNS = ["timestamp", "event", "lock_sha256", "git_head", "model", "status", "detail"]
PENDING, OK, ERROR = "pendente", "ok", "error"

Loader = Callable[[int, list[int]], tuple[TabularDataset, list[tuple[np.ndarray, np.ndarray]]]]


def openml_loader(task_id: int, folds: list[int]):
    """Carrega o dataset e as partições oficiais (treino, teste) dos folds externos."""
    return load_task(task_id), get_folds(task_id, folds=folds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="avalia apenas combinações pendentes ou com erro, com o mesmo lock",
    )
    return parser.parse_args(argv)


def unit_key(task_id: Any, fold: Any, model: Any) -> str:
    """Chave de uma combinação no manifesto e no log: "task_id|fold|sistema"."""
    return f"{int(task_id)}|{int(fold)}|{model}"


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _log(
    out: Path,
    event: str,
    lock_sha: str,
    head: str = "",
    model: str = "",
    status: str = "",
    detail: str = "",
) -> None:
    path = Path(out) / integrity.LOG_FILE
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        if new:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([_now(), event, lock_sha, head, model, status, detail.replace("\n", " ")])


def lock_sha256(out: Path) -> str:
    path = Path(out) / LOCK_NAME
    return lock_digest(path) if path.exists() else ""


def _format_value(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    return str(value).replace("\r", " ").replace("\n", " ")


def row_line(row: dict) -> str:
    """Linha de test.csv (sem terminador) de uma combinação, em formato fixo."""
    return csv_line(_format_value(row.get(c)) for c in CSV_COLUMNS)


def read_test_lines(test_path: Path) -> dict[str, str]:
    """{chave da combinação: linha} de test.csv (linhas sem terminador)."""
    if not Path(test_path).exists():
        return {}
    text = Path(test_path).read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = [line for line in text.split("\n")[1:] if line]
    pos = [CSV_COLUMNS.index(c) for c in ("task_id", "fold", "model")]
    out = {}
    for line in lines:
        fields = next(csv.reader([line]))
        out[unit_key(*(fields[p] for p in pos))] = line
    return out


def _write_test_csv(test_path: Path, lines: dict[str, str], order: list[str]) -> None:
    body = [lines[k] for k in order if k in lines]
    body += [line for k, line in lines.items() if k not in order]
    text = "\n".join([csv_line(CSV_COLUMNS), *body]) + "\n"
    Path(test_path).write_text(text, encoding="utf-8", newline="\n")


def done_keys(csv_path: Path) -> set[tuple]:
    """Combinações (task_id, dataset, fold, model) concluídas com sucesso em test.csv."""
    if not Path(csv_path).exists():
        return set()
    done = pd.read_csv(csv_path)
    done = done[done["status"] == OK]
    return set(map(tuple, done[RESULT_KEYS].itertuples(index=False, name=None)))


def _save_predictions(
    preds_dir: Path, task_id: int, fold: int, model: str, y_true, y_proba, classes
) -> None:
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        preds_dir / f"{task_id}_fold{fold}_{model}.npz",
        y_true=np.asarray(y_true).astype(str),
        y_proba=np.asarray(y_proba, dtype=np.float32),
        classes=np.asarray(classes).astype(str),
    )


def evaluate_dataset(
    ds: TabularDataset,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fold_ids: Iterable[int],
    models: Iterable[str],
    args: argparse.Namespace,
    pending: set[str] | None = None,
    on_row: Callable[[str, dict], None] | None = None,
) -> list[dict]:
    """Ajusta cada sistema no treino de cada fold externo e avalia uma vez no teste.

    `args` deve ter `output_dir`, `seed`, `n_trials`, `tune_timeout` e
    `tune_max_samples`. Com `pending`, só as combinações cujas chaves
    (`unit_key`) estão no conjunto são executadas; `on_row(chave, linha)` é
    chamada após cada combinação (sucesso ou falha). Falhas são registradas com
    status `error` e a execução segue.
    """
    rows: list[dict] = []
    for fold_id, (train_idx, test_idx) in zip(fold_ids, folds):
        X_train, X_test = ds.X.iloc[train_idx], ds.X.iloc[test_idx]
        y = np.asarray(ds.y)
        y_train, y_test = y[train_idx], y[test_idx]
        for model in models:
            key = unit_key(ds.task_id, fold_id, model)
            if pending is not None and key not in pending:
                continue
            row: dict[str, Any] = dict(zip(RESULT_KEYS, (ds.task_id, ds.name, fold_id, model)))
            t0 = time.perf_counter()
            try:
                estimator, tune_time_s, _ = build_runner(model, args)(X_train, y_train)
                result = fit_predict_evaluate(
                    estimator, X_train, y_train, X_test, y_test, tune_time_s=tune_time_s
                )
                row.update(result.to_dict(), status=OK, error="")
                _save_predictions(
                    args.output_dir / integrity.TEST_PREDS_DIR,
                    ds.task_id,
                    fold_id,
                    model,
                    y_test,
                    result.y_proba,
                    result.classes,
                )
                print(
                    f"[{ds.name} | fold {fold_id}] {model}: AUC={result.auc_ovo:.4f} "
                    f"ACC={result.accuracy:.4f} tempo={row['total_time_s']:.1f}s"
                )
            except Exception as exc:  # noqa: BLE001 - registra a falha e segue para a próxima combinação
                row.update(status=ERROR, error=f"{type(exc).__name__}: {exc}")
                row["total_time_s"] = time.perf_counter() - t0
                print(f"[{ds.name} | fold {fold_id}] {model}: FALHA ({row['error']})")
                traceback.print_exc()
            if on_row is not None:
                on_row(key, row)
            rows.append(row)
    return rows


# ------------------------------------------------------------------ pré-condições


def check_repository(out: Path, root: Path) -> str:
    """Condições (a), (b), (c) e (d); devolve o commit atual (HEAD)."""
    if not integrity.is_git_repo(root):
        raise SystemExit(
            f"{root} não é um repositório git: a avaliação final exige o histórico git "
            "(lock comitado antes dos resultados)."
        )
    lock_path = Path(out) / LOCK_NAME
    lock_rel = integrity.relative_to_root(lock_path, root)
    if not lock_path.exists():
        raise SystemExit(
            f"{lock_path} não encontrado. Congele o protocolo com `python -m src.pipeline.freeze` "
            "e comite o lock antes da avaliação final."
        )
    head = integrity.git_head(root)
    committed = integrity.show_file("HEAD", lock_rel, root) if head else None
    if committed is None:
        raise SystemExit(
            f"{lock_rel} não está versionado em HEAD. Registre o lock antes da avaliação final:\n"
            f'  git add {lock_rel} && git commit -m "Congela o protocolo"'
        )
    if integrity.sha256_bytes(committed) != lock_sha256(out):
        raise SystemExit(f"{lock_rel} difere da versão comitada em HEAD; comite o lock atual.")
    dirty = integrity.dirty_locked_paths(out, root)
    if dirty:
        raise SystemExit(
            "Há alterações não comitadas em arquivos travados ou no lock; comite-as (e "
            "recongele, se necessário) antes da avaliação final:\n  " + "\n  ".join(dirty)
        )
    problems = verify_lock(read_lock(lock_path), root)
    if problems:
        raise SystemExit(
            "O repositório diverge do protocolo congelado:\n  " + "\n  ".join(problems)
        )
    return head


def _manifest(out: Path) -> dict | None:
    path = Path(out) / integrity.MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def check_preconditions(out: Path, root: Path, resume: bool) -> tuple[dict, list[str], str]:
    """Confere repositório, lock e vestígios; devolve (lock, combinações a avaliar, HEAD)."""
    head = check_repository(out, root)
    lock = read_lock(Path(out) / LOCK_NAME)
    planned = [
        unit_key(t, f, m) for t in lock["task_ids"] for f in lock["folds"] for m in lock["models"]
    ]
    if not resume:
        evidence = integrity.test_evidence(out, root)
        if evidence:
            raise SystemExit(
                "A avaliação final é única e já há vestígios de uma avaliação anterior. "
                "Use --resume apenas para completar combinações pendentes ou com erro:\n  "
                + "\n  ".join(evidence)
            )
        return lock, planned, head
    manifest = _manifest(out)
    if manifest is None:
        raise SystemExit("--resume exige test_manifest.json de uma execução anterior.")
    if manifest.get("lock_sha256") != lock_sha256(out):
        raise SystemExit("O lock atual difere do usado na avaliação anterior; --resume recusado.")
    status = manifest.get("status", {})
    lines = read_test_lines(Path(out) / integrity.TEST_FILE)
    rows = manifest.get("rows", {})
    bad = [
        k
        for k, state in status.items()
        if state == OK and (k not in lines or sha256_text(lines[k]) != rows.get(k))
    ]
    if bad:
        raise SystemExit(
            "test.csv não confere com o manifesto (linha ok ausente ou alterada) para "
            f"{bad[:5]}: --resume recusado."
        )
    todo = [k for k in planned if status.get(k, PENDING) in (PENDING, ERROR)]
    if not todo:
        raise SystemExit("Nenhuma combinação pendente ou com erro no manifesto: nada a retomar.")
    return lock, todo, head


# ------------------------------------------------------------------ execução


def _new_manifest(sha: str, lock: dict, planned: list[str]) -> dict:
    import sklearn

    return {
        "lock_sha256": sha,
        "lock_git_commit": lock.get("git_commit"),
        "models": lock["models"],
        "folds": lock["folds"],
        "task_ids": lock["task_ids"],
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "status": {k: PENDING for k in planned},
        "rows": {},
        "runs": [],
    }


def main(
    argv: list[str] | None = None, root: Path = REPO_ROOT, loader: Loader | None = None
) -> list[dict]:
    """Executa a avaliação final; `root` e `loader` permitem testes sem OpenML."""
    args = parse_args(argv)
    root = Path(root)
    out: Path = args.output_dir
    sha = lock_sha256(out)
    try:
        lock, todo, head = check_preconditions(out, root, args.resume)
    except SystemExit as exc:
        if out.exists():
            _log(out, "recusa", sha, integrity.git_head(root) or "", detail=str(exc))
        raise
    planned = [
        unit_key(t, f, m) for t in lock["task_ids"] for f in lock["folds"] for m in lock["models"]
    ]
    manifest = (_manifest(out) if args.resume else None) or _new_manifest(sha, lock, planned)
    for key in todo:
        manifest["status"][key] = PENDING
    run = {
        "started_at": _now(),
        "finished_at": None,
        "resume": args.resume,
        "git_head": head,
        "units": todo,
    }
    manifest["runs"].append(run)
    manifest_path = out / integrity.MANIFEST_FILE
    integrity.write_json(manifest_path, manifest)
    _log(
        out,
        integrity.START_EVENT,
        sha,
        head,
        detail=f"combinacoes={len(todo)}; resume={args.resume}",
    )

    test_path = out / integrity.TEST_FILE
    lines = read_test_lines(test_path)

    def on_row(key: str, row: dict) -> None:
        lines[key] = row_line(row)
        _write_test_csv(test_path, lines, planned)
        manifest["status"][key] = row["status"]
        if row["status"] == OK:
            manifest["rows"][key] = sha256_text(lines[key])
        else:
            manifest["rows"].pop(key, None)
        integrity.write_json(manifest_path, manifest)
        _log(out, "sistema", sha, head, key, row["status"], row.get("error", ""))

    run_args = argparse.Namespace(
        output_dir=out,
        seed=lock["seed"],
        n_trials=lock["budget"]["n_trials"],
        tune_timeout=lock["budget"]["tune_timeout"],
        tune_max_samples=lock["budget"]["tune_max_samples"],
    )
    loader = loader or openml_loader
    pending = set(todo)
    rows: list[dict] = []
    for task_id in lock["task_ids"]:
        if not any(k.startswith(f"{int(task_id)}|") for k in pending):
            continue
        ds, folds = loader(task_id, list(lock["folds"]))
        rows += evaluate_dataset(
            ds, folds, lock["folds"], lock["models"], run_args, pending, on_row
        )

    run["finished_at"] = _now()
    integrity.write_json(manifest_path, manifest)
    n_ok = sum(r["status"] == OK for r in rows)
    _log(out, integrity.END_EVENT, sha, head, detail=f"{n_ok} de {len(rows)} combinações ok")
    rel_out = integrity.relative_to_root(out, root)
    print(f"\nAvaliação final gravada em {test_path} (lock {sha[:12]}, commit {head[:12]}).")
    print(
        "Comite os resultados AGORA e confira o protocolo:\n"
        f'  git add {rel_out} && git commit -m "Avaliação final"\n'
        "  python -m src.pipeline.check_protocol"
    )
    return rows


if __name__ == "__main__":
    main()
