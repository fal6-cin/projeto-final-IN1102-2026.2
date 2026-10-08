"""Fase de desenvolvimento: avalia os sistemas por CV interna no TREINO de cada fold externo.

Protocolo da pós-graduação (IN1102) (CV aninhada):
    - CV EXTERNA: os 3 folds oficiais do TabArena (repetição 0);
    - CV INTERNA: 3 folds estratificados dentro do treino de cada fold externo.

`load_task` carrega o dataset inteiro, mas este comando usa SOMENTE os índices
de treino dos folds externos (`data.load_tabarena.get_train_indices`): as linhas
de teste não participam de nenhum ajuste nem de nenhuma predição antes de
`final_test`. Para cada dataset, fold externo e sistema, ajusta o sistema em
cada treino interno e grava:

- `results/validation.csv`: métricas na validação interna, uma linha por
  (task_id, dataset, fold externo, fold interno, sistema);
- `results/validation_train.csv`: métricas no treino interno do mesmo estimador
  já ajustado, medidas numa subamostra estratificada de até 2.000 instâncias
  (base da análise de overfitting em `src.pipeline.overfitting`);
- `results/preds_val/`: probabilidades na validação interna
  (`{task_id}_fold{f}_inner{i}_{sistema}.npz`, com y_true, y_proba e classes),
  base da análise de erros por classe (`src.pipeline.class_errors`). Nunca há
  predições de teste nesta fase.

Para `group_tuned`, a busca Optuna (`tune.py`) é feita no treino do fold
externo; os hiperparâmetros escolhidos são reavaliados nas MESMAS partições
internas da busca. Essas linhas têm `optimistic = True`: a validação interna foi
usada na própria seleção e, portanto, superestima o desempenho. A coluna
`tune_best_score` guarda a melhor AUC OvO da busca.

`validation.csv` é APENAS DIAGNÓSTICO. O treino de um fold externo contém o
teste dos outros folds externos; por isso, os sistemas comparados, o controle
(`group_tuned`), o espaço de busca e o orçamento são fixados a priori. Depois
desta fase só se admitem correções de FALHA DE EXECUÇÃO (status error, contrato
do estimador), registradas no relatório; mudanças motivadas por desempenho não
são permitidas.

Os estudos do grupo (sensibilidade, estudo específico, curvas de complexidade)
são feitos em notebook com `src.pipeline.cv_grid.cv_grid` ou
`src.pipeline.overfitting.validation_curve`, sobre o treino de cada fold externo.

Depois desta fase: `python -m src.pipeline.freeze`, commit do lock e, por fim,
uma única vez, `python -m src.pipeline.final_test`. Este comando recusa executar
quando já há vestígios de avaliação final.

Exemplos:
    python -m src.pipeline.run_validation
    python -m src.pipeline.run_validation --models lightgbm group_default --task-ids 363621
    python -m src.pipeline.run_validation --resume
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from data.load_tabarena import (
    OFFICIAL_FOLDS,
    REPO_ROOT,
    TabularDataset,
    get_train_indices,
    load_task,
    read_selection,
    validate_selection,
)
from src.pipeline.cv_grid import inner_cv_splits, stratified_subsample
from src.pipeline.evaluate import METRICS, compute_metrics, fit_predict_evaluate
from src.pipeline.integrity import test_evidence
from src.pipeline.systems import ALL_MODELS, build_runner
from src.pipeline.tune import (
    DEFAULT_CV_FOLDS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_N_TRIALS,
    DEFAULT_TIMEOUT_S,
)

VALIDATION_FILE = "validation.csv"
VALIDATION_TRAIN_FILE = "validation_train.csv"
PREDS_VAL_DIR = "preds_val"
DEFAULT_TRAIN_EVAL_MAX = 2_000
UNIT_KEYS = ["task_id", "dataset", "fold", "model"]
KEYS = ["task_id", "dataset", "fold", "inner_fold", "model"]
VALIDATION_COLUMNS = KEYS + [
    *METRICS,
    "n_train",
    "n_val",
    "fit_time_s",
    "predict_time_s",
    "tune_time_s",
    "tune_best_score",
    "optimistic",
    "params",
    "status",
    "error",
]
TRAIN_COLUMNS = KEYS + ["n_train_eval", *METRICS, "status", "error"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="task IDs do OpenML; se omitido, usa data/selecao_datasets.txt",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(OFFICIAL_FOLDS))
    parser.add_argument("--inner-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--resume", action="store_true", help="pula combinações já concluídas")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--tune-timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--tune-max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="subamostra estratificada máxima do treino externo (busca e CV interna)",
    )
    parser.add_argument("--train-eval-max", type=int, default=DEFAULT_TRAIN_EVAL_MAX)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="não exige que a seleção cumpra as regras do enunciado (uso em testes parciais)",
    )
    return parser.parse_args(argv)


def done_keys(csv_path: Path) -> set[tuple]:
    """Combinações (task_id, dataset, fold, model) já concluídas com sucesso."""
    if not csv_path.exists():
        return set()
    done = pd.read_csv(csv_path)
    done = done[done["status"] == "ok"]
    return set(map(tuple, done[UNIT_KEYS].itertuples(index=False, name=None)))


def _append(csv_path: Path, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows).reindex(columns=columns)
    frame.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def _save_val_predictions(path: Path, y_true, y_proba, classes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_true=np.asarray(y_true).astype(str),
        y_proba=np.asarray(y_proba, dtype=np.float32),
        classes=np.asarray(classes).astype(str),
    )


def _evaluate_unit(
    model: str,
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    args: argparse.Namespace,
    preds_prefix: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """CV interna de um sistema no treino de um fold externo; devolve (linhas val, linhas treino).

    Com `preds_prefix`, grava as probabilidades de cada validação interna em
    `{preds_prefix}_inner{i}_{model}.npz`.
    """
    estimator, tune_time_s, info = build_runner(model, args)(X_dev, y_dev)
    X_cv, y_cv = stratified_subsample(X_dev, y_dev, args.tune_max_samples, args.seed)
    splits = inner_cv_splits(y_cv, args.inner_folds, args.seed)
    val_rows, train_rows = [], []
    for inner, (tr, va) in enumerate(splits):
        est = clone(estimator)
        result = fit_predict_evaluate(est, X_cv.iloc[tr], y_cv[tr], X_cv.iloc[va], y_cv[va])
        if preds_prefix is not None:
            path = preds_prefix.with_name(f"{preds_prefix.name}_inner{inner}_{model}.npz")
            _save_val_predictions(path, y_cv[va], result.y_proba, result.classes)
        val_rows.append(
            {
                "inner_fold": inner,
                **{m: getattr(result, m) for m in METRICS},
                "n_train": len(tr),
                "n_val": len(va),
                "fit_time_s": result.fit_time_s,
                "predict_time_s": result.predict_time_s,
                "tune_time_s": tune_time_s,
                "tune_best_score": info.get("best_score", np.nan),
                "optimistic": bool(info),
                "params": json.dumps(info["best_params"], default=str) if info else "",
                "status": "ok",
                "error": "",
            }
        )
        X_ev, y_ev = stratified_subsample(X_cv.iloc[tr], y_cv[tr], args.train_eval_max, args.seed)
        train = compute_metrics(y_ev, np.asarray(est.predict_proba(X_ev)), result.classes)
        train_rows.append(
            {"inner_fold": inner, "n_train_eval": len(y_ev), **train, "status": "ok", "error": ""}
        )
    return val_rows, train_rows


def validate_dataset(
    ds: TabularDataset,
    train_folds: list[np.ndarray],
    fold_ids: Iterable[int],
    models: Iterable[str],
    args: argparse.Namespace,
    done: set[tuple] = frozenset(),
) -> list[dict]:
    """Avalia os sistemas por CV interna no treino de cada fold externo.

    Args:
        train_folds: SOMENTE os índices de treino de cada fold externo
            (`get_train_indices`); o teste não é recebido.

    Grava `validation.csv`, `validation_train.csv` e `preds_val/` em `args.output_dir`.

    Returns:
        Linhas gravadas em `validation.csv` nesta chamada.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_path = args.output_dir / VALIDATION_FILE
    train_path = args.output_dir / VALIDATION_TRAIN_FILE
    written: list[dict] = []
    for fold_id, train_idx in zip(fold_ids, train_folds):
        X_dev, y_dev = ds.X.iloc[train_idx], np.asarray(ds.y)[train_idx]
        for model in models:
            key = (ds.task_id, ds.name, fold_id, model)
            if key in done:
                continue
            base = dict(zip(UNIT_KEYS, key))
            t0 = time.perf_counter()
            try:
                prefix = args.output_dir / PREDS_VAL_DIR / f"{ds.task_id}_fold{fold_id}"
                val_rows, train_rows = _evaluate_unit(model, X_dev, y_dev, args, prefix)
                val_rows = [{**base, **r} for r in val_rows]
                train_rows = [{**base, **r} for r in train_rows]
                auc = np.mean([r["auc_ovo"] for r in val_rows])
                print(
                    f"[{ds.name} | fold externo {fold_id}] {model}: AUC val. interna={auc:.4f} "
                    f"tempo={time.perf_counter() - t0:.1f}s"
                )
            except Exception as exc:  # noqa: BLE001 - registra a falha e segue
                error = f"{type(exc).__name__}: {exc}"
                val_rows = [{**base, "status": "error", "error": error}]
                train_rows = [{**base, "status": "error", "error": error}]
                print(f"[{ds.name} | fold externo {fold_id}] {model}: FALHA ({error})")
                traceback.print_exc()
            _append(val_path, val_rows, VALIDATION_COLUMNS)
            _append(train_path, train_rows, TRAIN_COLUMNS)
            written.extend(val_rows)
    return written


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> None:
    """Executa a fase de desenvolvimento; `root` (raiz do repositório) permite testes isolados."""
    args = parse_args(argv)
    traces = test_evidence(args.output_dir, root)
    if traces:
        raise SystemExit(
            "A avaliação final já foi executada: a fase de desenvolvimento está encerrada e "
            "não pode ser refeita.\n  " + "\n  ".join(traces)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_ids = args.task_ids if args.task_ids else read_selection()
    if not task_ids:
        raise SystemExit("Nenhum task ID informado. Preencha data/selecao_datasets.txt.")
    if not args.skip_validation and not args.task_ids:
        report = validate_selection(task_ids)
        for msg in report.warnings:
            print(f"Aviso: {msg}")
        if not report.ok:
            raise SystemExit(
                "Seleção inválida:\n  "
                + "\n  ".join(report.errors)
                + "\nUse --skip-validation apenas para execuções parciais."
            )

    val_path = args.output_dir / VALIDATION_FILE
    if val_path.exists() and not args.resume:
        raise SystemExit(
            f"{val_path} já existe. Use --resume para continuar a execução "
            "ou outro --output-dir para um novo estudo."
        )
    done = done_keys(val_path) if args.resume else set()
    for task_id in task_ids:
        ds = load_task(task_id)
        train_folds = get_train_indices(task_id, folds=args.folds)
        validate_dataset(ds, train_folds, args.folds, args.models, args, done)

    print(f"\nValidação interna gravada em {val_path} e {args.output_dir / VALIDATION_TRAIN_FILE}")


if __name__ == "__main__":
    main()
