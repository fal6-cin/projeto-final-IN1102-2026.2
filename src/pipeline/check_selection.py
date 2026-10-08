"""Confere a seleção de datasets do grupo (data/selecao_datasets.txt).

Uso:
    python -m src.pipeline.check_selection
    python -m src.pipeline.check_selection --markdown results/tabela_datasets.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.load_tabarena import SELECTION_PATH, read_selection, summarize, validate_selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--selection", type=Path, default=SELECTION_PATH)
    parser.add_argument(
        "--markdown", type=Path, default=None, help="grava a tabela da seleção em Markdown"
    )
    args = parser.parse_args(argv)

    report = validate_selection(read_selection(args.selection))
    print(
        report.table[["task_id", "name", "n_samples", "n_classes", "regime_size"]].to_string(
            index=False
        )
    )
    print()
    for msg in report.warnings:
        print(f"Aviso: {msg}")
    for msg in report.errors:
        print(f"Erro: {msg}")
    print("\nSeleção válida." if report.ok else "\nSeleção inválida.")

    if args.markdown is not None and len(report.table):
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        table = summarize(report.table["task_id"].tolist())
        args.markdown.write_text(table.to_markdown(index=False, floatfmt=".2f"), encoding="utf-8")
        print(f"Tabela gravada em {args.markdown}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
