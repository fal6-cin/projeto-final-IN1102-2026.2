"""Os notebooks de desenvolvimento (01 a 05) não podem acessar o teste dos folds externos."""

from __future__ import annotations

import io
import json
import re
import tokenize

import pytest

from data.load_tabarena import REPO_ROOT

NOTEBOOKS = REPO_ROOT / "notebooks"
# Padrões proibidos em células de código de TODOS os notebooks 01 a 05.
FORBIDDEN = {
    "índices de teste de get_folds": r"\bget_folds\s*\(",
    "funções internas do final_test": r"\b(?:final_test|openml_loader|evaluate_dataset)\b",
}
# Resultados de teste só podem ser lidos no notebook 04 (análise da avaliação final);
# os notebooks 01 a 03 e o 05 (overfitting) usam apenas treino e validação.
FORBIDDEN_OUTSIDE_04 = {
    "resultados de teste (test.csv)": r"test\.csv|load_raw",
    "predições de teste": r"preds_test|['\"/]preds['\"/]",
    "manifesto ou log do teste": r"test_manifest|test_log",
}


def _strip_comments(source: str) -> str:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        return source
    return tokenize.untokenize(t for t in tokens if t.type != tokenize.COMMENT)


def _code(path) -> str:
    """Código das células (sem comentários) de um notebook."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        _strip_comments("".join(c["source"]))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    )


def _development_notebooks():
    paths = sorted(p for p in NOTEBOOKS.glob("0[1-5]_*.ipynb"))
    assert len(paths) == 5, paths
    return paths


@pytest.mark.parametrize("path", _development_notebooks(), ids=lambda p: p.name)
def test_development_notebooks_do_not_touch_test(path):
    code = _code(path)
    number = int(path.name[:2])
    rules = dict(FORBIDDEN)
    if number != 4:
        rules.update(FORBIDDEN_OUTSIDE_04)
    found = [name for name, pattern in rules.items() if re.search(pattern, code)]
    assert not found, f"{path.name} referencia acesso ao teste: {found}"


def test_scan_detects_forbidden_access():
    code = "train_idx, test_idx = get_folds(task_id)[0]\nraw = pd.read_csv('results/test.csv')"
    assert re.search(FORBIDDEN["índices de teste de get_folds"], code)
    assert re.search(FORBIDDEN_OUTSIDE_04["resultados de teste (test.csv)"], code)
    assert not re.search(FORBIDDEN["índices de teste de get_folds"], "get_train_indices(t)")
    assert re.search(FORBIDDEN["funções internas do final_test"], "final_test.main([])")
    preds = FORBIDDEN_OUTSIDE_04["predições de teste"]
    assert re.search(preds, "RESULTS / 'preds'") and not re.search(preds, "RESULTS / 'preds_val'")
