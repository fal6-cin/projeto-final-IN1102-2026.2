"""Catálogo do TabArena-v0.1, validação da seleção e leitura do arquivo de seleção."""

from __future__ import annotations

from data.load_tabarena import catalog, read_selection, validate_selection


def valid_selection() -> list[int]:
    cat = catalog()
    ids = []
    for regime in ("small", "medium", "large"):
        sub = cat[cat["regime_size"] == regime]
        multi = sub[sub["problem"] == "multiclass"]["task_id"].tolist()
        binary = sub[sub["problem"] == "binary"]["task_id"].tolist()
        ids += (multi[:2] + binary + multi[2:])[:10]
    return ids


def test_catalog_has_tabarena_classification_tasks():
    cat = catalog()
    assert len(cat) == 38
    assert cat["task_id"].between(363612, 363712).all()
    assert cat["regime_size"].value_counts().to_dict() == {"large": 16, "small": 12, "medium": 10}


def test_validate_selection_accepts_valid_choice():
    report = validate_selection(valid_selection())
    assert report.ok, report.errors
    assert not report.warnings


def test_validate_selection_rejects_ids_outside_suite():
    report = validate_selection([11, 14, 15] + valid_selection()[3:])
    assert not report.ok
    assert any("fora do catálogo" in e for e in report.errors)


def test_read_selection_ignores_comments(tmp_path):
    path = tmp_path / "selecao.txt"
    path.write_text("# comentário\n363621  # blood\n\n363629\n", encoding="utf-8")
    assert read_selection(path) == [363621, 363629]
