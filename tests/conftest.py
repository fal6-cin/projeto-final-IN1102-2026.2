"""Configuração dos testes.

Testes de aceitação (`tests/aceitacao/`) verificam componentes implementados pelo
grupo. Enquanto um componente levantar `NotImplementedError`, o teste aparece
como xfail ("ainda não implementado"); qualquer outro erro ou valor incorreto é
reportado como falha. Na correção, a variável de ambiente AM_STRICT=1 desliga
essa conversão.
"""

from __future__ import annotations

import os

import pytest

STRICT = os.environ.get("AM_STRICT") == "1"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if STRICT or call.excinfo is None or call.when not in ("setup", "call"):
        return
    if call.excinfo.errisinstance(NotImplementedError):
        report.outcome = "skipped"
        report.wasxfail = f"ainda não implementado: {call.excinfo.value}"


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "aceitacao" in item.nodeid.replace("\\", "/").split("/"):
            item.add_marker(pytest.mark.acceptance)
