"""★ P3-1: bảo vệ regression — docstring phải nằm TRƯỚC from __future__."""
import importlib
import pkgutil

import pytest

PACKAGES = ["core", "ingestion", "retrieval", "evaluation"]


def _iter_modules():
    for pkg in PACKAGES:
        for m in pkgutil.walk_packages([f"src/{pkg}"], prefix=f"{pkg}."):
            yield m.name


@pytest.mark.parametrize("module_name", list(_iter_modules()))
def test_module_has_docstring(module_name):
    mod = importlib.import_module(module_name)
    assert (mod.__doc__ or "").strip(), (
        f"{module_name} mất docstring — kiểm tra 'from __future__ import annotations' "
        f"có bị đặt TRƯỚC docstring không (xem review/pr5-eval-deps-docs.md P3-1)"
    )
