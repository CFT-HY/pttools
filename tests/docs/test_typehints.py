"""Unit tests for the fixes of the evaluation of the type hints by Sphinx."""

from pathlib import Path

import pytest

pytest.importorskip("sphinx", reason="Sphinx is not installed")

from pttools.docs.typehints import type_checking_imports


def test_type_checking_imports(tmp_path: Path) -> None:
    """Only the module-level imports ``from module import name`` under ``TYPE_CHECKING`` are found."""
    package = tmp_path / "package"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text(
        "import typing as tp\n"
        "if tp.TYPE_CHECKING:\n"
        "    from package.sub.a import A\n"
        "    from package.sub.b import B as C\n"
        "    import package.sub.d\n"
        "    from .e import E\n",
        encoding="utf-8")
    (package / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (package / "sub" / "mod.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from package.sub.a import A\n"
        "if True:\n"
        "    from package.sub.f import F\n"
        "def func():\n"
        "    if TYPE_CHECKING:\n"
        "        from package.sub.g import G\n",
        encoding="utf-8")
    assert type_checking_imports(package) == {
        "package": {"A": ("package.sub.a", "A"), "C": ("package.sub.b", "B")},
        "package.sub.mod": {"A": ("package.sub.a", "A")},
    }


def test_type_checking_imports_pttools() -> None:
    """The imports of PTtools that would cause circular imports are found."""
    imports = type_checking_imports(Path(__file__).parents[2] / "pttools")
    assert imports["pttools.analysis.parallel"]["Model"] == ("pttools.models.model", "Model")
