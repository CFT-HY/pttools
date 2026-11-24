"""Unit tests for the fixes of the evaluation of the type hints by Sphinx."""

from collections.abc import Generator
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("sphinx", reason="Sphinx is not installed")

from pttools.docs.typehints import define_type_checking_imports, type_checking_imports


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


@pytest.fixture
def tc_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[str]:
    """An importable package, whose module imports a name only for type checking."""
    package = tmp_path / "tc_package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text("class A:\n    pass\n", encoding="utf-8")
    (package / "mod.py").write_text(
        "import typing as tp\n"
        "if tp.TYPE_CHECKING:\n"
        "    from tc_package.a import A\n",
        encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "tc_package"
    for name in [name for name in sys.modules if name == "tc_package" or name.startswith("tc_package.")]:
        del sys.modules[name]


def test_define_type_checking_imports_config(tc_package: str) -> None:
    """The packages are read from the configuration value ``type_checking_packages`` of the Sphinx application."""
    app = SimpleNamespace(config=SimpleNamespace(type_checking_packages=(tc_package,)))
    define_type_checking_imports(app)  # pyrefly: ignore[bad-argument-type]
    assert importlib.import_module(f"{tc_package}.mod").A is importlib.import_module(f"{tc_package}.a").A


def test_define_type_checking_imports_missing_package(tc_package: str, caplog: pytest.LogCaptureFixture) -> None:
    """A package that cannot be imported is skipped with a warning, and the other packages are processed."""
    define_type_checking_imports(packages=("nonexistent_package_of_pttools_tests", tc_package))
    assert importlib.import_module(f"{tc_package}.mod").A is importlib.import_module(f"{tc_package}.a").A
    assert "Could not import the package nonexistent_package_of_pttools_tests" in caplog.text
