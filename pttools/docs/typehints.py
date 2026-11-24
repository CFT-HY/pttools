"""Fixes for the evaluation of the type hints by Sphinx.

As the modules use ``from __future__ import annotations``, the type hints are strings,
which Sphinx evaluates with :py:func:`typing.get_type_hints`.
If this fails, Sphinx shows the type hints as unevaluated strings, e.g. ``th.FloatArr1D``,
which cannot be linked to their targets. This happens in two cases, which are fixed here.

- The modules import the names that would cause circular imports only ``if tp.TYPE_CHECKING``,
  see :py:data:`pttools.type_hints.TYPE_CHECKING_OR_DOCS`. These names are therefore not defined
  when Sphinx evaluates the type hints, and if any type hint of a function fails,
  all the type hints of that function are left unevaluated.
  :py:func:`define_type_checking_imports` imports these names to the modules
  after all the modules can be imported.
  The ``autodoc_type_aliases`` option of Sphinx is not used for this, since it applies to all the modules,
  and its placeholders of the names are not rendered correctly within other types, e.g. ``list[Model]``.
- For a class, Sphinx evaluates the type hints of the class instead of those of its ``__init__``,
  and therefore the parameter types in the description of the class are not evaluated.
  :py:func:`record_class_typehints` records the evaluated type hints of ``__init__`` instead.
"""

import ast
import importlib
import logging
from pathlib import Path
import typing as tp

from sphinx.util.inspect import signature
from sphinx.util.typing import stringify_annotation

import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from sphinx.application import Sphinx
    from sphinx.ext.autodoc import Options

__all__ = ["TYPE_CHECKING_PACKAGES", "define_type_checking_imports", "record_class_typehints", "type_checking_imports"]

logger: logging.Logger = logging.getLogger(__name__)

#: The default packages whose type checking imports are defined by :py:func:`define_type_checking_imports`,
#: i.e. the default of the Sphinx configuration value ``type_checking_packages``.
#: Other projects that use :py:func:`pttools.docs.setup.setup_sphinx` should set it to their own packages.
#: The examples are not included, as importing an example without a main guard would run it.
TYPE_CHECKING_PACKAGES: tuple[str, ...] = ("pttools", "tests")


def _is_type_checking(test: ast.expr) -> bool:
    """Whether the condition of an if statement is ``TYPE_CHECKING``, e.g. ``tp.TYPE_CHECKING``."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING" \
        and isinstance(test.value, ast.Name) and test.value.id in ("tp", "typing")


def _module_name(path: Path, root: Path) -> str:
    """The name of the module of a Python file."""
    parts = path.relative_to(root).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def define_type_checking_imports(app: Sphinx | None = None, packages: tp.Iterable[str] | None = None) -> None:
    """Import the names that the modules of the given packages import only ``if TYPE_CHECKING``.

    The names are imported to the modules, so that Sphinx can evaluate the type hints that use them.
    The names that are already defined in a module are not changed.
    The packages that cannot be imported are skipped with a warning.
    This is a handler for the ``builder-inited`` event of Sphinx.

    :param app: the Sphinx application, whose configuration value ``type_checking_packages``
        gives the packages if ``packages`` is None
    :param packages: the names of the packages, defaults to those of ``app``,
        or :py:data:`TYPE_CHECKING_PACKAGES` if ``app`` is None
    """
    if packages is None:
        packages = TYPE_CHECKING_PACKAGES if app is None else app.config.type_checking_packages
    for package in packages:
        try:
            path = Path(importlib.import_module(package).__path__[0])
        except ImportError as e:
            logger.warning("Could not import the package %s to define its type checking imports: %s", package, e)
            continue
        for module_name, imports in type_checking_imports(path).items():
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                logger.warning("Could not import %s to define its type checking imports: %s", module_name, e)
                continue
            for name, (source, obj_name) in imports.items():
                if name in module.__dict__:
                    continue
                try:
                    setattr(module, name, getattr(importlib.import_module(source), obj_name))
                except (AttributeError, ImportError) as e:
                    logger.warning("Could not import %s from %s to %s: %s", obj_name, source, module_name, e)


def record_class_typehints(
        app: Sphinx, objtype: str, name: str, obj: object, options: Options, args: str, retann: str) -> None:
    """Record the evaluated type hints of the ``__init__`` of a class for the description of its parameters.

    This is a handler for the ``autodoc-process-signature`` event of Sphinx,
    and has to be run after ``sphinx.ext.autodoc.typehints.record_typehints``,
    whose type hints of the parameters of a class are not evaluated.
    """
    if objtype not in ("class", "exception") or app.config.autodoc_typehints not in ("both", "description"):
        return
    init = getattr(obj, "__init__", None)
    if init is None or init is object.__init__:
        return
    try:
        sig = signature(init, bound_method=True, type_aliases=app.config.autodoc_type_aliases)
    except (TypeError, ValueError):
        return
    mode = "smart" if app.config.autodoc_typehints_format == "short" else "fully-qualified"
    annotations = app.env.current_document.autodoc_annotations.setdefault(name, {})
    for param in sig.parameters.values():
        if param.annotation is not param.empty:
            annotations[param.name] = stringify_annotation(
                param.annotation, mode, short_literals=app.config.python_display_short_literal_types)


def type_checking_imports(directory: Path) -> dict[str, dict[str, tuple[str, str]]]:
    """Find the module-level imports ``from module import name`` that are done only ``if TYPE_CHECKING``.

    :param directory: the directory of a package, which is searched recursively
    :return: a mapping from the module names to mappings from the imported names
        to the modules and the names from which they are imported
    """
    root = directory.parent
    imports: dict[str, dict[str, tuple[str, str]]] = {}
    for path in sorted(directory.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path)
        for node in tree.body:
            if not (isinstance(node, ast.If) and _is_type_checking(node.test)):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom) and stmt.module is not None and stmt.level == 0:
                    module_imports = imports.setdefault(_module_name(path, root), {})
                    for alias in stmt.names:
                        module_imports[alias.asname or alias.name] = (stmt.module, alias.name)
    return imports
