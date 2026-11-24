"""Resolve the Python cross-references whose target is not documented with the name given in the reference.

Sphinx documents each object in the module where it's defined, e.g. :py:class:`pttools.bubble.bubble.bubble.Bubble`,
but the references in the docstrings and the evaluated type hints may use other names for the same object.
:py:func:`resolve_alias_reference` resolves the references whose target is not found with these names:

- The names of the objects that are re-exported by a package, e.g. ``pttools.bubble.bubble.Bubble``,
  are resolved to the module where they are defined by importing them.
  The inherited attributes, e.g. ``pttools.bubble.bubble.Bubble.kappa``,
  are resolved to the class where they are defined, as the inherited members are not documented.
  The unqualified names, e.g. ``LatticeRun`` in the docstring of :py:mod:`pttools.lattice`,
  are looked up in the class and the module of the docstring.
- The names of the objects of other projects in their private modules,
  e.g. ``numpy._typing._array_like.ArrayLike``, which are given by the evaluated type hints,
  are replaced with their public names of :py:data:`REFERENCE_ALIASES`.
- The references to the objects of other projects whose type in their intersphinx inventory
  differs from the role of the reference, e.g. a type hint ``numpy.typing.ArrayLike``,
  which is a ``py:class`` reference to a ``py:data`` object, are resolved regardless of the type.
"""

import importlib
import logging
import re
from types import ModuleType

from docutils import nodes
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.intersphinx import missing_reference

__all__ = [
    "REFERENCE_ALIASES",
    "RESOLVED_PACKAGES",
    "canonical_name",
    "canonical_target",
    "resolve_alias_reference",
]

logger: logging.Logger = logging.getLogger(__name__)

#: Patterns of the names in the private modules of other projects,
#: and the public names with which these objects are documented in their intersphinx inventories.
#: The first matching pattern is replaced with :py:func:`re.sub`.
REFERENCE_ALIASES: dict[str, str] = {
    r"^_thread\.lock$": "_thread.LockType",
    r"^concurrent\.futures\.\w+\.": "concurrent.futures.",
    # The Python documentation documents the profilers of cProfile and profile together as profile.Profile.
    r"^cProfile\.Profile$": "profile.Profile",
    # The Numba documentation documents CFunc without its module.
    r"^numba\.core\.ccallback\.CFunc$": "CFunc",
    r"^numpy\._typing\._\w+\.": "numpy.typing.",
    r"^plotly\.graph_objs\._\w+\.": "plotly.graph_objects.",
    r"^unittest\.case\.": "unittest.",
}
#: The packages whose objects are resolved to the modules where they are defined.
RESOLVED_PACKAGES: tuple[str, ...] = ("examples", "pttools", "tests")


def canonical_name(name: str) -> str | None:
    """The name of an object in the module where it's defined, which is where Sphinx documents it.

    The attributes without a name of their own, e.g. properties and constants,
    are given as the attributes of the object that contains them.

    :param name: full name of the object, e.g. ``pttools.bubble.bubble.Bubble.v_wall``
    :return: the canonical name, or None if the object cannot be imported
    """
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        canonical = ".".join(parts[:i])
        for j, part in enumerate(parts[i:], start=i):
            if isinstance(obj, type):
                # The inherited attributes are documented only in the class where they are defined.
                owner = next((cls for cls in obj.__mro__ if part in vars(cls)), obj)
                canonical = f"{owner.__module__}.{owner.__qualname__}"
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return None
            if isinstance(obj, ModuleType):
                canonical = obj.__name__
                continue
            module = getattr(obj, "__module__", None)
            qualname = getattr(obj, "__qualname__", None)
            if isinstance(module, str) and isinstance(qualname, str) and qualname.endswith(part):
                canonical = f"{module}.{qualname}"
            else:
                # The remaining parts are attributes without a name of their own.
                return ".".join([canonical, *parts[j:]])
        return canonical
    return None


def canonical_target(target: str, module: str | None, cls: str | None) -> str | None:
    """The canonical name of the target of a reference to an object of :py:data:`RESOLVED_PACKAGES`.

    An unqualified target, e.g. ``run_params``, is looked up in the class and the module
    of the docstring in which it's referred to, as Python would look up the name.

    :param target: target of the reference
    :param module: the module of the docstring in which the reference is
    :param cls: the class of the docstring in which the reference is
    :return: the canonical name, or None if the target is not found
    """
    target = target.removeprefix(".")
    candidates: list[str] = []
    if module is not None and module.split(".", 1)[0] in RESOLVED_PACKAGES:
        if cls is not None:
            candidates.append(f"{module}.{cls}.{target}")
        candidates.append(f"{module}.{target}")
    if target.split(".", 1)[0] in RESOLVED_PACKAGES:
        candidates.append(target)
    for candidate in candidates:
        canonical = canonical_name(candidate)
        if canonical is not None:
            return canonical
    return None


def resolve_alias_reference(
        app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: nodes.TextElement
        ) -> nodes.reference | None:
    """Resolve a Python cross-reference whose target is documented with another name.

    This is a handler for the ``missing-reference`` event of Sphinx.

    :return: the resolved reference, or None if the target is not found with another name either
    """
    if node.get("refdomain") != "py":
        return None
    target: str = node["reftarget"]
    new_target: str = target
    for pattern, alias in REFERENCE_ALIASES.items():
        if re.search(pattern, target):
            new_target = re.sub(pattern, alias, target, count=1)
            break
    else:
        new_target = canonical_target(target, node.get("py:module"), node.get("py:class")) or target
    if new_target == target and node["reftype"] == "obj":
        return None

    # The role "obj" matches the objects of all types.
    for reftype in dict.fromkeys((node["reftype"], "obj")):
        new_node = node.deepcopy()
        new_node["reftarget"] = new_target
        new_node["reftype"] = reftype
        resolved = env.get_domain("py").resolve_xref(
            env, node["refdoc"], app.builder, reftype, new_target, new_node, contnode)
        if resolved is None:
            resolved = missing_reference(app, env, new_node, contnode)
        if resolved is not None:
            return resolved
    return None
