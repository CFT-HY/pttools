"""Fixes for the backreferences of Sphinx-Gallery

Sphinx-Gallery records an object by the module from which the example imports it,
but Sphinx documents the object by the module in which it is defined.
For classes Sphinx-Gallery finds the defining module by inspecting the object,
but for functions and other objects it does not.
For example, the examples import ``power_gw_bag`` from :mod:`pttools.ssm`,
but it is defined in and documented as a part of :mod:`pttools.ssm.spectrum_bag`,
and therefore neither the hyperlinks from the examples nor the mini-galleries were created for it.

The functions below add the defining module of each object to the code objects of Sphinx-Gallery.
This has to be done by patching Sphinx-Gallery, as it does not provide a hook for this.
The patch is applied in the process that reads ``conf.py``,
which is sufficient as long as the ``parallel`` option of Sphinx-Gallery is disabled.
"""

import importlib
import inspect
import sys
import typing as tp

from sphinx_gallery import backreferences, gen_rst

from docs.utils import DOC_MODULES, MISSING, is_same_object, resolve_object

#: A code object of Sphinx-Gallery, which describes an object that is used in an example
type CodeObject = dict[str, tp.Any]


def ensure_imported(module: str) -> None:
    """Import a module of this repository, if it has not been imported yet

    Sphinx-Gallery resolves the objects only from the already imported modules,
    since importing arbitrary third-party modules may have side effects.
    The modules of this repository are imported by autodoc anyway,
    so importing them here is safe, and it's needed for the builds
    in which the examples are not executed, e.g. "make html-noplot".
    """
    if module in sys.modules or module.split(".")[0] not in DOC_MODULES:
        return
    try:
        importlib.import_module(module)
    except ImportError:
        pass


def defining_submodule(package: str, name: str, obj: tp.Any) -> str | None:
    """Find the submodule of the given package in which the given value is defined

    Values such as constants don't have the __module__ attribute,
    so the submodules are searched for an annotated assignment of the same object.
    A re-export is not an annotated assignment, and therefore only the defining submodule matches.

    :param package: name of the package from which the value is imported
    :param name: name of the value
    :param obj: the value itself
    :return: name of the defining submodule, or None if it cannot be determined unambiguously
    """
    if not hasattr(sys.modules.get(package), "__path__"):
        return None
    submodules = [
        submodule for submodule, module in list(sys.modules.items())
        if submodule.startswith(f"{package}.")
        and name in getattr(module, "__annotations__", {})
        and getattr(module, name, MISSING) is obj
    ]
    return submodules[0] if len(submodules) == 1 else None


def defining_module(cobj: CodeObject) -> str | None:
    """Find the module in which the object of the given code object is defined

    :param cobj: code object of Sphinx-Gallery
    :return: name of the defining module, or None if it's unknown or already recorded
    """
    ensure_imported(cobj["module"])
    attrs = cobj["name"].split(".")
    obj = resolve_object(cobj["module"], attrs)
    if obj is MISSING:
        return None
    module = getattr(obj, "__module__", None)
    if not isinstance(module, str):
        module = defining_submodule(cobj["module"], attrs[0], obj) if len(attrs) == 1 else None
    if module is None or module == cobj["module"]:
        return None
    # The __module__ attribute may have been inherited from a parent class,
    # in which case the object is not accessible from the module it points to.
    if not is_same_object(resolve_object(module, attrs), obj):
        return None
    return module


def add_defining_modules(code_objects: dict[str, list[CodeObject]]) -> dict[str, list[CodeObject]]:
    """Add code objects that refer to the modules in which the objects are defined"""
    for cobjs in code_objects.values():
        for cobj in list(cobjs):
            module = defining_module(cobj)
            if module is None or any(
                    other["module"] == module and other["name"] == cobj["name"] for other in cobjs):
                continue
            # Sphinx-Gallery uses the first code object that it manages to resolve,
            # so the new one has to be the first.
            cobjs.insert(0, {
                **cobj,
                "module": module,
                # pylint: disable-next=protected-access
                "module_short": backreferences._get_short_module_name(module, cobj["name"]) or module,
                "is_class": inspect.isclass(resolve_object(module, cobj["name"].split(".")))
            })
    return code_objects


def identify_names(*args, **kwargs) -> dict[str, list[CodeObject]]:
    """Wrapper for the identify_names function of Sphinx-Gallery, which adds the defining modules"""
    return add_defining_modules(backreferences.identify_names(*args, **kwargs))


def patch_sphinx_gallery() -> None:
    """Patch Sphinx-Gallery to also record the modules in which the objects are defined"""
    gen_rst.identify_names = identify_names
