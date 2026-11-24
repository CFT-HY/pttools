"""Document the vectorized functions, which autodoc would skip.

Autodoc documents only the members of a module that it recognizes as functions, classes or documented attributes.
The functions vectorized with :py:class:`numpy.vectorize` or :py:func:`numba.vectorize` are objects that are neither,
and autodoc therefore skips them, even if they are in ``__all__``.
:py:func:`add_vectorized_functions` adds them to the documentation of their module
with the ``autofunction`` directive, which can document them.
"""

import functools
import inspect
import types
import typing as tp

import numpy as np
from sphinx.util import inspect as sphinx_inspect

import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from sphinx.application import Sphinx

__all__ = ["add_vectorized_functions", "vectorized_function"]


def add_vectorized_functions(
        app: Sphinx | None, what: str, name: str, obj: object, options: tp.Any, lines: list[str]) -> None:
    """Add the vectorized functions of a module to its documentation.

    This is a handler for the ``autodoc-process-docstring`` event of Sphinx.
    The ``autofunction`` directives are added after the docstring of the module,
    and therefore the vectorized functions are documented before the other members of the module.
    Sphinx does not emit the event for a module without a docstring.
    The signatures of the functions vectorized with :py:class:`numpy.vectorize` are taken from the original functions,
    which are set as their ``__wrapped__`` attributes with :py:func:`functools.update_wrapper`.

    :param app: the Sphinx application, which is not used
    :param what: the type of the documented object
    :param name: the full name of the documented object
    :param obj: the documented object
    :param options: the options of the autodoc directive
    :param lines: the lines of the docstring, to which the directives are added
    """
    if what != "module" or not isinstance(obj, types.ModuleType):
        return
    members = getattr(options, "members", None)
    if not members:
        return
    if isinstance(members, list):
        member_names = members
    else:
        module_all = getattr(obj, "__all__", None)
        member_names = list(module_all) if module_all is not None \
            else [member_name for member_name in vars(obj) if not member_name.startswith("_")]
    excluded = getattr(options, "exclude_members", None) or set()

    for member_name in member_names:
        if member_name in excluded:
            continue
        member = getattr(obj, member_name, None)
        func = vectorized_function(member)
        if func is None or func.__module__ != obj.__name__:
            continue
        if isinstance(member, np.vectorize) and not hasattr(member, "__wrapped__"):
            functools.update_wrapper(member, func)
        lines.extend(["", f".. autofunction:: {name}.{member_name}"])


def vectorized_function(obj: object) -> types.FunctionType | None:
    """The original Python function of a vectorized function, which autodoc would not document.

    :param obj: a member of a module
    :return: the original function, or None if the object is not a vectorized function,
        or if autodoc documents it as a function
    """
    if isinstance(obj, type) or sphinx_inspect.isroutine(obj) or sphinx_inspect.isfunction(obj):
        return None
    func = getattr(obj, "pyfunc", None) if isinstance(obj, np.vectorize) else getattr(obj, "__wrapped__", None)
    return func if inspect.isfunction(func) else None
