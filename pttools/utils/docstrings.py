"""Utilities for handling docstrings"""

import logging
import typing as tp

logger = logging.getLogger(__name__)


class HasDocstring(tp.Protocol):
    __doc__: str | None


class WrappedDecoratorFunction(tp.Protocol):
    """A decorator that returns the decorated callable unchanged

    This is a callback protocol instead of a type alias,
    so that the signature of the decorated callable is preserved.
    """
    # pylint: disable=too-few-public-methods
    def __call__[**P, T](self, target: tp.Callable[P, T]) -> tp.Callable[P, T]: ...


def copy_docstring_dec(source: HasDocstring, without_params: bool = False) -> WrappedDecoratorFunction:
    """Copies the docstring of the given function to another.

    This function is intended to be used as a decorator.
    From: https://stackoverflow.com/a/68901244

    .. code-block:: python3

        def foo():
            '''This is a foo docstring'''
            ...

        @copy_doc(foo)
        def bar():
            ...
    """

    def wrapped[**P, T](target: tp.Callable[P, T]) -> tp.Callable[P, T]:
        copy_docstring(target, source, without_params=without_params)
        return target

    return wrapped


def copy_docstring(target: tp.Any, source: HasDocstring, without_params: bool = False) -> None:
    """Copy a docstring from source to target"""
    if without_params and source.__doc__ is not None:
        target.__doc__ = source.__doc__.split("\n:param", 1)[0]
    else:
        target.__doc__ = source.__doc__


def copy_docstrings(mapping: dict[tp.Any, HasDocstring], without_params: bool = False) -> tp.List[str]:
    """Copy docstrings from sources to targets
    :param mapping: A dictionary of (target, source) pairs
    :param without_params: Whether to exclude parameter documentation
    :return: A list of target names that already had docstrings
    """
    already_had_docstrings = []
    for target, source in mapping.items():
        if hasattr(target, "__doc__") and target.__doc__ is not None:
            already_had_docstrings.append(get_name(target))
        copy_docstring(target, source, without_params=without_params)
    if already_had_docstrings:
        logger.warning(
            "These targets already had docstrings, which were overwritten: %s",
            already_had_docstrings
        )
    return already_had_docstrings


def get_name(obj: tp.Any) -> str:
    """Get the name of an object"""
    if hasattr(obj, "__name__") and obj.__name__ is not None:
        return obj.__name__
    elif hasattr(obj, "attrname") and obj.attrname is not None:
        return obj.attrname
    return str(obj)
