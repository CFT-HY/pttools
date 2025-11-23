"""Utilities for handling docstrings"""

import logging
import typing as tp

logger = logging.getLogger(__name__)


def copy_docstring_without_params(target: tp.Any, source: tp.Callable) -> None:
    """Copy a docstring from source to target, excluding parameter documentation."""
    target.__doc__ = source.__doc__.split("\n:param", 1)[0]


def copy_docstrings_without_params(mapping: dict[tp.Any, tp.Callable]) -> tp.List[str]:
    """Copy docstrings from sources to targets, excluding parameter documentation.
    :param mapping: A dictionary of (target, source) pairs
    :return: A list of target names that already had docstrings
    """
    already_had_docstrings = []
    for target, source in mapping.items():
        if hasattr(target, "__doc__") and target.__doc__ is not None:
            if hasattr(target, "__name__"):
                name = target.__name__
            elif hasattr(target, "attrname"):
                name = target.attrname
            else:
                name = str(target)
            already_had_docstrings.append(name)
        copy_docstring_without_params(target, source)
    if already_had_docstrings:
        logger.warning(
            "These targets already had docstrings, which were overwritten: %s",
            already_had_docstrings
        )
    return already_had_docstrings
