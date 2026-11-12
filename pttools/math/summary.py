"""Summaries of the values of parameters, e.g. for describing datasets."""

from collections.abc import Iterable
import typing as tp

import numpy as np

from pttools.math.formatting import compact

__all__ = ["FLOAT_DIGITS", "MAX_LISTED_VALUES", "summarize_values"]

#: Number of significant digits for comparing floats in :py:func:`summarize_values`
FLOAT_DIGITS: int = 12

#: The distinct values are listed by :py:func:`summarize_values` if there are at most this many of them
MAX_LISTED_VALUES: int = 10


def summarize_values(
        name: str,
        values: Iterable[tp.Any],
        description: str = "",
        max_listed: int = MAX_LISTED_VALUES,
        float_digits: int = FLOAT_DIGITS) -> str:
    """Summarize the values of a parameter on a single line.

    - Booleans: the number of true values.
    - Numbers: the range and the number of distinct values.
    - Other values, e.g. strings: the number of distinct values.
    - Constants: the value.

    The distinct values are listed if there are at most ``max_listed`` of them.
    Floats that are equal to ``float_digits`` significant digits are considered the same,
    so that rounding errors don't create distinct values.

    :param name: name of the parameter
    :param values: the values of the parameter
    :param description: description of the parameter
    :param max_listed: maximum number of distinct values to list
    :param float_digits: number of significant digits for comparing floats
    :return: the summary
    """
    arr = np.asarray(values if isinstance(values, np.ndarray) else list(values))
    suffix = f"  # {description}" if description else ""
    if arr.size == 0:
        return f"{name}: no values{suffix}"
    if arr.dtype == np.bool_:
        return f"{name}: {np.count_nonzero(arr)}/{arr.size} true{suffix}"
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.array([float(f"{value:.{float_digits}g}") for value in arr.flat])
    unique = np.unique(arr)
    if unique.size == 1:
        return f"{name}: {compact(unique[0])}{suffix}"
    if np.issubdtype(arr.dtype, np.number):
        text = f"{name}: {compact(unique[0])} ... {compact(unique[-1])}, {unique.size} distinct"
    else:
        text = f"{name}: {unique.size} distinct"
    if unique.size <= max_listed:
        text += f": [{', '.join(compact(value) for value in unique)}]"
    return text + suffix
