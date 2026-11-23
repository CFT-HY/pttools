"""Utilities for floats with uncertainties of the uncertainties package and arrays of them."""

import typing as tp

import numpy as np
import uncertainties
from uncertainties import umath, unumpy

from pttools.type_hints import FloatOrArrU

__all__ = ["sqrt_u"]


def sqrt_u[T: FloatOrArrU](x: T) -> T:
    """Square root of a float, a float with an uncertainty, or an array of either of them.

    NumPy cannot compute the square root of a float with an uncertainty or of an array of them.
    The square root of :py:mod:`uncertainties.unumpy` gives an object array also for an array of floats,
    and a zero-dimensional array for a float with an uncertainty.
    Therefore, this uses :py:mod:`uncertainties.umath` for floats with uncertainties,
    :py:mod:`uncertainties.unumpy` for object arrays, and NumPy otherwise.

    :param x: the value
    :return: the square root of the value
    """
    if isinstance(x, uncertainties.UFloat):
        return tp.cast(T, umath.sqrt(x))  # pyrefly: ignore[missing-attribute]
    if isinstance(x, np.ndarray) and x.dtype == object:
        return tp.cast(T, unumpy.sqrt(x))  # pyrefly: ignore[missing-attribute]
    return tp.cast(T, np.sqrt(x))
