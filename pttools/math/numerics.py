"""Numerical utilities."""

import typing as tp

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from pttools.speedup import threads
import pttools.type_hints as th

#: Smallest float
EPS: tp.Final[np.float64] = np.nextafter(0, 1)
#: Powers of 2 up to a certain value, see :py:func:`pttools.speedup.threads.powers_of_2`.
#: It's implemented in :py:mod:`pttools.speedup`, as :py:mod:`pttools.speedup` cannot depend on :py:mod:`pttools.math`.
powers_of_2 = threads.powers_of_2


def finite_edge(
        func: tp.Callable[[float], float],
        x_finite: float,
        x_nan: float,
        n_iter: int = 60) -> float:
    """Find the edge of the region where the function is finite with bisection.

    The function should be finite at ``x_finite`` and non-finite (e.g. nan) at ``x_nan``,
    and have a single edge between them.

    :param func: function to evaluate
    :param x_finite: point where the function is finite
    :param x_nan: point where the function is not finite
    :param n_iter: number of bisection steps. Each step halves the distance to the edge.
    :return: the last point found, where the function is finite
    """
    for _ in range(n_iter):
        x_mid = (x_finite + x_nan) / 2
        if np.isfinite(func(x_mid)):
            x_finite = x_mid
        else:
            x_nan = x_mid
    return x_finite


def is_nan_or_none(value: float | None = None) -> bool:
    """Determine if a value is NaN or None."""
    return value is None or np.isnan(value)


def rel_diff_arr[T: NDArray](x: T, y: T) -> T:
    """Relative differences of two arrays."""
    if not np.count_nonzero(y):
        return np.full_like(x, np.nan)
    nonzero = y != 0
    return np.abs(x - y)[nonzero] / np.abs(y[nonzero])


def rel_diff_scalar(x: float, y: float) -> float:
    """Relative difference of two scalars."""
    if y == 0:
        return np.nan
    return abs(x - y) / y


def running_mean(values: th.FloatArr1D, window: int) -> th.FloatArr1D:
    r"""Running mean with mirrored boundaries.

    :param values: the values
    :param window: number of points in the window, which must be odd
    :return: the running mean
    """
    return savgol_filter(values, window_length=window, polyorder=0, mode="mirror")
