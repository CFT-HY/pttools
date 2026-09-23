"""Mathematical utilities."""

import math
import typing as tp

import numpy as np
from numpy.typing import NDArray

import pttools.type_hints as th

#: Smallest float
EPS: tp.Final[np.float64] = np.nextafter(0, 1)


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


def powers_of_2(max_val: int, start_exp: int = 0, min_end_exp: int = 0, include_max: bool = False) -> th.IntArr1D:
    """Get the powers of 2 up to a certain value."""
    ret = [2 ** i for i in range(start_exp, max(min_end_exp, int(math.log2(max_val)) + 1))]
    # If max_val is not a power of 2
    if include_max and max_val.bit_count() != 1:
        ret.append(max_val)
    return np.array(ret)


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
