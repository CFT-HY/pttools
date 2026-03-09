"""Mathematic utilities"""

import numpy as np
from numpy.typing import NDArray


def rel_diff_arr[T: NDArray](x: T, y: T) -> T:
    """Relative differences of two arrays"""
    if not np.count_nonzero(y):
        return np.full_like(x, np.nan)
    nonzero = y != 0
    return np.abs(x - y)[nonzero] / np.abs(y[nonzero])


def rel_diff_scalar(x: float, y: float) -> float:
    """Relative difference of two scalars"""
    if y == 0:
        return np.nan
    return abs(x - y) / y
