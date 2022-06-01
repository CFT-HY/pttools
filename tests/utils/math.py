import numpy as np


def rel_diff_arr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Relative differences of two arrays"""
    nonzero = y != 0
    return np.abs(x - y)[nonzero] / np.abs(y[nonzero])


def rel_diff_scalar(x: float, y: float) -> float:
    """Relative difference of two scalars"""
    if y == 0:
        return np.nan
    return abs(x - y) / y
