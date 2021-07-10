"""Faster or Numba-jitted versions of library functions"""

import numba
import numpy as np


@numba.njit(parallel=True)
def logspace(start: float, stop: float, num: int, base: float = 10.0) -> np.ndarray:
    """Numba version of numpy.logspace"""
    y = np.linspace(start, stop, num)
    return base**y
