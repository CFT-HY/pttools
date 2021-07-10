"""Faster or Numba-jitted versions of library functions"""

import numba
from numba.extending import overload
import numpy as np


@overload(np.any)
def np_any(a):
    if isinstance(a, numba.types.Boolean):
        def func(a):
            return a
        return func
    if isinstance(a, numba.types.Number):
        def func(a):
            return bool(a)
        return func

# @overload(np.asanyarray)
# def asanyarray(arr: np.ndarray):
#     if isinstance(arr, numba.types.Array):
#         def func(arr: np.ndarray):
#             return arr
#         return func
#     raise NotImplementedError
#
#
# @overload(np.ndim)
# def ndim(val):
#     if isinstance(val, numba.types.Number):
#         def func(val):
#             return 0
#         return func
#     if isinstance(val, numba.types.Array):
#         def func(val):
#             return val.ndim
#         return func
#     raise NotImplementedError


@numba.njit
def gradient(f: np.ndarray):
    """np.gradient() for Numba"""

    if f.ndim > 1:
        raise NotImplementedError

    out = np.empty_like(f)
    out[1:-1] = (f[2:] - f[:-2]) / 2.
    out[0] = (f[1] - f[0])
    out[-1] = (f[-1] - f[-2])
    return out


@numba.njit(parallel=True)
def logspace(start: float, stop: float, num: int, base: float = 10.0) -> np.ndarray:
    """Numba version of numpy.logspace"""
    y = np.linspace(start, stop, num)
    return base**y
