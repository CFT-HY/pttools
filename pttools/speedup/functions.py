"""Faster or Numba-jitted versions of library functions"""

import numba
import numpy as np

# from pttools.speedup.options import NUMBA_NESTED_PARALLELISM
import pttools.type_hints as th
from pttools.type_hints import FloatArr, FloatArr1D


@numba.njit
def gradient[T: FloatArr](f: T) -> T:
    """Numba version of :func:`np.gradient`."""

    if f.ndim > 1:
        raise NotImplementedError

    out = np.empty_like(f)
    out[1:-1] = (f[2:] - f[:-2]) / 2.
    out[0] = f[1] - f[0]
    out[-1] = f[-1] - f[-2]
    return out


# @numba.njit(parallel=options.NUMBA_NESTED_PARALLELISM)
@numba.njit
def logspace(start: float, stop: float, num: int, base: float = 10.) -> th.FloatArr1D:
    """Numba version of :func:`numpy.logspace`."""
    y = np.linspace(start, stop, num)
    return base**y


@numba.njit
def resample_log(x: FloatArr1D, nx: int) -> FloatArr1D:
    """Resample a variable over a logarithmic range"""
    return logspace(np.log10(x.min()), np.log10(x.max()), num=nx)
