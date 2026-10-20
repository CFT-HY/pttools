"""Interpolation utilities."""

import numpy as np

from pttools.speedup import njit
import pttools.type_hints as th


@njit(cache=True, nogil=True)
def interpolate_log_log(x: th.FloatArr1D, xp: th.FloatArr1D, fp: th.FloatArr1D) -> th.FloatArr1D:
    r"""Interpolate linearly in log-log scale.

    $$\ln f(x) = \ln f_i + \frac{\ln x - \ln x_i}{\ln x_{i+1} - \ln x_i} (\ln f_{i+1} - \ln f_i),
    \quad x_i \leq x \leq x_{i+1}$$
    This is exact for power laws $f \propto x^n$.
    If the values are not all positive, the interpolation is linear in $\ln x$ instead,
    as their logarithms are not defined.
    Outside the range of $x_p$, the values at the endpoints are used, as in :external:py:func:`numpy.interp`.

    :param x: $x$, the points at which to interpolate
    :param xp: $x_p$, the points of the data, which must be positive and increasing
    :param fp: $f_p$, the values of the data
    :return: $f(x)$, the interpolated values
    """
    log_x = np.log(x)
    log_xp = np.log(xp)
    if np.all(fp > 0):
        return np.exp(np.interp(log_x, log_xp, np.log(fp)))
    return np.interp(log_x, log_xp, fp)
