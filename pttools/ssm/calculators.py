"""Numerical utilities for the Sound Shell Model"""

import logging

import numba
import numpy as np

from pttools.ssm import const
import pttools.type_hints as th

logger = logging.getLogger(__name__)


@numba.njit
def resample_uniform_xi(
        xi: th.FloatArr1D,
        f: th.FloatOrArr,
        n_xi: int = const.NPTDEFAULT[0]) -> tuple[th.FloatArr1D, th.FloatOrArr]:
    r"""
    Provide uniform resample of function defined by $(x,y) = (\xi,f)$.
    Returns f interpolated and the uniform grid of n_xi points in range [0,1].

    :param xi: $\xi$
    :param f: function values $f$ at the points $\xi$
    :param n_xi: number of interpolated points
    """
    xi_re = np.linspace(0, 1-1/n_xi, n_xi)
    return xi_re, np.interp(xi_re, xi, f)


@numba.njit
def trapezoid_loglog(x: th.FloatArr1D, y: th.FloatArr1D, minus1_atol: float = 1e-12) -> float:
    """Power-law (log-log) trapezoidal integration

    Based on https://scicomp.stackexchange.com/a/31374
    """
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("x and y must be positive for log-log trapezoid integration.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing for log-log trapezoid integration.")
    log_x = np.log(x)
    log_y = np.log(y)

    # Power law for each step: y = k * x**m
    # In log-log space: log(y) = m * log(x) + log(k)
    m = np.diff(log_y) / np.diff(log_x)
    # k = e^n, where n = log(y) - m * log(x)
    k = y[:-1] / (x[:-1]**m)
    mp1 = m + 1.
    close_to_minus1 = np.isclose(mp1, 0., rtol=0., atol=minus1_atol)
    mp1_not_close_to_minus1 = mp1[~close_to_minus1]

    integral = np.empty_like(m)
    integral[close_to_minus1] = k[close_to_minus1] * np.log(x[1:][close_to_minus1] / x[:-1][close_to_minus1])
    integral[~close_to_minus1] = (
        k[~close_to_minus1] *
        (x[1:][~close_to_minus1]**mp1_not_close_to_minus1 - x[:-1][~close_to_minus1]**mp1_not_close_to_minus1) /
        mp1_not_close_to_minus1
    )
    return np.sum(integral)
