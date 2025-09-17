"""Numerical utilities for SSMtools"""

import logging

import numba
import numpy as np

import pttools.type_hints as th
from . import const

logger = logging.getLogger(__name__)


@numba.njit
def resample_uniform_xi(
        xi: np.ndarray,
        f: th.FloatOrArr,
        n_xi: int = const.NPTDEFAULT[0]) -> tuple[np.ndarray, th.FloatOrArr]:
    r"""
    Provide uniform resample of function defined by $(x,y) = (\xi,f)$.
    Returns f interpolated and the uniform grid of n_xi points in range [0,1].

    :param xi: $\xi$
    :param f: function values $f$ at the points $\xi$
    :param n_xi: number of interpolated points
    """
    xi_re = np.linspace(0, 1-1/n_xi, n_xi)
    return xi_re, np.interp(xi_re, xi, f)
