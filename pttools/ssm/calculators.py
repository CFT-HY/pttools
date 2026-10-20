"""Numerical utilities for the Sound Shell Model."""

import logging

import numpy as np

from pttools.speedup import njit
from pttools.ssm import const
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)


@njit(cache=True, nogil=True)
def resample_uniform_xi(
        xi: th.FloatArr1D,
        f: th.FloatArr1D,
        n_xi: int = const.DEFAULT_N_XI_SSM) -> tuple[th.FloatArr1D, th.FloatArr1D]:
    r"""
    Provide uniform resample of function defined by $(x,y) = (\xi,f)$.

    Returns f interpolated and the uniform grid of n_xi points in range [0,1].

    :param xi: $\xi$
    :param f: function values $f$ at the points $\xi$
    :param n_xi: number of interpolated points
    """
    xi_re = np.linspace(0, 1-1/n_xi, n_xi)
    return xi_re, np.interp(xi_re, xi, f)  # pyrefly: ignore[bad-return]
