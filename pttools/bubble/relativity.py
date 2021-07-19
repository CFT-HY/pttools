"""Functions for calculating quantities from Einstein's special theory of relativity"""

import numba
import numpy as np

import pttools.type_hints as th


@numba.njit
def lorentz(xi: th.FLOAT_OR_ARR, v: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Lorentz transformation of fluid speed $v$ between moving frame and plasma frame:
    $\frac{\xi - v}{1 - v\xi}$.

    :param xi: $\xi = \frac{r}{t}$
    :param v: fluid speed $v$
    """
    return (xi - v)/(1 - v*xi)


@numba.njit
def gamma2(v: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Square of Lorentz gamma, $\gamma^2 = \frac{1}{1 - v^2}$.

    :param v: [fluid] speed $v$
    :return: $\gamma^2$
    """
    return 1./(1. - v**2)


@numba.njit
def gamma(v: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Lorentz gamma, $\gamma = (1 - v^2)^{-\frac{1}{2}}$.

    :param v: [fluid] speed $v$
    :return: $\gamma$
    """
    return np.sqrt(gamma2(v))
