"""Functions for calculating quantities from Einstein's special theory of relativity"""

import numba
import numpy as np

import pttools.type_hints as th
from pttools.type_hints import FloatOrArr


@numba.njit(cache=True)
def gamma[T: FloatOrArr](v: T) -> T:
    r"""
    Lorentz gamma, $\gamma = (1 - v^2)^{-\frac{1}{2}}$.

    :param v: [fluid] speed $v$
    :return: Lorentz $\gamma$
    """
    return np.sqrt(gamma2(v))


@numba.njit(cache=True, error_model="numpy")
def gamma2[T: FloatOrArr](v: T) -> T:
    r"""
    Square of Lorentz gamma, $\gamma^2 = \frac{1}{1 - v^2}$.

    :param v: [fluid] speed $v$
    :return: $\gamma^2$
    """
    return 1. / (1. - v**2)


@numba.njit(cache=True)
def lorentz(xi: th.FloatOrArr, v: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Lorentz transformation $\mu$ of fluid speed $v$ between moving frame and plasma frame:
    $$\mu = \frac{\xi - v}{1 - v\xi}$$.
    :notes:`\ ` eq. 7.32
    This function is its own inverse:
    $$v = \frac{\xi - \mu}{1 - \mu\xi}$$.
    However, be careful with the signs.

    :param xi: $\xi = \frac{r}{t}$
    :param v: fluid speed $v$
    """
    return (xi - v) / (1. - v*xi)
