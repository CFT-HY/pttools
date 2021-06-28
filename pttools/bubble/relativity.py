"""Functions for calculating quantities from Einstein's special theory of relativity"""

import numpy as np

import pttools.type_hints as th


def lorentz(xi: th.FLOAT_OR_ARR, v: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Lorentz transformation of fluid speed v between moving frame and plasma frame.
    """
    return (xi - v)/(1 - v*xi)


def gamma2(v: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Square of Lorentz gamma, $\gamma^2$.
    """
    return 1./(1. - v**2)


def gamma(v: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Lorentz gamma, $\gamma = (1 - v^2)^{-\frac{1}{2}}$
    """
    return np.sqrt(gamma2(v))
