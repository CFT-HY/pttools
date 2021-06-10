import numpy as np


def lorentz(xi, v):
    """
     Lorentz transformation of fluid speed v between moving frame and plasma frame.
    """
    return (xi - v)/(1 - v*xi)


def gamma2(v):
    """
     Square of Lorentz gamma
    """
    return 1./(1. - v**2)


def gamma(v):
    """
     Lorentz gamma
    """
    return np.sqrt(gamma2(v))
