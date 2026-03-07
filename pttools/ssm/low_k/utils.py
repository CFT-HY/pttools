"""Utilities for the low-k SSM calculations"""

import numpy as np

import pttools.type_hints as th

#: Lorenzo eq. 3.7
# In the Sound Shell Model this is an integral instead of a constant
Iv: float = 1 / (32 * np.pi**2)

#: Another constant
Jv: float = 1 / (128 * np.pi**4)


def rho(z: th.FloatOrArr, x: th.FloatOrArr, cs: th.FloatOrArr) -> th.FloatOrArr:
    r"""Geometric function $\rho(z,x,\tilde{x})$
    $$\rho(z,x,\tilde{x}) = \frac{
    \left( \tilde{x}^2 - (x-z)^2 \right)^2
    \left( (x + z)^2 - \tilde{x}^2 \right)^2
    }{
    16 x \tilde{x} z^2
    }$$
    :giombi_2024_cs:`\ ` eq. 2.37
    """
    xp = 0.5*z * (1+cs)/cs
    xm = 0.5*z * (1-cs)/cs
    return ((xp+xm-x)**2 - (x-z)**2)**2 * ((xp+xm-x)**2 - (x+z)**2)**2 /16/x/(xp+xm-x)/z**2


def U(x: th.FloatOrArr, a: th.FloatOrArr) -> th.FloatOrArr:
    r"""Lorenzo eq. 3.13 stuff in the brackets"""
    return (1 - x**a) / a
