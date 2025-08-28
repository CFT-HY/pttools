r"""
This is the example code from
:giese_2020:`\ `

Commented for better readability.
"""

import numpy as np
from scipy.integrate import odeint, simpson


def kappaNuModel(cs2: float, al: float, vp: float) -> float:
    r"""Calculate the efficiency factor $\kappa_\bar{\theta}$

    :param cs2: speed of sound squared
    :param al: strength parameter $\alpha$
    :param vp:
    """
    nu = 1./cs2 + 1.
    tmp = 1. - 3.*al + vp**2 * (1./cs2 + 3.*al)
    disc = 4*vp**2 * (1. - nu) + tmp**2
    if disc < 0:
        print("vp too small for detonation")
        return 0
    vm = (tmp + np.sqrt(disc))/2/(nu-1.)/vp
    wm = (-1. + 3.*al + (vp/vm)*(-1. + nu + 3.*al))
    wm /= (-1. + nu - vp/vm)

    def dfdv(xiw, v, nu):
        """Integrand"""
        xi, w = xiw
        dxidv = (((xi-v)/(1.-xi*v))**2*(nu-1.)-1.)
        dxidv *= (1.-v*xi)/2./v/(1.-v**2)
        dwdv = nu*(xi-v)/(1.-xi*v)*w/(1.-v**2)
        return [dxidv, dwdv]

    n = 501  # change accuracy here
    vs = np.linspace((vp-vm)/(1.-vp*vm), 0, n)
    sol = odeint(dfdv, [vp, 1.], vs, args=(nu,))
    xis, ws = sol[:, 0], -sol[:, 1]*wm/al*4./vp**3

    return simpson(ws*(xis*vs)**2/(1.-vs**2), xis)
