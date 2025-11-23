r"""
This is the example code from
:giese_2021:`\ `

Commented for better readability.
"""

import logging
import typing as tp

import numpy as np
from scipy.integrate import odeint, simpson

logger = logging.getLogger(__name__)


def mu(xi, v):
    """Relative velocity (special relativistic)"""
    return (xi - v)/(1. - xi*v)


def getwow(v1, v2):
    """Ratio of enthalpies across the bubble wall, "w over w"
    from the junction conditions

    :param v1: $v_a$
    :param v2: $v_b$
    """
    return v1/(1. - v1 ** 2)/v2*(1. - v2 ** 2)


def getvm(al: float, vw: float, cs2b: float) -> tp.Tuple[float, int]:
    r"""Fluid velocity behind the wall, $\tilde{v}_-$, and the expansion mode
    0 = deflagration
    1 = hybrid
    2 = detonation

    :return: $\tilde{v}_-$, sol_type (0-2)
    """
    # If the wall velocity is smaller than the speed of sound in the broken phase, it's a subsonic deflagration.
    if vw**2 < cs2b:
        return vw, 0
    cc = 1. - 3. * al + vw**2 * (1./cs2b + 3.*al)
    # Discriminant
    disc = -4.*vw**2/cs2b + cc**2
    # If there is no detonation solution, then it's a hybrid.
    if disc < 0. or cc < 0.:
        return np.sqrt(cs2b), 1
    return (cc + np.sqrt(disc))/2.*cs2b/vw, 2


def dfdv(xiw: tp.Union[tp.Tuple[float, float], np.ndarray], v: float, cs2: float) -> tp.Tuple[float, float]:
    """The differential equation that is solved in the shock/rarefaction wave

    Rarefaction = the opposite of compression
    """
    xi, w = xiw
    dxidv = mu(xi, v)**2 / cs2 - 1.
    dxidv *= (1. - v*xi)*xi/2./v/(1.-v**2)
    dwdv = (1.+1./cs2) * mu(xi, v) * w/(1.-v**2)
    return dxidv, dwdv


def getKandWow(vw: float, v0: float, cs2: float) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Returns two values
    - Enthalpy-weighted kinetic energy in the shock/rarefaction wave
    - Ratio between the enthalpy density at the start of the shock/rarefaction compared to its end

    For the shocks the end is in the phase in front of the shock
    For the rarefaction wave, the enthalpy density is normalized to 1 behind the wall and has to be rescaled in the other part of the code.

    :param vw: wall velocity
    :param v0: starting point of the integration (?)
    :param cs2: speed of sound in the phase of interest
    """
    if v0 == 0:
        arr = np.array([])
        return arr, arr, arr, 0, 1
    n = 8*1024  # change accuracy here
    # n = 32*1024
    vs = np.linspace(v0, 0, n)
    # Get (xi, wow) for each v
    sol = odeint(dfdv, y0=[vw, 1.], t=vs, args=(cs2, ))
    xis, wows = (sol[:, 0], sol[:, 1])
    # If the wall moves at less than the sound speed = is subsonic
    if mu(vw, v0) * vw <= cs2:
        ll = max(int(
            # Counts the number of points until the shock
            sum(
                # Finds the position of the shock
                np.heaviside(cs2 - (mu(xis, vs)*xis), 0.0)
        )), 1)
        # Cut the vectors at the shock
        vs = vs[:ll]
        xis = xis[:ll]
        # Scaling with w_shock and then multiplying with the value of wow just before the shock
        # (This is also done in PTtools in
        wows = wows[:ll] / wows[ll-1] * getwow(xis[-1], mu(xis[-1], vs[-1]))
    Kint = simpson(y=wows*(xis*vs)**2/(1.-vs**2), x=xis)
    # Alternative trapezoidal integration
    # Kint = trapezoid(wows * (xis * vs) ** 2 / (1. - vs ** 2), xis)
    return vs, wows, xis, Kint*4./vw**3, wows[0]


def alN(al, wow, cs2b, cs2s):
    r"""$\alpha_{\bar{\theta}n}$ in the nucleation phase (in front of the shock)"""
    da = (1./cs2b - 1./cs2s)/(1./cs2s + 1.)/3.
    return (al + da)*wow - da


def getalNwow(vp, vm, vw, cs2b, cs2s):
    r"""Get
    - $\alpha_{\bar{\theta}}n}$ in the nucleation phase
    - Ratio of the enthalpies for fixed boundary conditions at the wall
    """
    _, _, _, Ksh, wow = getKandWow(vw, mu(vw, vp), cs2s)
    al = (vp/vm-1.)*(vp*vm/cs2b - 1.)/(1-vp**2)/3.
    return alN(al, wow, cs2b, cs2s), wow


def kappaNuMuModel(
        cs2b: float, cs2s: float,
        al: float, vw: float) -> tp.Tuple[float, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    r"""Calculate the efficiency factor $\kappa$.
    This uses the other functions.

    :param cs2b: $c_{s,b}^2$
    :param cs2s: $c_{s,s}^2$
    :param al: $\alpha_{\bar{\theta}n}$
    :param vw: $v_\text{wall}$
    """
    vm, mode = getvm(al, vw, cs2b)
    # If solving for a sub- or supersonic deflagration
    if mode in (0, 1):
        # Validate alpha
        almax, wow = getalNwow(0, vm, vw, cs2b, cs2s)
        if almax < al:
            logger.error(
                "alpha too large for shock, alpha=%s, max=%s",
                al, almax
            )
            raise ValueError
        vp = min(cs2s/vw, vw)
        almin, wow = getalNwow(vp, vm, vw, cs2b, cs2s)
        if almin > al:
            logger.error(
                "alpha too small for shock, alpha=%s, min=%s",
                al, almin
            )
            raise ValueError
        # Iterate to find v+ using binary search until the corresponding alpha matches the given value.
        # Set binary search limits.
        iv = [[vp, almin], [0, almax]]
        while abs(iv[1][0] - iv[0][0]) > 1e-7:
            # Get mean v of the limits
            vpm = (iv[1][0] + iv[0][0])/2.
            alm = getalNwow(vpm, vm, vw, cs2b, cs2s)[0]
            if alm > al:
                # Result is above the target. Search in the lower section.
                iv = [iv[0], [vpm, alm]]
            else:
                # Result is below the target. Search in the upper section.
                iv = [[vpm, alm], iv[1]]
        # Use the mean as v+
        vp = (iv[1][0] + iv[0][0])/2.
        v_out, wow_out, xi_out, Ksh, wow = getKandWow(vw, mu(vw, vp), cs2s)
    elif mode == 2:
        # For detonations no shooting is needed.
        Ksh, wow, vp = 0, 1, vw
        v_out = wow_out = xi_out = np.array([])
    else:
        raise ValueError("Invalid mode")

    # If the model is a detonation or a hybrid
    if mode > 0:
        v_in, wow_in, xi_in, Krf, wow3 = getKandWow(vw=vw, v0=mu(vw, vm), cs2=cs2b)
        v_in = np.flip(v_in)
        wow_mult = wow * getwow(vp, vm)
        wow_in = np.flip(wow_in) * wow_mult
        xi_in = np.flip(xi_in)
        Krf *= -wow_mult
    else:
        Krf = 0
        v_in = wow_in = xi_in = np.array([])

    # Expand v array to inside and outside the fluid shell
    v_f = v_b = np.array([0, 0])
    v_arr = np.concatenate((v_b, v_in, v_out, v_f))

    wow_arr = np.concatenate((wow_in, wow_out))
    wow_b = np.array([wow_arr[0], wow_arr[0]])
    wow_f = np.array([1, 1])
    wow_arr = np.concatenate((wow_b, wow_arr, wow_f))

    xi_arr = np.concatenate((xi_in, xi_out))
    xi_b = np.array([0, xi_arr[0]])
    xi_f = np.array([xi_arr[-1], 1])
    xi_arr = np.concatenate((xi_b, xi_arr, xi_f))

    if np.any(xi_arr < 0) or np.any(xi_arr > 1):
        logger.error("Invalid xi values")
    if np.any(v_arr < 0) or np.any(v_arr > 1):
        logger.error("Invalid v values")

    return (Ksh + Krf)/al, v_arr, wow_arr, xi_arr, mode, vp, vm
