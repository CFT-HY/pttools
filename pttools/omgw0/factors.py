r"""Factors used in calculating $\Omega_{\text{gw},0}$"""

import numpy as np

from pttools.omgw0 import const
import pttools.type_hints as th


def F_gw0(
        g_star: th.FloatOrArr,
        g0: th.FloatOrArr = const.G0,
        gs0: th.FloatOrArr = const.GS0,
        gs_star: th.FloatOrArr | None = None,
        om_gamma0: th.FloatOrArr = const.OMEGA_RADIATION) -> th.FloatOrArr:
    r"""Power attenuation following the end of the radiation era
    $$F_{\text{gw},0} = \Omega_{\gamma,0} \left( \frac{g_{s0}}{g_{s*}} \right)^{4/9} \frac{g_*}{g_0}
    = (3.57 \pm 0.05) \cdot 10^{-5} \left( \frac{100}{g_*} \right)^{1/3}$$
    There is a typo in :gowling_2021:`\ ` eq. 2.11: the $\frac{4}{9}$ should be $\frac{4}{3}$.
    """
    if g0 is None or gs0 is None or gs_star is None or om_gamma0 is None:
        return 3.57e-5 * (100/g_star)**(1/3)
    return om_gamma0 * (gs0 / gs_star)**(4/3) * g_star / g0


def J(r_star: th.FloatOrArr, K_frac: th.FloatOrArr, nu: float = 0) -> th.FloatOrArr:
    r"""
    Pre-factor to convert power_gw_scaled to predicted spectrum
    approximation of $(H_n R_*)(H_n \tau_v)$
    updating to properly convert from flow time to source time

    $$J = H_n R_* H_n \tau_v = r_* \left(1 - \frac{1}{\sqrt{1 + 2x}} \right)$$
    :gowling_2021:`\ ` eq. 2.8
    """
    sqrt_K = np.sqrt(K_frac)
    return r_star * (1 - (np.sqrt(1 + 2*r_star/sqrt_K)**(-1-2*nu)))
