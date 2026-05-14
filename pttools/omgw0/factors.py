r"""Factors used in calculating $\Omega_{\text{gw},0}$"""

from pttools.omgw0 import const
import pttools.type_hints as th


def F_gw0(
        g_star: th.FloatOrArr,
        g0: th.FloatOrArr = const.G0,
        gs0: th.FloatOrArr = const.GS0,
        gs_star: th.FloatOrArr | None = None,
        om_gamma0: th.FloatOrArr = const.OMEGA_RADIATION) -> th.FloatOrArr:
    r"""Power attenuation following the end of the radiation era
    $$F_{\text{gw},0} = \Omega_{\gamma,0} \left( \frac{g_{s0}}{g_{s\ast}} \right)^{4/9} \frac{{g}_\ast}{g_0}
    = (3.57 \pm 0.05) \cdot 10^{-5} \left( \frac{100}{{g}_\ast} \right)^{1/3}$$
    :hindmarsh_2017:`\ ` eq. 44
    :gowling_2021:`\ ` eq. 2.11

    There is a typo in :gowling_2021:`\ ` eq. 2.11: the $\frac{4}{9}$ should be $\frac{4}{3}$.

    Note that $\Omega_{\gamma,0}$ depends on the value of $h$.
    Therefore, when multiplying a value with $F_{\text{gw},0}$,
    you will have to multiply the result with $h^2$ to get a quantity that is independent of $h$.
    The default value has been calculated using $h_\text{PLANCK} = 0.678$.

    :param g_star: Degrees of freedom ${g}_\ast$ for pressure at the time the GWs were produced
    :param g0: Degrees of freedom $g_0$ for pressure today
    :param gs0: Degrees of freedom $g_{s,0}$ for entropy today
    :param gs_star: Degrees of freedom $g_{s,\ast}$ for entropy at the time the GWs were produced
    :param om_gamma0: $\Omega_{\gamma,0}$, the radiation density parameter today
    :return: Power attenuation factor $F_{\text{gw},0}$
    """
    if g0 is None or gs0 is None or gs_star is None or om_gamma0 is None:
        return const.F_GW0 * (100 / g_star)**(1/3)
    return om_gamma0 * (gs0 / gs_star)**(4/3) * g_star / g0
