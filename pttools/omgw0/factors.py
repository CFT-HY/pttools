r"""Factors used in calculating $\Omega_{\text{gw},0}$"""

from pttools.omgw0 import const
import pttools.type_hints as th


def F_gw0(
        g_star: th.FloatOrArr,
        g0: th.FloatOrArr = const.G0,
        gs0: th.FloatOrArr = const.GS0,
        gs_star: th.FloatOrArr | None = None,
        om_gamma0: th.FloatOrArr = const.OMEGA_PHOTON) -> th.FloatOrArr:
    r"""Power attenuation following the end of the radiation era
    $$F_{\text{gw},0}
    = \left( \frac{{a}_\ast}{a_0} \right)^4 \left( \frac{{H}_\ast}{H_0} \right)^2
    = \Omega_{\gamma,0} \left( \frac{g_{s0}}{g_{s\ast}} \right)^\frac{4}{3} \frac{{g}_\ast}{g_0}$$
    :hindmarsh_2017:`\ ` eq. 44
    :gowling_2021:`\ ` eq. 2.11

    The first form is the redshifting of a radiation-like energy density.
    The second form follows from the conservation of entropy,
    $\frac{{a}_\ast}{a_0} = \frac{T_0}{{T}_\ast} \left( \frac{g_{s0}}{g_{s\ast}} \right)^\frac{1}{3}$,
    and from radiation domination,
    $\left( \frac{{H}_\ast}{H_0} \right)^2 = \frac{\rho_\ast}{\rho_{c,0}}$
    with $\rho_\ast = \frac{\pi^2}{30} {g}_\ast {T}_\ast^4$, which makes ${T}_\ast$ cancel out.

    When $g_{s\ast} = {g}_\ast$, this reduces to
    $$F_{\text{gw},0} = (3.57 \pm 0.05) \cdot 10^{-5} \left( \frac{100}{{g}_\ast} \right)^\frac{1}{3}$$
    :caprini_2020:`\ ` eq. 20

    There is a typo in :gowling_2021:`\ ` eq. 2.11: the $\frac{4}{9}$ should be $\frac{4}{3}$.

    Note that $\Omega_{\gamma,0}$ depends on the value of $h$.
    Therefore, when multiplying a value with $F_{\text{gw},0}$,
    you will have to multiply the result with $h^2$ to get a quantity that is independent of $h$.

    :param g_star: Degrees of freedom ${g}_\ast$ for pressure at the time the GWs were produced
    :param g0: Degrees of freedom $g_0$ for pressure today
    :param gs0: Degrees of freedom $g_{s,0}$ for entropy today
    :param gs_star: Degrees of freedom $g_{s,\ast}$ for entropy at the time the GWs were produced.
        If not given, the species are assumed to be in equilibrium, so that $g_{s\ast} = {g}_\ast$.
    :param om_gamma0: $\Omega_{\gamma,0}$, the photon density parameter today
    :return: Power attenuation factor $F_{\text{gw},0}$
    """
    if gs_star is None:
        gs_star = g_star
    return om_gamma0 * (gs0 / gs_star)**(4/3) * g_star / g0
