r"""Factors used in calculating $\Omega_{\text{gw},0}$."""

from pttools.omgw0.const import G0, GS0, OMEGA_PHOTON_H2
from pttools.type_hints import FloatOrArr


def F_gw0_h2[T: FloatOrArr](
        g_star: T,
        g0: T | float = G0,
        gs0: T | float = GS0,
        gs_star: T | float | None = None,
        om_gamma0_h2: T | float = OMEGA_PHOTON_H2) -> T:
    r"""$F_{\text{gw},0} h^2$, power attenuation following the end of the radiation era.

    $$F_{\text{gw},0} h^2
    = \left( \frac{{a}_\ast}{a_0} \right)^4 \left( \frac{{H}_\ast}{H_{100}} \right)^2
    = \Omega_{\gamma,0} h^2 \left( \frac{g_{s0}}{g_{s\ast}} \right)^\frac{4}{3} \frac{{g}_\ast}{g_0}$$
    This is adapted from
    $$F_{\text{gw},0}
    = \left( \frac{{a}_\ast}{a_0} \right)^4 \left( \frac{{H}_\ast}{H_0} \right)^2
    = \Omega_{\gamma,0} \left( \frac{g_{s0}}{g_{s\ast}} \right)^\frac{4}{3} \frac{{g}_\ast}{g_0}$$
    :hindmarsh_2017:`\ ` eq. 44
    :gowling_2021:`\ ` eq. 2.11.

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
    :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
    :return: Power attenuation factor $F_{\text{gw},0}$
    """
    if gs_star is None:
        gs_star = g_star
    return om_gamma0_h2 * (gs0 / gs_star)**(4/3) * g_star / g0  # pyrefly: ignore[bad-return]
