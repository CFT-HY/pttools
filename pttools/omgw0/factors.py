r"""Factors used in calculating $\Omega_{\text{gw},0}$."""

import math
import typing as tp

import numpy as np

from pttools.omgw0.const import (
    DEFAULT_G_STAR,
    DEFAULT_T_STAR,
    GE0_PHOTON,
    GEV_IN_J,
    GS0,
    H_BAR_C,
    K_B,
    OMEGA_PHOTON_H2,
    T_CMB,
    G,
    c,
)
from pttools.type_hints import FloatOrArr


def a_ratio[T: FloatOrArr](
        T_star: T = DEFAULT_T_STAR,
        T0: T = T_CMB,
        gs_star: T = DEFAULT_G_STAR,
        gs0: T = GS0) -> T:
    r"""$\frac{a_{\ast}}{a_0}$.

    $$\frac{a_*}{a_0} = \frac{T_0}{T_*} \left( \frac{g_{s0}}{g_{s\ast}} \right)^\frac{1}{3}$$

    :param T_star: $T_*$, temperature at the time of GW formation in GeV
    :param T0: $T_0$, temperature today (for e.g. photons, corresponding to $g_{s0}$)
    :param gs_star: $g_{s\ast}$, degrees of freedom for entropy density at the time of GW formation
    :param gs0: $g_{s0}$, degrees of freedom for entropy density today (for e.g. photons, corresponding to $T_0$)
    """
    return tp.cast(T, K_B * T0 / (T_star * GEV_IN_J) * (gs0 / gs_star)**(1/3))


def e_SI[T2: FloatOrArr](T: T2, ge: T2) -> T2:
    r"""Energy density $e$ in SI units $\frac{\text{J}}{\text{m}^3}$.

    :param T: $T$, temperature in GeV
    :param ge: $g_e$, degrees of freedom for energy density
    :return: $e$, energy density in SI units $\frac{\text{J}{\text{m}^3}
    """
    # k_B T in J
    kT = T * GEV_IN_J
    return tp.cast(T2, np.pi**2 / 30 * ge * kT**4 / (H_BAR_C ** 3))


def F_gw0_h2[T: FloatOrArr](
        ge_star: T,
        gs_star: T | float | None = None,
        ge0_photon: T | float = GE0_PHOTON,
        gs0: T | float = GS0,
        om_gamma0_h2: T | float = OMEGA_PHOTON_H2) -> T:
    r"""$F_{\text{gw},0} h^2$, power attenuation of the GWs from the time of their production to today.

    $$F_{\text{gw},0} h^2
    = \left( \frac{{a}_\ast}{a_0} \right)^4 \left( \frac{{H}_\ast}{H_{100}} \right)^2
    = \Omega_{\gamma,0} h^2 \left( \frac{g_{s0}}{g_{s\ast}} \right)^\frac{4}{3} \frac{g_{e\ast}}{g_0}$$
    This is adapted from :gowling_2021:`\ ` eq. 2.11,
    $$F_{\text{gw},0}
    = \left( \frac{{a}_\ast}{a_0} \right)^4 \left( \frac{{H}_\ast}{H_0} \right)^2
    = \Omega_{\gamma,0} \left( \frac{g_{s0}}{g_{s\ast}} \right)^\frac{4}{3} \frac{g_{e\ast}}{g_0}.$$
    In the article the $\frac{4}{9}$ is a typo and should be $\frac{4}{3}$.

    The first form is the redshifting of a radiation-like energy density, $e_\text{gw} \propto a^{-4}$,
    relative to the critical density.
    The second form follows from the conservation of entropy,
    $\frac{{a}_\ast}{a_0} = \frac{T_0}{{T}_\ast} \left( \frac{g_{s0}}{g_{s\ast}} \right)^\frac{1}{3}$,
    and from the Friedmann equation, $\left( \frac{{H}_\ast}{H_0} \right)^2 = \frac{e_{\ast}}{e_{c,0}}$,
    where $e_\ast = \frac{\pi^2}{30} g_{e\ast} {T}_\ast^4$ is the energy density at the time of GW production.
    Writing $\Omega_{\gamma,0} = \frac{\pi^2}{30} g_0 T_0^4 / e_{c,0}$ makes $T_\ast$ and $T_0$ cancel out.
    Therefore the Hubble rate brings in $g_{e\ast}$ and the redshift $g_{s\ast}$,
    whereas the degrees of freedom for pressure do not appear.

    When $g_{s\ast} = g_{e\ast} \equiv g_{\ast}$, this reduces to
    $$F_{\text{gw},0} = (3.57 \pm 0.05) \cdot 10^{-5} \left( \frac{100}{g_{\ast}} \right)^\frac{1}{3}$$
    :caprini_2020:`\ ` eq. 20,
    :hindmarsh_2017:`\ ` eq. 44.

    Note that this function returns $\Omega_{\gamma,0} h^2$ instead of $\Omega_{\gamma,0}$,
    since it's independent of the value of $h$, unlike $\Omega_{\gamma,0}$,
    which depends on $h$.

    :param ge_star: $g_{e\ast}$, degrees of freedom for energy density at the time of GW production
    :param ge0_photon: $g_0$, degrees of freedom for the energy density of the photons today,
        corresponding to $\Omega_{\gamma,0}$
    :param gs0: $g_{s0}$, degrees of freedom for entropy today
    :param gs_star: $g_{s\ast}$, degrees of freedom for entropy at the time of GW production.
        If not given, the species are assumed to be in equilibrium, so that $g_{s\ast} = g_{e\ast}$.
    :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
    :return: $F_{\text{gw},0} h^2$
    """
    if gs_star is None:
        gs_star = ge_star
    return om_gamma0_h2 * (gs0 / gs_star)**(4/3) * ge_star / ge0_photon  # pyrefly: ignore[bad-return]


#: Factor $\sqrt{\frac{8\pi}{3}}$, used in computing Hubble rate $H$.
H_FACTOR: float = math.sqrt(8 * math.pi / 3)
#: Factor $\sqrt{\frac{8\pi G}{3c^2}}$, used for computing Hubble rate $H$ in SI units.
H_FACTOR_SI: float = H_FACTOR * math.sqrt(G) / c


def H[T: FloatOrArr](e: T) -> T:
    r"""Hubble rate without unit conversion.

    $$H = \sqrt{\frac{8 \pi e}{3}}$$

    :param e: $e$, energy density
    :return: Hubble rate
    """
    return tp.cast(T, H_FACTOR * np.sqrt(e))


def H_SI[T: FloatOrArr](e: T) -> T:
    r"""Hubble rate $H$ in SI units of $\frac{1}{\text{s}}$.

    $$H = \sqrt{\frac{8 \pi G e}{3 c^2}}$$

    :param e: $e$, energy density in SI units $\frac{\text{J}}{\text{m}^3}$
    :return: $H$, Hubble rate in SI units of $\frac{1}{\text{s}}$
    """
    return tp.cast(T, H_FACTOR_SI * np.sqrt(e))
