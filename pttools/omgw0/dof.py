"""Degrees of freedom $g$."""

import typing as tp

from pttools.omgw0.const import GE0_PHOTON, GS0_PHOTON, N_NU
from pttools.type_hints import FloatOrArrU

__all__ = ["ge0", "gs0"]


def ge0[T: FloatOrArrU](ge0_photon: T = GE0_PHOTON, n_nu: T | float = N_NU) -> T:
    r"""$g_{e0}$, the degrees of freedom for energy density today.

    $$g_{e0} = 2 + \frac{7}{8} \cdot 2 N_{\nu} \cdot \left( \frac{4}{11} \right)^\frac{4}{3}$$
    """
    return tp.cast(T, ge0_photon + 7 / 8 * 2 * n_nu * (4 / 11)**(4/3))


def gs0[T: FloatOrArrU](gs0_photon: T = GS0_PHOTON, n_nu: T | float = N_NU) -> T:
    r"""$g_{s0}$, the degrees of freedom for entropy today.

    $$g_{s0} = g_0 + \frac{7}{8} \cdot 2 \cdot 3 \cdot \frac{4}{11} \left( \frac{N_{\nu}}{3} \right)^\frac{3}{4}$$
    This is $\frac{43}{11} \approx 3.91$ for $N_{\nu} = 3$ and $3.930$ for $N_{\nu} = 3.044$.
    The factors in this formula come from the sources below.

    For ultrarelativistic particles,
    $$p = \frac{g}{6\pi^2} \int_0^\infty \frac{p^3 dp}{e^{\frac{p}{T}} \pm 1}$$.
    :maki_msc:`\ ` eq. 2.103
    For fermions,
    $$\int_0^\infty \frac{x^n}{e^x + 1} dx = (1 - 2^{-n}) \Gamma(n+1) \zeta(n+1)$$.
    :schroeder_book:`\ ` eq. B.36
    This gives a factor of $1 - 2^{-3} = \frac{7}{8}$ compared to bosons.

    In the Standard Model, each neutrino species contributes one helicity state for the neutrino $\nu$
    and one for the antineutrino $\bar{\nu}$.
    This gives a factor of 2.

    When neutrinos decouple at a few MeV, photons are still interacting with electrons and positrons.
    For this interacting sector with 2 photon polarizations and 2 fermions with 2 spins,
    $$g_s = 2 + \frac{7}{8} \cdot 4 = \frac{11}{2}$$.
    When electrons and positrons annihilate, the photons are left with $g_s = 2$.

    For a perfect fluid in local equilibrium, the comoving entropy $sa^3$ is a conserved quantity,
    $$\frac{d}{dt} (sa^3) = 0 \Rightarrow g_s (aT)^3 = \text{const}$$.
    Therefore, the decoupling results in
    $$\frac{11}{2} (a T_{\gamma})^3_\text{before} = 2 (a T_{\gamma})^3_\text{after}$$.
    The neutrinos continue carrying entropy corresponding to the degrees of freedom before the annihilation,
    resulting in
    $$\left( \frac{T_{\nu}}{T_{\gamma}} \right)^3 = \frac{4}{11}$$,
    which gives $g_{s0,\nu}$ an effective multiplier of $\frac{4}{11}$.
    See :wikipedia:`Cosmic_neutrino_background`.

    Together, these factors result in $g_{s0} \approx 3.91$ of :caprini_2020:`\ ` p. 12
    for three neutrino species that decouple instantaneously.

    In reality the neutrinos are slightly heated by the annihilation, which is described by $N_{\nu} > 3$.
    $N_{\nu}$ is defined through the energy density of the neutrinos,
    $e_{\nu} \propto N_{\nu} \propto T_{\nu}^4$ (:escudero_2026:`\ ` eq. 5.2 and 5.3),
    whereas their entropy density is $s_{\nu} \propto T_{\nu}^3$.
    Therefore, the entropy of the neutrinos scales as $N_{\nu}^\frac{3}{4}$ instead of linearly.
    This neglects the spectral distortions of the neutrinos, and reproduces
    $h_\text{eff} \approx 3.930$ of :escudero_2026:`\ ` table 1 (:py:data:`pttools.omgw0.const.GS0`),
    which is defined through the conservation of $h_{\ast} (a T_{\gamma})^3$ in :escudero_2026:`\ ` eq. 5.4.
    """
    return tp.cast(T, gs0_photon + 7 / 8 * 2 * 3 * (4 / 11) * (n_nu / 3)**(3/4))
