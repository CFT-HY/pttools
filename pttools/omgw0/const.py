"""Constants for the omgw0 module"""

import math

from pttools.type_hints import FloatOrArr

#: Speed of light (m/s)
#: :codata_2018:`\ ` table XXX
c: float = 299792458.
#: Elementary charge $e$ in C
#: :codata_2018:`\ ` table XXX
e: float = 1.602176634e-19

#: 1 eV in J
EV_IN_J: float = e
#: 1 GeV in J
GEV_IN_J: float = 1e9 * EV_IN_J

#: Default $g_*$
DEFAULT_G_STAR: float = 100.
#: Default $T_*$ in GeV
DEFAULT_T_STAR: float = 100.

# Todo: Is this 2.6e-6 or 2.7e-6?
F_STAR0_REF: float = 2.6e-6
r"""
$f_{\ast,0,\text{ref}}$,
the factor used for converting from frequencies at the time of the GW formation to frequencies today.

This value is valid as long as the universe is radiation dominated at the time of GW production.
This value is used in:
:caprini_2020:`\ ` eq. 31
:gowling_2021:`\ `, eq. 2.13
:gowling_2023:`\ `, eq. 2.9

It's derived in
:croon_2024:`\ `, eq. 38
"""

#: :lisa_sci_req:`\ ` eq. 3 (Hz)
F1_LISA: float = 4e-4

#: Gravitational constant $G$ in SI units $\frac{\text{m}^3}{\text{kg s}^2}$
G: float = 6.67430e-11

#: Gravitational constant $G$ in GeV
# G_GEV: float = 1.22e19**(-2)

#: $g_0$, the degrees of freedom for pressure today, aka. the two photon polarizations.
#: :caprini_2020:`\ ` p. 12
G0: float = 2.

N_NU: float = 3.044
r"""
$N_{\nu,\text{eff}}$, the effective number of neutrino species today.
:escudero_2026:`\ ` table 1 has several values in the range $N_{\nu,\text{eff}} \in [3.0435, 3.0453],
giving a reasonable estimate of $N_{\nu,\text{eff}} \approx 3.044$.
"""


def gs0(g0: FloatOrArr = G0, n_nu: FloatOrArr = N_NU) -> FloatOrArr:
    r"""
    $g_{s0}$, the degrees of freedom for entropy today

    $$g_{s0} = g_0 + \frac{7}{8} \cdot 2 N_\nu \cdot \frac{4}{11} \approx 3.91$$
    The factors in this formula come from the sources below.

    For ultrarelativistic particles,
    $$p = \frac{g}{6\pi^2} \int_0^\infty \frac{p^3 dp}{e^{\frac{p}{T}} \pm 1}$$.
    :maki_msc:`\ ` eq. 2.103
    For fermions,
    $$\int_0^\infty \frac{x^n}{e^x + 1} dx = (1 - 2^{-n}) \Gamma(n+1) \zeta(n+1)$$.
    :schroeder_thermal:`\ ` eq. B.36
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
    $$\frac{11}{2} (a T_\gamma)^3_\text{before} = 2 (a T_\gamma)^3_\text{after}$$.
    The neutrinos continue carrying entropy corresponding to the degrees of freedom before the annihilation,
    resulting in
    $$\left( \frac{T_\nu}{T_\gamma} \right)^3 = \frac{4}{11}$$,
    which gives $g_{s0,\nu}$ an effective multiplier of $\frac{4}{11}$.
    See :wikipedia:`Cosmic_neutrino_background`.

    Together, these factors result $g_{s0} \approx 3.91$ of :caprini_2020:`\ ` p. 12.
    """
    return g0 + 7 / 8 * 2 * n_nu * (4 / 11)


GS0: float = 3.9298
r"""
$g_{s0}$, degrees of freedom for the entropy density today
:escudero_2026:`\ ` table 1
This version is denoted in the article as $h_\text{eff}$,
and it's the one used for the temperature scaling from the conservation of comoving entropy.
"""

#: :notes:`\ ` p. 10
# G_EFF_SM: float = 106.75

#: Reduced Planck constant $\hbar$ in SI units $\text{J} \cdot \text{s}$
#: :codata_2018:`\ ` table XXX
H_BAR: float = 1.054571817e-34

#: Boltzmann constant $k_B$ in SI units $\frac{\text{J}}{\text{K}}$
#: :codata_2018:`\ ` table XXX
K_B: float = 1.380649e-23

#: $T_0$, the CMB temperature today (K)
# :fixsen_2009:`\ ` table 2
T_CMB: float = 2.72548

STEFAN_BOLTZMANN: float = math.pi**2 * K_B**4 / (60 * c**2 * H_BAR**3)
r"""
Stefan-Boltzmann constant $\sigma$ in SI units $\frac{W}{m^2 K^4}$
$$\sigma
= \frac{2 \pi^5 k_B^4}{15 c^2 h^3}
= \frac{\pi^2 k_B^4}{60 c^2 \hbar^3}
\approx 5.670374419 \cdot 10^{-8} \frac{W}{m^2 K^4}
$$
:codata_2018:`\ ` table XXX
:wikipedia:`Stefan–Boltzmann_law`
"""

A_RADIATION: float = 4 * STEFAN_BOLTZMANN / c
r"""
$a$, the radiation constant in SI units $\frac{\text{J}}{\text{m}^3 \text{K}^4}$
$$a = \frac{\pi^2 k_B^4}{15 \hbar^3 c^3} = \frac{4\sigma}{c}$$
The energy density of a gas of $g$ relativistic degrees of freedom is
$\rho c^2 = \frac{\pi^2}{30} g \frac{(k_B T)^4}{(\hbar c)^3}$,
which for the $g_0 = 2$ photon polarizations reduces to $\rho_{\gamma} c^2 = a T^4$.
"""

#: Parsec to meters
PC_TO_M: float = 3.0857e16

H: float = 0.6766
r"""
$h$, dimensionless reduced Hubble constant :planck_2018:`\ `
Please note that the observable quantity is $h^2 \Omega$,
and that the $h$ of :py:data:`OMEGA_PHOTON` cancels out when converting to it.
"""
#: $h^2$, dimensionless reduced Hubble constant squared
H2: float = H**2
#: Hubble constant $H_0$ in $\frac{\text{km}}{\text{s Mpc}}$
H0_KM_S_MPC: float = 100. * H
#: Hubble constant, Planck value in Hz (about 2.27e-18 Hz)
H0_HZ: float = H0_KM_S_MPC * 1e3 / (PC_TO_M * 1e6)

H0_100_HZ: float = 100. * 1e3 / (PC_TO_M * 1e6)
r"""
${H}_{100} = 100 \frac{\text{km}}{\text{s Mpc}}$ in Hz,
the reference value by which $H_0 = h {H}_{100}$ is defined.
"""

#: LISA arm length (m)
LISA_ARM_LENGTH: float = 2.5e9

#: Number of seconds in a day
DAY_IN_SECONDS: float = 24 * 60 * 60
# Number of seconds in a year
YEAR_IN_SECONDS: float = 365.2425 * DAY_IN_SECONDS
#: LISA observation time (s)
LISA_OBS_TIME: float = 4 * 0.75 * YEAR_IN_SECONDS

PLANCK_LENGTH: float = math.sqrt(H_BAR * G / (c**3))
r"""Planck length $l_\text{P}$
$$l_P = \sqrt{\frac{\hbar G}{c^3}}$$
"""

OMEGA_PHOTON_H2: float = 8 * math.pi * G * A_RADIATION * T_CMB ** 4 / (3 * H0_100_HZ ** 2 * c ** 2)
r"""
$\Omega_{\gamma,0} h^2$, the photon density parameter today, scaled by $h^2$
$$\Omega_{\gamma,0} h^2
= \frac{\rho_{\gamma,0}}{\rho_{c,0}} h^2
= \frac{8 \pi G a {T}_0^4}{3 {H}_{100}^2 c^2}
\approx 2.473 \cdot 10^{-5}$$
obtained from $\rho_{\gamma,0} c^2 = a {T}_0^4$ and $\rho_{c,0} = \frac{3 {H}_0^2}{8 \pi G}$
with ${H}_0 = h {H}_{100}$.
This depends only on ${T}_0$ and the fundamental constants,
and is therefore independent of the value of $h$.
"""

OMEGA_PHOTON: float = OMEGA_PHOTON_H2 / H2
r"""
$\Omega_{\gamma,0}$, the photon density parameter today.
:caprini_2020:`\ ` p. 11-12

Please note that this is the density of the photons only,
and does not include the neutrinos, which are instead accounted for by :py:data:`GS0`.
Please also note that this depends on the value of :py:data:`H`,
which is why quantities computed from it have to be multiplied by $h^2$
to get a quantity that is independent of $h$.
"""
