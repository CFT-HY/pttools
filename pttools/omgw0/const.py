r"""Constants for the omgw0 module.

The constants that have an experimental uncertainty are also provided as
floats with an uncertainty of the :external+uncertainties:doc:`uncertainties <index>` package,
which are denoted by the suffix ``_U``, e.g. :py:data:`T_CMB_U`.
The plain constants without the suffix are their nominal values as floats,
so that they can be used in Numba-compiled functions and in NumPy arrays of floats.
The uncertainties are standard uncertainties, i.e. estimated standard deviations,
as expected by the uncertainties package.
The derived constants with the suffix ``_U`` propagate the uncertainties of the constants they are computed from,
including the correlations between them.
The constants that are exact in the SI units since 2019, such as $c$, $e$, $\hbar$ and $k_B$,
or exact by definition, such as the astronomical unit, have no uncertainty.
"""

import math

from uncertainties import ufloat, umath

from pttools.type_hints import UFloat

__all__ = [
    "AU_IN_M",
    "A_RADIATION",
    "DAY_IN_SECONDS",
    "DEFAULT_G_STAR",
    "DEFAULT_T_STAR",
    "EV_IN_J",
    "GE0_PHOTON",
    "GEV_IN_J",
    "GP0_PHOTON",
    "GS0",
    "GS0_PHOTON",
    "G_EFF_SM",
    "G_U",
    "H0_100_HZ",
    "H0_HZ",
    "H0_HZ_U",
    "H0_KM_S_MPC",
    "H0_KM_S_MPC_U",
    "H2",
    "H2_U",
    "H_BAR",
    "H_BAR_C",
    "H_U",
    "K_B",
    "LISA_ARM_LENGTH",
    "LISA_OBS_TIME",
    "N_NU",
    "OMEGA_PHOTON",
    "OMEGA_PHOTON_H2",
    "OMEGA_PHOTON_H2_U",
    "OMEGA_PHOTON_U",
    "PC_IN_M",
    "PLANCK_LENGTH",
    "PLANCK_LENGTH_U",
    "STEFAN_BOLTZMANN",
    "T_CMB",
    "T_CMB_U",
    "YEAR_IN_SECONDS",
    "G",
    "H",
    "c",
    "e",
]

#: Astronomical unit au in m
#: :wikipedia:`Astronomical_unit`
AU_IN_M: float = 149597870700.

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

G_U: UFloat = ufloat(6.67430e-11, 0.00015e-11)
r"""Gravitational constant $G$ in SI units $\frac{\text{m}^3}{\text{kg s}^2}$, with its uncertainty

$$G = 6.674 30(15) \cdot 10^{-11} \frac{\text{m}^3}{\text{kg s}^2}$$
:codata_2018:`\ ` eq. 201 (p. 42).
The uncertainty is the standard uncertainty, i.e. the estimated standard deviation.
"""
#: Gravitational constant $G$ in SI units $\frac{\text{m}^3}{\text{kg s}^2}$, nominal value of :py:data:`G_U`
G: float = G_U.nominal_value

#: $g_{e\gamma 0}$, the degrees of freedom for energy density of photons today.
#: This is used in :caprini_2020:`\ ` p. 12
GE0_PHOTON: float = 2.

GP0_PHOTON: float = GE0_PHOTON
r"""
$g_{p\gamma 0}$, the degrees of freedom for pressure of photons today.
Since photons are massless and therefore ultrarelativistic,
$p = \frac{e}{3}$ and consequently $g_{p\gamma 0} = g_{e\gamma 0}$.
"""

GS0_PHOTON: float = GE0_PHOTON
r"""
$g_{s\gamma 0}$, the degrees of freedom for entropy density of photons today.
Since photons are massless and therefore ultrarelativistic,
$p = \frac{e}{3}$ and consequently $g_{s\gamma 0} = g_{e\gamma 0}$.
"""

N_NU: float = 3.044
r"""
$N_{\nu,\text{eff}}$, the effective number of neutrino species today.
:escudero_2026:`\ ` table 1 has several values in the range $N_{\nu,\text{eff}} \in [3.0435, 3.0453]$,
giving a reasonable estimate of $N_{\nu,\text{eff}} \approx 3.044$.
The article does not give an uncertainty for the value,
and therefore this constant has no counterpart with an uncertainty.

The older value $N \approx 3.046$ is used in
:planck_2015:`\ `,
:cosmo1:`\ ` eq. 4.43,
:cosmo2:`\ ` table 4.
"""

GS0: float = 3.9298
r"""
$g_{s0}$, degrees of freedom for the entropy density today
:escudero_2026:`\ ` table 1
This version is denoted in the article as $h_\text{eff}$,
and it's the one used for the temperature scaling from the conservation of comoving entropy.
The article does not give an uncertainty for the value,
and therefore this constant has no counterpart with an uncertainty.
"""

G_EFF_SM: float = 106.75
r"""Degrees of freedom $g$ in the Standard Model at high temperatures.

$$g_{SM}(T \gtrsim 200 \text{GeV}) = 28 + \frac{7}{8} \cdot 90 = 106.75$$,
:physics_stack:`170682`,
:notes:`\ ` p. 10.
"""

#: Reduced Planck constant $\hbar$ in SI units $\text{J} \cdot \text{s}$
#: :codata_2018:`\ ` table XXX
H_BAR: float = 1.054571817e-34

#: $\hbar c$, reduced Planck constant $\hbar$ times the speed of light $c$ in SI units $\text{J} \cdot \text{m}$
H_BAR_C: float = H_BAR * c

#: Boltzmann constant $k_B$ in SI units $\frac{\text{J}}{\text{K}}$
#: :codata_2018:`\ ` table XXX
K_B: float = 1.380649e-23

T_CMB_U: UFloat = ufloat(2.72548, 0.00057)
r"""
$T_0$, the CMB temperature today (K), with its uncertainty

$$T_{CMB} = 2.72548 \pm 0.00057 \text{K}$$
:fixsen_2009:`\ ` table 2.
This is the weighted mean of the measurements, and the uncertainty is its standard deviation ($1\sigma$).

An older reference is provided by
$$T_{CMB} \approx 2725 \pm 1 \text{mK}$$
:fixsen_2002:`\ `.
"""
#: $T_0$, the CMB temperature today (K), nominal value of :py:data:`T_CMB_U`
T_CMB: float = T_CMB_U.nominal_value

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

#: Parsec (pc) in meters (m)
#: :wikipedia:`Parsec`
PC_IN_M: float = 180 * 60 * 60 * AU_IN_M / math.pi

H_U: UFloat = ufloat(0.6766, 0.0042)
r"""
$h$, dimensionless reduced Hubble constant, with its uncertainty.

$$H_0 = 67.66 \pm 0.42 \frac{\text{km}}{\text{s Mpc}}$$
TT,TE,EE+lowE+lensing+BAO (68 % limits),
:planck_2018:`\ ` table 2.
The value is the mean of the marginalized posterior,
and the uncertainty is the half-width of its 68 % interval,
which corresponds to the standard deviation ($1\sigma$) for a Gaussian posterior.
This is the value that :wikipedia:`Hubble's_law` quotes as the Planck 2018 value.

Some references use the value
$$H_0 \approx 67.27 \pm 0.60$$
TT,TE,EE+lowE (68 % limits),
:planck_2018:`\ ` table 2.

Please note that the observable quantity is $\Omega_\text{gw} h^2$,
and that the $h$ of :py:data:`OMEGA_PHOTON` cancels out when converting to it.
:wikipedia:`Hubble's_law`
"""
#: $h$, dimensionless reduced Hubble constant, nominal value of :py:data:`H_U`
H: float = H_U.nominal_value
#: $h^2$, dimensionless reduced Hubble constant squared, with its uncertainty
H2_U: UFloat = H_U**2
#: $h^2$, dimensionless reduced Hubble constant squared, nominal value of :py:data:`H2_U`
H2: float = H2_U.nominal_value
#: Hubble constant $H_0$ in $\frac{\text{km}}{\text{s Mpc}}$, with its uncertainty
H0_KM_S_MPC_U: UFloat = 100. * H_U
#: Hubble constant $H_0$ in $\frac{\text{km}}{\text{s Mpc}}$, nominal value of :py:data:`H0_KM_S_MPC_U`
H0_KM_S_MPC: float = H0_KM_S_MPC_U.nominal_value
#: Hubble constant $H_0$, Planck value in Hz (about 2.19e-18 Hz), with its uncertainty
H0_HZ_U: UFloat = H0_KM_S_MPC_U * 1e3 / (PC_IN_M * 1e6)
#: Hubble constant $H_0$, Planck value in Hz (about 2.19e-18 Hz), nominal value of :py:data:`H0_HZ_U`
H0_HZ: float = H0_HZ_U.nominal_value

H0_100_HZ: float = 100. * 1e3 / (PC_IN_M * 1e6)
r"""
${H}_{100} = 100 \frac{\text{km}}{\text{s Mpc}}$ in Hz,
the reference value by which $H_0 = h {H}_{100}$ is defined.
"""

#: LISA arm length (m)
LISA_ARM_LENGTH: float = 2.5e9

#: Number of seconds in a day
DAY_IN_SECONDS: float = 24 * 60 * 60
#: Number of seconds in a year
YEAR_IN_SECONDS: float = 365.2425 * DAY_IN_SECONDS
#: LISA observation time (s)
LISA_OBS_TIME: float = 4 * 0.75 * YEAR_IN_SECONDS

PLANCK_LENGTH_U: UFloat = umath.sqrt(H_BAR * G_U / (c**3))  # pyrefly: ignore[missing-attribute]
r"""Planck length $l_\text{P}$ in m, with its uncertainty
$$l_P = \sqrt{\frac{\hbar G}{c^3}}$$
"""
#: Planck length $l_\text{P}$ in m, nominal value of :py:data:`PLANCK_LENGTH_U`
PLANCK_LENGTH: float = PLANCK_LENGTH_U.nominal_value

OMEGA_PHOTON_H2_U: UFloat = 8 * math.pi * G_U * A_RADIATION * T_CMB_U ** 4 / (3 * H0_100_HZ ** 2 * c ** 2)
r"""
$\Omega_{\gamma,0} h^2$, the photon density parameter today, scaled by $h^2$, with its uncertainty
$$\Omega_{\gamma,0} h^2
= \frac{\rho_{\gamma,0}}{\rho_{c,0}} h^2
= \frac{8 \pi G a {T}_\text{CMB}^4}{3 {H}_{100}^2 c^2}
\approx 2.473 \cdot 10^{-5}$$
obtained from $\rho_{\gamma,0} c^2 = a {T}_0^4$ and $\rho_{c,0} = \frac{3 {H}_0^2}{8 \pi G}$
with ${H}_0 = h {H}_{100}$.
This depends only on ${T}_0$ and the fundamental constants,
and is therefore independent of the value of $h$.
The uncertainty comes from those of ${T}_0$ and $G$, and is dominated by that of ${T}_0$.
"""
#: $\Omega_{\gamma,0} h^2$, the photon density parameter today, scaled by $h^2$,
#: nominal value of :py:data:`OMEGA_PHOTON_H2_U`
OMEGA_PHOTON_H2: float = OMEGA_PHOTON_H2_U.nominal_value

OMEGA_PHOTON_U: UFloat = OMEGA_PHOTON_H2_U / H2_U
r"""
$\Omega_{\gamma,0}$, the photon density parameter today, with its uncertainty.
:caprini_2020:`\ ` p. 11-12

Please note that this is the density of the photons only,
and does not include the neutrinos, which are instead accounted for by :py:data:`GS0`.
Please also note that this depends on the value of :py:data:`H`,
which is why quantities computed from it have to be multiplied by $h^2$
to get a quantity that is independent of $h$.
"""
#: $\Omega_{\gamma,0}$, the photon density parameter today, nominal value of :py:data:`OMEGA_PHOTON_U`
OMEGA_PHOTON: float = OMEGA_PHOTON_U.nominal_value
