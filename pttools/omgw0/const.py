"""Constants for the omgw0 module"""

import math

#: Speed of light (m/s)
c: float = 299792458.

#: Elementary charge $e$ in C
e: float = 1.602176634e-19

#: 1 eV in J
EV_IN_J: float = e

#: 1 GeV in J
GEV_IN_J: float = 1e9 * EV_IN_J

DEFAULT_G_STAR: float = 100.
DEFAULT_T_STAR: float = 100.  # GeV


F_GW0: float = 3.57e-5
r"""$F_{\text{gw},0}$
$$F_{\text{gw},0} = (3.57 \pm 0.05) \cdot 10^{-5} \left( \frac{100}{g_*} \right)^\frac{1}{3}$$
:caprini_2020:`\ `, eq. 20
Note that this constant assumes that $g_* = 100$, and the corresponding factor will have to be implemented elsewhere.
"""

F_STAR0_REF: float = 2.6e-6
r"""
$f_{\ast,0,\text{ref}}$, the factor used for converting from frequencies at the time of the GW formation to frequencies today.
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

#: :caprini_2020:`\ ` p. 12
G0: float = 2.
#: :caprini_2020:`\ ` p. 12
GS0: float = 3.91

#: Reduced Planck constant in SI units $\text{J} \cdot \text{s}$
H_BAR: float = 1.054571817e-34

#: Parsec to meters
PC_TO_M: float = 3.0857e16

#: Hubble constant, :planck_2018:`\ `
H0_KM_S_MPC: float = 67.4
#: Hubble constant, Planck value in Hz (about 2.27e-18 Hz)
H0_HZ: float = H0_KM_S_MPC * 1e3 / (PC_TO_M * 1e6)

#: LISA arm length (m)
LISA_ARM_LENGTH: float = 2.5e9

DAY_IN_SECONDS: float = 24 * 60 * 60
YEAR_IN_SECONDS: float = 365.2425 * DAY_IN_SECONDS
#: LISA observation time (s)
LISA_OBS_TIME: float = 4 * 0.75 * YEAR_IN_SECONDS

PLANCK_LENGTH: float = math.sqrt(H_BAR * G / (c**3))
r"""Planck length $l_\text{P}$
$$l_P = \sqrt{\frac{\hbar G}{c^3}}$$
"""

OMEGA_RADIATION: float = F_GW0 * GS0 ** (4 / 3) / G0
r"""
$\Omega_{\gamma,0}$, the radiation density parameter today.
Calculated from :caprini_2020:`\ ` p. 11-12.
Note that this value has been calculated assuming $h_\text{PLANCK} = 0.678$.
"""
