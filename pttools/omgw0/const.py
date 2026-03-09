"""Constants for the omgw0 module"""

# Speed of light (m/s)
c: float = 299792458.

DEFAULT_G_STAR: float = 100.
DEFAULT_T_STAR: float = 100.  # GeV

#: :caprini_2020:`\ `, eq. 20
FGW0: float = 3.57e-5

F_STAR0_REF: float = 2.6e-6
r"""
$f_{\ast,0,\text{ref}$, the factor used for converting from frequencies at the time of the GW formation to frequencies today.
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

#: :caprini_2020:`\ ` p. 12
G0: float = 2.
#: :caprini_2020:`\ ` p. 12
GS0: float = 3.91

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

OMEGA_RADIATION: float = FGW0 * GS0**(4/3) / G0
r"""
$\Omega_{\gamma,0}$, the radiation density parameter today.
Calculated from :caprini_2020:`\ ` p. 11-12.
Note that this value has been calculated assuming $h_\text{PLANCK} = 0.678$.
"""
