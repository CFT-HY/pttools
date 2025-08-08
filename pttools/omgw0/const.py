"""Constants for the omgw0 module"""

# Speed of light (m/s)
c: float = 299792458

G_STAR_DEFAULT = 100
T_default = 100  # GeV

#: arXiv:1910.13125v1 eqn 20
Fgw0 = 3.57e-5

#: # Eqn 2.13 of arXiv:2106.05984
fs0_ref = 2.6e-6

#: :lisa_sci_req:`\ ` eq. 3 (Hz)
F1_LISA: float = 4e-4

#: :caprini_2020:`\ ` p. 12
G0 = 2
#: :caprini_2020:`\ ` p. 12
GS0 = 3.91

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

#: $\Omega_{\gamma,0}$, the radiation density parameter today. Calculated from :caprini_2020:`\ ` p. 11-12
OMEGA_RADIATION: float = Fgw0 * GS0**(4/3) / G0
