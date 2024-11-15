"""
Created on Thu Jul 28 11:35:38 2022

Constants for the omgw0 submodule of pttools

@author: hindmars
"""

# Speed of light (m/s)
c: float = 299792458

T_default = 100  # GeV

#: arXiv:1910.13125v1 eqn 20
Fgw0 = 3.57e-5

#: # Eqn 2.13 of arXiv:2106.05984
fs0_ref = 2.6e-6

#: :caprini_2020:`\ ` p. 12
G0 = 2
#: :caprini_2020:`\ ` p. 12
GS0 = 3.91

#: Hubble constant, TODO
H0: float = None

#: LISA arm length (m)
LISA_ARM_LENGTH: float = 2.5e8

#: LISA observation time (s)
LISA_OBS_TIME: float = 126227808.

#: $\Omega_{\gamma,0}$, the radiation density parameter today. Calculated from :caprini_2020:`\ ` p. 11-12
OMEGA_RADIATION = Fgw0 * GS0**(4/3) / G0
