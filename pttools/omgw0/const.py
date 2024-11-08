"""
Created on Thu Jul 28 11:35:38 2022

Constants for the omgw0 submodule of pttools

@author: hindmars
"""

T_default = 100  # GeV
Fgw0 = 3.57e-5  # arXiv:1910.13125v1 eqn 20
fs0_ref = 2.6e-6  # Eqn 2.13 of arXiv:2106.05984
# Todo: why is this different from the one in the suppression file?
SUP_METHOD_DEFAULT = "none"

#: :caprini_2020:`\ ` p. 12
G0 = 2
#: :caprini_2020:`\ ` p. 12
GS0 = 3.91

#: $\Omega_{\gamma,0}$, the radiation density parameter today. Calculated from :caprini_2020:`\ ` p. 11-12
OMEGA_RADIATION = Fgw0 * GS0**(4/3) / G0
