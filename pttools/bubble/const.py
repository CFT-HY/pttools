import numpy as np

# smallest float
eps = np.nextafter(0, 1)

# Default and maximum number of entries in xi array
N_XI_DEFAULT = 5000
N_XI_MAX = 1000000
# How accurate is alpha_plus(alpha_n)
find_alpha_plus_tol=1e-6
# Integration limit for parametric form of fluid equations
T_END_DEFAULT = 50.
dxi_small = 1./N_XI_DEFAULT

# Some functions useful for the bag equation of state.

cs0 = 1/np.sqrt(3)  # ideal speed of sound
cs0_2 = 1./3  # ideal speed of sound squared
symm_phase = 0.0
brok_phase = 1.0
