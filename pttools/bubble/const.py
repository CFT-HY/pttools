import numpy as np

# TODO: Use typing.Final, when the oldest supported Python version is 3.8
# https://www.python.org/dev/peps/pep-0591/

# smallest float
eps: np.float_ = np.nextafter(0, 1)

# Default and maximum number of entries in xi array
N_XI_DEFAULT: int = 5000
N_XI_MAX: int = 1000000
# How accurate is alpha_plus(alpha_n)
find_alpha_plus_tol: float = 1e-6
# Integration limit for parametric form of fluid equations
T_END_DEFAULT: float = 50.
dxi_small: float = 1./N_XI_DEFAULT

# Some functions useful for the bag equation of state.

cs0: np.float_ = 1 / np.sqrt(3)  # ideal speed of sound
cs0_2: float = 1./3  # ideal speed of sound squared
symm_phase: float = 0.0
brok_phase: float = 1.0
