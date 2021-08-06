"""Constants for the bubble module"""

import numpy as np

# TODO: Use typing.Final, when the oldest supported Python version is 3.8
# https://www.python.org/dev/peps/pep-0591/

#: Smallest float
EPS: np.float_ = np.nextafter(0, 1)

#: Default number of entries in $\xi$ array
N_XI_DEFAULT: int = 5000
#: Maximum number of entries in $\xi$ array
N_XI_MAX: int = 1000000
#: How accurate is $\alpha_+ (\alpha_n)$
FIND_ALPHA_PLUS_TOL: float = 1e-6
#: Integration limit for the parametric form of the fluid equations
T_END_DEFAULT: float = 50.
#: Difference between consequent $\xi$ values
DXI_SMALL: float = 1. / N_XI_DEFAULT

# Some functions useful for the bag equation of state.

#: Ideal speed of sound
CS0: np.float_ = 1 / np.sqrt(3)
#: Ideal speed of sound squared
CS0_2: float = 1/3

# TODO: In general the phase is a scalar variable (real number).
# However, in the bag model it's approximated as an integer.
#: Symmetric phase
SYMM_PHASE: int = 0
#: Broken phase
BROK_PHASE: int = 1
