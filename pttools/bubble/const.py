"""Constants for the bubble module"""

import numpy as np

# TODO: Add typing.Final for these when Python 3.8 becomes the oldest supported version.
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
