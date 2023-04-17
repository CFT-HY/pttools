"""Constants for the bubble module"""

import typing as tp

import numpy as np


#: Smallest float
EPS: tp.Final[np.float_] = np.nextafter(0, 1)

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
#: Array with one NaN
nan_arr = np.array([np.nan])

#: Ideal speed of sound
CS0: tp.Final[np.float_] = 1 / np.sqrt(3)
#: Ideal speed of sound squared
CS0_2: tp.Final[float] = 1/3

JUNCTION_ATOL: float = 2.4e-8
JUNCTION_CACHE_SIZE: int = 1024
