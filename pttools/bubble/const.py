"""Constants for the bubble module"""

import math
import typing as tp

# -----
# Default values
# -----
#: Default adiabatic ratio $\Gamma$
DEFAULT_ADIABATIC_RATIO: float = 4 / 3
#: Default number of entries in $\xi$ array
DEFAULT_N_XI: int = 5000
#: $\nu_\text{gdh2024}$ of :giombi_2024_cs:`\ ` eq. 2.11 for the bag model
DEFAULT_NU_GDH2024: float = 0.
#: Integration limit for the parametric form of the fluid equations
DEFAULT_T_END: float = 50.

# -----
# Limits
# -----
#: Maximum number of entries in $\xi$ array
N_XI_MAX: int = 1000000
#: Limit of points for a shell to be so thin that it should be re-computed with more points
THIN_SHELL_T_POINTS_MIN: int = 100

# -----
# Tolerances
# -----
#: How accurate is $\alpha_+ (\alpha_n)$
FIND_ALPHA_PLUS_TOL: float = 1e-6
# JUNCTION_ATOL: float = 2.4e-8
#: Relative tolerance of the junction solver
JUNCTION_RTOL: float = 1e-6

# -----
# Constants
# -----
ALPHA_PLUS_MAX_DEF: tp.Final[float] = 1 / 3
r"""
Fluid must flow into the bubble
($\tilde{v}_+ > 0$) in :py:func:`pttools.bubble.v_plus`.
For the deflagration branch, which has the negative sign,
this requires that $\alpha_+ < \frac{1}{3}$.
This applies for both subsonic deflagrations and hybrids.
:notes:`\ ` p. 36
"""

#: Ideal speed of sound
CS0: tp.Final[float] = 1 / math.sqrt(3)
#: Ideal speed of sound squared
CS0_2: tp.Final[float] = 1 / 3

#: Difference between consequent $\xi$ values
DXI_SMALL: float = 1. / DEFAULT_N_XI
#: Cache size for the junction solver
JUNCTION_CACHE_SIZE: int = 1024
