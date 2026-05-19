"""Constants for the Sound Shell Model module"""

import typing as tp

import numpy as np

from pttools import bubble
import pttools.type_hints as th

NptType = th.IntArr1D | tuple[int, int, int]

# -----
# Default values
# -----
DEFAULT_A_STAR_A_R_RATIO: float = 1.
r"""
Default $\frac{a_*}{a_r}$,
the ratio of the scale factor $a_*$ at GW formation and $a_r$ at return to radiation dominance
:giombi_2024_cs:`\ ` eq. 2.18
"""

DEFAULT_N_SH: float = 1.
r"""
Default number of shock formation times $N_\text{sh}$
Note that $N_\text{sh}$ is not necessarily an integer.
:giombi_2024_cs:`\ ` eq. 4.1
"""

#: Default number of T-tilde values for bubble lifetime distribution integration
DEFAULT_N_T: int = 10000
#: Default number of $\xi$ points used in SSM computations
DEFAULT_N_XI_SSM: int = 2000
#: Default number of wavevectors used in the velocity convolution integrations.
# This should be at least as large as the default number of GW frequencies.
DEFAULT_N_Z_LOOKUP: int = 10000
DEFAULT_N_PT: NptType = (DEFAULT_N_XI_SSM, DEFAULT_N_T, DEFAULT_N_Z_LOOKUP)

#: Default nucleation parameters
DEFAULT_NUC_PARM: tuple[int] = (1,)

#: Default $r_*$
DEFAULT_R_STAR: float = 1.

#: Default range for wavenumbers $y$
DEFAULT_Y: th.FloatArr1D = np.logspace(-1, 3, 1000)

# It seems that NPTDEFAULT should be something like NXIDEFAULT/(2.pi), otherwise one
# gets a GW power spectrum which drifts up at high k.
#
# The maximum trustworthy k is approx NXIDEFAULT/(2.pi)
#
# NTDEFAULT can be left as it is, or even reduced to 100


# -----
# Limits
# -----

BETA_TILDE_CONVERSION_MIN: float = 10.
r"""
Lower limit for $\tilde{\beta} \equiv \beta/H_*$ conversion
The conversion $r_* \rightarrow \tilde{\beta}$ should give a warning when under this limit.
This is because the $\tilde{\beta} \leftrightarrow r_*$ conversion breaks down
in the case of a very slow phase transition, which corresponds to $\beta/H_* \approx 1$.
:caprini_2020:`\ ` p. 6
"""

BETA_TILDE_MIN: float = 3.8
r"""
Lower limit for $\tilde{\beta} \equiv \beta/H_*$
This is a conservative estimate.
Below this value, the abundance of primordial black holes would be too high.
:giombi_2026:`\ ` p. 3
:lewicki_2023:`\ `
"""

#: Maximum in bubble lifetime distribution integration
T_TILDE_MAX: float = 20.0
#: Minimum in bubble lifetime distribution integration
T_TILDE_MIN: float = 0.01

#: Default dimensionless wavenumber above which to use approximation for sin_transform, sin_transform_approx.
Z_ST_THRESH: float = 50.


# -----
# Constants
# -----

#: Default sound speed
CS0: tp.Final[float] = bubble.CS0
#: Default sound speed squared
CS0_2: tp.Final[float] = bubble.CS0_2

#: Default wavenumber overlap for matching sin_transform_approx
DZ_ST_BLEND: float = np.pi

#: Default mean adiabatic index $\Gamma$
GAMMA: float = 4/3
