"""Constants for the Sound Shell Model module"""

import typing as tp

import numpy as np

from pttools import bubble
import pttools.type_hints as th

#: Default number of xi points used in bubble profiles
DEFAULT_N_XI: int = 2000
#: Default number of T-tilde values for bubble lifetime distribution integration
DEFAULT_N_T: int = 10000
#: Default number of wavevectors used in the velocity convolution integrations.
# This should be at least as large as the default number of GW frequencies.
DEFAULT_N_Z_LOOKUP: int = 10000
NptType = th.IntArr1D | tuple[int, int, int]
DEFAULT_N_PT: NptType = (DEFAULT_N_XI, DEFAULT_N_T, DEFAULT_N_Z_LOOKUP)
DEFAULT_Y: th.FloatArr1D = np.logspace(-1, 3, 1000)

# It seems that NPTDEFAULT should be something like NXIDEFAULT/(2.pi), otherwise one
# gets a GW power spectrum which drifts up at high k.
#
# The maximum trustworthy k is approx NXIDEFAULT/(2.pi)
#
# NTDEFAULT can be left as it is, or even reduced to 100

#: Default dimensionless wavenumber above which to use approximation for sin_transform, sin_transform_approx.
# TODO: check that this can actually be a float
Z_ST_THRESH: float = 50

#: Default wavenumber overlap for matching sin_transform_approx
DZ_ST_BLEND: float = np.pi

#: Maximum in bubble lifetime distribution integration
T_TILDE_MAX: float = 20.0
#: Minimum in bubble lifetime distribution integration
T_TILDE_MIN: float = 0.01

#: Default nucleation parameters
DEFAULT_NUC_PARM: tuple[int] = (1,)

#: Default sound speed
CS0: tp.Final[float] = bubble.CS0

#: Default mean adiabatic index
GAMMA: float = 4/3

#: Gravitational constant in GeV
G: float = 1.22e19**(-2)
