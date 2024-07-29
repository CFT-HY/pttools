"""Constants for the SSMtools module."""

import typing as tp

import numpy as np

from pttools import bubble

#: Default number of xi points used in bubble profiles
NXIDEFAULT: int = 2000
#: Default number of T-tilde values for bubble lifetime distribution integration
NTDEFAULT: int = 200
#: Default number of wavevectors used in the velocity convolution integrations.
NQDEFAULT: int = 320
NptType = tp.Union[np.ndarray, tp.Tuple[int, int, int]]
NPTDEFAULT: NptType = (NXIDEFAULT, NTDEFAULT, NQDEFAULT)

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
DEFAULT_NUC_PARM: tp.Tuple[int] = (1,)

#: Default sound speed
CS0: tp.Final[np.float64] = bubble.CS0
