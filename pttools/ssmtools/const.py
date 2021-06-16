import typing as tp

import numpy as np

from pttools import bubble

# TODO: Use typing.Final, when the oldest supported Python version is 3.8
# https://www.python.org/dev/peps/pep-0591/

NXIDEFAULT: int = 2000  # Default number of xi points used in bubble profiles
NTDEFAULT: int = 200    # Default number of T-tilde values for bubble lifetime distribution integration
NQDEFAULT: int = 320    # Default number of wavevectors used in the velocity convolution integrations.
NPT_TYPE = tp.Union[np.ndarray, tp.Tuple[int, int, int]]
NPTDEFAULT: NPT_TYPE = (NXIDEFAULT, NTDEFAULT, NQDEFAULT)

# It seems that NPTDEFAULT should be something like NXIDEFAULT/(2.pi), otherwise one
# gets a GW power spectrum which drifts up at high k.
#
# The maximum trustworthy k is approx NXIDEFAULT/(2.pi)
#
# NTDEFAULT can be left as it is, or even reduced to 100

# Default dimensionless wavenumber above which to use approximation for sin_transform, sin_transform_approx.
# TODO: check that this can actually be a float
Z_ST_THRESH: float = 50

DZ_ST_BLEND: float = np.pi  # Default wavenumber overlap for matching sin_transform_approx

T_TILDE_MAX: float = 20.0   # Maximum in bubble lifetime distribution integration
T_TILDE_MIN: float = 0.01   # Minimum in bubble lifetime distribution integration

DEFAULT_NUC_PARM: tp.Tuple[int] = (1,)

CS0: np.float_ = bubble.CS0  # Default sound speed
