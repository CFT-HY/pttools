import numpy as np

from pttools import bubble


NXIDEFAULT = 2000 # Default number of xi points used in bubble profiles
NTDEFAULT  = 200   # Default number of T-tilde values for bubble lifetime distribution integration
NQDEFAULT  = 320  # Default number of wavevectors used in the velocity convolution integrations.
NPTDEFAULT = [NXIDEFAULT, NTDEFAULT, NQDEFAULT]

#It seems that NPTDEFAULT should be something like NXIDEFAULT/(2.pi), otherwise one
#gets a GW power spectrum which drifts up at high k.
#
#The maximum trustworthy k is approx NXIDEFAULT/(2.pi)
#
#NTDEFAULT can be left as it is, or even reduced to 100

Z_ST_THRESH = 50    # Default dimensionless wavenumber above which to use approximation for
                    # sin_transform, sin_transform_approx.
DZ_ST_BLEND = np.pi # Default wavenumber overlap for matching sin_transform_approx

T_TILDE_MAX = 20.0 # Maximum in bubble lifetime distribution integration
T_TILDE_MIN = 0.01 # Minimum in bubble lifetime distribution integration

DEFAULT_NUC_TYPE = 'exponential'
DEFAULT_NUC_PARM = (1,)

cs0 = bubble.cs0 # Default sound speed
