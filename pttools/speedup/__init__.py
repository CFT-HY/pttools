"""Utilities for speeding up the simulations with Numba."""

# The native libraries have to be loaded before Numba is imported, as Numba checks for SVML on import.
from .libraries import *

# isort: split
# These have to be first so that the overloads and the fixes are applied to all other parts of this module.
# The name "overload" may be overwritten later.
from . import numba_fixes, overload

# isort: split
# Register the Numba overloads of scipy.integrate.
from . import integrate
from .differential import *
from .functions import *
from .jit import *
from .numba_wrapper import *
from .options import *
from .parallel import *
from .spline import *
from .threading import *
from .threads import *
from .utils import *
