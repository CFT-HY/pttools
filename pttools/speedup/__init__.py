"""Utilities for speeding up the simulations with Numba."""

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
from .tbb import *
from .threading import *
from .threads import *
from .utils import *

# isort: split
import importlib

# The wildcard imports above also export the modules that the submodules import,
# e.g. the "threading" of the standard library imported by utils.py,
# which replaces the submodule pttools.speedup.threading as an attribute of this package.
# "from . import threading" would return that attribute, so the submodule is taken from sys.modules instead.
threading = importlib.import_module(".threading", __name__)
