"""Loaders for the native libraries that Numba loads by their names, i.e. SVML and TBB.

The libraries are loaded when this package is imported, which :mod:`pttools.speedup` does before importing Numba.
See :mod:`pttools.speedup.libraries.library` for why this is needed.
Each library is optional, so this works even if only some of them are installed,
e.g. on macOS, where ``intel-cmplr-lib-rt`` is not available.

To check that the libraries can be loaded, run ``python -m pttools.speedup.libraries``.
This is in a separate ``__main__`` module, since this package is imported by :py:mod:`pttools.speedup`,
and running a module of it directly with ``python -m`` would execute it a second time.
"""

from .library import *
from .svml import *
from .tbb import *

# isort: split
from . import library, svml, tbb
from .library import NativeLibrary
from .svml import SVML
from .tbb import TBB

# The submodule names are not exported, so that they don't become attributes of pttools.speedup.
__all__ = ["LIBRARIES", "check_libraries"]
__all__ += library.__all__
__all__ += svml.__all__
__all__ += tbb.__all__

#: The native libraries loaded by this package, by their lowercase names
LIBRARIES: dict[str, NativeLibrary] = {"svml": SVML, "tbb": TBB}


def check_libraries(names: list[str] | None = None) -> int:
    """Print the state of the given libraries, and return the exit code for their check.

    :param names: lowercase names of the libraries in :data:`LIBRARIES`. If not given, all of them are checked.
    :return: 0 if all the required libraries have been loaded, and 1 otherwise
    :raises KeyError: if a library name is unknown
    """
    returncode = 0
    for name in names or list(LIBRARIES):
        returncode = max(returncode, LIBRARIES[name].check())
    return returncode
