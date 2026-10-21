"""A fix for loading Intel Thread Building Blocks (TBB) for Numba.

Numba loads the TBB library by its name, but the ``tbb`` package from PyPI installs it
in the ``lib`` directory of the Python environment (``Library/bin`` on Windows),
and the ``tbb4u`` package for ARM in its own package directory.
These are not on the default search path of the dynamic loader,
so this module loads the library by its absolute path before Numba initialises its threading layer,
as explained in :mod:`pttools.speedup.libraries.library`.

Based on numba.np.ufunc.parallel._check_tbb_version_compatible()
"""

from ctypes import CDLL, c_int
import os
from pathlib import Path

from pttools.speedup.libraries.library import NativeLibrary

__all__ = [
    "TBB", "TBB_LIBRARY_NAME", "TBB_MIN_VERSION", "TBB_VERSION", "TBBLibrary",
    "get_tbb_version", "load_tbb", "tbb_library_dirs"
]

# As required by Numba in numba.np.ufunc.parallel._check_tbb_version_compatible()
TBB_MIN_VERSION = 12060


class TBBLibrary(NativeLibrary):
    """The TBB library, whose version is checked as Numba does before using the TBB threading layer."""

    def get_version(self, library: CDLL) -> int:
        """Get the TBB runtime interface version of the loaded library."""
        version_func = library.TBB_runtime_interface_version
        version_func.argtypes = []
        version_func.restype = c_int
        return version_func()


#: The TBB library
TBB: TBBLibrary = TBBLibrary(
    name="TBB",
    file_names={"linux": "libtbb.so.12", "darwin": "libtbb.12.dylib", "win32": "tbb12.dll"},
    packages=("tbb", "tbb4u"),
    min_version=TBB_MIN_VERSION,
    not_found_note="Numba will use another threading layer, which may not support nested parallelism.",
)
#: File name of the TBB library on the current operating system, or None if the operating system is unknown
TBB_LIBRARY_NAME: str | None = TBB.file_name


def get_tbb_version(path: str | os.PathLike[str] | None = None) -> int:
    """Get TBB library version.

    :param path: directory of the TBB library. If not given, the default search path of the operating system is used.
    :return: TBB runtime interface version
    :raises OSError: if the library cannot be loaded
    """
    return TBB.get_version(TBB.load_from(path))


def load_tbb() -> int | None:
    """Load a TBB library that is compatible with Numba.

    See :meth:`NativeLibrary.load <pttools.speedup.libraries.library.NativeLibrary.load>` for the search order.

    :return: TBB runtime interface version of the found library, or None if no TBB library was found.
        If the found library is too old for Numba, its version is returned, but it's not loaded.
    """
    TBB.load()
    return TBB.version


def tbb_library_dirs() -> list[Path]:
    """Directories of the Python environment that contain the TBB library."""
    return TBB.search_dirs()


#: TBB runtime interface version of the TBB library that was found when this module was imported,
#: or None if no TBB library was found.
#: The library is loaded on import, so that it's loaded before Numba initialises its threading layer.
TBB_VERSION: int | None = load_tbb()
