"""A fix for loading Intel Thread Building Blocks (TBB) for Numba.

Numba loads the TBB library by its name using the default library search path of the operating system.
However, the ``tbb`` package from PyPI installs the library in the ``lib`` directory of the Python environment
(or ``Library/bin`` on Windows), which is not on the default search path of the dynamic loader.
Therefore, this module loads the library by its absolute path before Numba initialises its threading layer.
The loader then finds the already-loaded library by its name when Numba imports its TBB extension.

https://github.com/numba/numba/issues/7531

Based on numba.np.ufunc.parallel._check_tbb_version_compatible()
"""
# https://github.com/numba/numba/issues/7531#issuecomment-1614510255

from ctypes import CDLL, c_int
import logging
import os
import sys

from pttools.utils.system import IS_LINUX, IS_OSX, IS_WINDOWS

logger: logging.Logger = logging.getLogger(__name__)

# From numba.np.ufunc.parallel

# As required by Numba in numba.np.ufunc.parallel._check_tbb_version_compatible()
TBB_MIN_VERSION = 12060

TBB_LIBRARY_NAME: str | None
if IS_WINDOWS:
    TBB_LIBRARY_NAME = "tbb12.dll"
elif IS_OSX:
    TBB_LIBRARY_NAME = "libtbb.12.dylib"
elif IS_LINUX:
    TBB_LIBRARY_NAME = "libtbb.so.12"
else:
    TBB_LIBRARY_NAME = None

# References to the loaded libraries are kept so that they stay loaded for the lifetime of the process.
_LOADED_LIBRARIES: list[CDLL] = []


def _tbb_library_name() -> str:
    if TBB_LIBRARY_NAME is None:
        raise ValueError("Unknown operating system")
    return TBB_LIBRARY_NAME


def tbb_library_dirs() -> list[str]:
    """Directories of the Python environment where the ``tbb`` package from PyPI installs its libraries.

    These are the ``lib`` directories (``Library/bin`` on Windows) of the current virtual environment
    and its base interpreter.
    """
    prefixes = [sys.prefix, sys.base_prefix]
    venv = os.getenv("VIRTUAL_ENV")
    if venv:
        prefixes.append(venv)

    dirs: list[str] = []
    for prefix in prefixes:
        if IS_WINDOWS:
            candidates = [os.path.join(prefix, "Library", "bin"), os.path.join(prefix, "bin")]
        else:
            candidates = [os.path.join(prefix, "lib")]
        for candidate in candidates:
            if candidate not in dirs and os.path.isdir(candidate):
                dirs.append(candidate)
    return dirs


def _load_library(path: str | None = None) -> CDLL:
    """Load the TBB library from the given directory, or from the default search path if no directory is given."""
    name = _tbb_library_name()
    if path is not None:
        # sys.platform is checked instead of IS_WINDOWS, as type checkers use it to know that the function exists.
        if sys.platform == "win32":
            # Allow the loader to find the dependencies of the library, and the library itself when Numba loads it.
            os.add_dll_directory(path)
        name = os.path.join(path, name)
    return CDLL(name)


def _library_version(libtbb: CDLL) -> int:
    version_func = libtbb.TBB_runtime_interface_version
    version_func.argtypes = []
    version_func.restype = c_int
    return version_func()


def get_tbb_version(path: str | None = None) -> int:
    """Get TBB library version.

    :param path: directory of the TBB library. If not given, the default search path of the operating system is used.
    :return: TBB runtime interface version
    :raises OSError: if the library cannot be loaded
    """
    return _library_version(_load_library(path))


def load_tbb() -> int | None:
    """Load a TBB library that is compatible with Numba.

    The default search path of the operating system is tried first,
    and then the library directories of the Python environment given by :func:`tbb_library_dirs`.

    :return: TBB runtime interface version of the loaded library, or None if no TBB library was found.
    """
    if TBB_LIBRARY_NAME is None:
        logger.warning("Cannot load TBB on an unknown operating system.")
        return None

    try:
        libtbb = _load_library()
        system_version = _library_version(libtbb)
    except OSError:
        system_version = None
    else:
        if system_version >= TBB_MIN_VERSION:
            _LOADED_LIBRARIES.append(libtbb)
            return system_version
        logger.warning(
            "The TBB found from the default library search path is too old for Numba: %s < %s. "
            "Trying the library directories of the Python environment.",
            system_version, TBB_MIN_VERSION
        )

    for lib_dir in tbb_library_dirs():
        if not os.path.isfile(os.path.join(lib_dir, TBB_LIBRARY_NAME)):
            continue
        try:
            libtbb = _load_library(lib_dir)
            version = _library_version(libtbb)
        except OSError as e:
            logger.warning("Failed to load the TBB library from %s: %s", lib_dir, e)
            continue
        if version >= TBB_MIN_VERSION:
            _LOADED_LIBRARIES.append(libtbb)
            logger.debug("Loaded TBB version %s from %s", version, lib_dir)
            return version
        logger.warning("The TBB library at %s is too old for Numba: %s < %s.", lib_dir, version, TBB_MIN_VERSION)

    if system_version is None:
        logger.info(
            "TBB was not found. Numba will use another threading layer, which may not support nested parallelism. "
            "You can install TBB with the \"performance\" extra of PTtools."
        )
        return None
    logger.error(
        "The installed TBB is too old for Numba: %s < %s. "
        "Please install a newer version, e.g. with the \"performance\" extra of PTtools.",
        system_version, TBB_MIN_VERSION
    )
    return system_version


if __name__ == "__main__":
    tbb_version = load_tbb()
    print("TBB version:", tbb_version)
    if tbb_version is None or tbb_version < TBB_MIN_VERSION:
        sys.exit(1)
else:
    load_tbb()
