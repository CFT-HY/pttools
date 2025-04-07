"""A fix for loading Intel Thread Building Blocks (TBB) for Numba

https://github.com/numba/numba/issues/7531

Based on numba.np.ufunc.parallel._check_tbb_version_compatible()
"""
# https://github.com/numba/numba/issues/7531#issuecomment-1614510255

import logging
import os
import sys
import typing as tp
from ctypes import CDLL, c_int

logger = logging.getLogger(__name__)

# From numba.np.ufunc.parallel
IS_OSX = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')
IS_WINDOWS = sys.platform.startswith('win32')

# As required by Numba in numba.np.ufunc.parallel._check_tbb_version_compatible()
TBB_MIN_VERSION = 12060


def get_tbb_version(path: str = None):
    """Get TBB library version"""
    if IS_WINDOWS:
        libtbb_name = 'tbb12.dll'
    elif IS_OSX:
        libtbb_name = 'libtbb.12.dylib'
    elif IS_LINUX:
        libtbb_name = 'libtbb.so.12'
    else:
        raise ValueError("Unknown operating system")

    if path is not None:
        libtbb_name = os.path.join(path, libtbb_name)
    libtbb = CDLL(libtbb_name)
    version_func = libtbb.TBB_runtime_interface_version
    version_func.argtypes = []
    version_func.restype = c_int
    return version_func()


def load_tbb() -> tp.Optional[int]:
    """Update environment variables so that the proper TBB is found

    This may not work
    https://stackoverflow.com/a/52408140
    """
    try:
        tbb_version = get_tbb_version()
        if tbb_version >= TBB_MIN_VERSION:
            return tbb_version

        venv_path = os.getenv("VIRTUAL_ENV", default=None)
        if venv_path is None:
            logger.error(
                "Automatically found TBB is too old for Numba, "
                "and no virtualenv was detected to load a newer version.")
            return tbb_version

        venv_lib_path = os.path.join(venv_path, "lib")
        ld_library_path = os.getenv("LD_LIBRARY_PATH", default=None)
        if ld_library_path is None:
            os.environ["LD_LIBRARY_PATH"] = venv_lib_path
        else:
            os.environ["LD_LIBRARY_PATH"] = venv_lib_path + os.pathsep + ld_library_path
        os.environ["PATH"] = venv_lib_path + os.pathsep + os.environ["PATH"]

        venv_tbb_version = get_tbb_version(venv_lib_path)
        if venv_tbb_version < TBB_MIN_VERSION:
            logger.error("The TBB version in the virtualenv is too old: %s. Please upgrade.", venv_tbb_version)
            return venv_tbb_version
        tbb_version2 = get_tbb_version()
        if tbb_version2 < TBB_MIN_VERSION:
            logger.error(
                "Fixing TBB load path did not work to update from globally installed %s to %s in the virtualenv. "
                "Please set LD_LIBRARY_PATH=\"%s\" manually.",
                tbb_version, venv_tbb_version, venv_lib_path
            )
        # print(os.environ["LD_LIBRARY_PATH"])
        # print(os.environ["PATH"])
        return tbb_version2
    except Exception as e:
        logger.exception("Error with loading TBB:", exc_info=e)
        return None


if __name__ == "__main__":
    print("TBB version:", load_tbb())
else:
    if IS_LINUX:
        load_tbb()
