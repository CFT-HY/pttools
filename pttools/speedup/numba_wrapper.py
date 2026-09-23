"""Wrapper for importing Numba libraries without version dependencies."""

# ruff: noqa: F401

import logging
import shutil
import subprocess
import sys
import typing as tp

import numba

try:
    from numba.core.ccallback import CFunc
    from numba.core.dispatcher import Dispatcher
    from numba.core.registry import CPUDispatcher
    from numba.experimental import jitclass
    #: Whether the Numba version used is from before the major refactoring of the module structure.
    NUMBA_OLD_STRUCTURE = False
except ImportError:
    from numba import jitclass
    from numba.ccallback import CFunc
    from numba.dispatcher import Dispatcher
    from numba.targets.registry import CPUDispatcher
    NUMBA_OLD_STRUCTURE = True
import numpy as np

from pttools.speedup import options
import pttools.type_hints as th

logger = logging.getLogger(__name__)

OLD_NUMBALSODA = False
if options.NUMBA_DISABLE_JIT:
    # As of 0.3.3 NumbaLSODA can't be imported when Numba is disabled
    numbalsoda = None
else:
    try:
        import numbalsoda
    except ImportError:
        try:
            import NumbaLSODA as numbalsoda  # noqa: N813
            OLD_NUMBALSODA = True
            logger.warning(
                "You are using an old version of NumbaLSODA. "
                "Please upgrade, as compatibility may break without notice."
            )
        except ImportError:
            numbalsoda = None
            logger.warning(
                "Could not import NumbaLSODA. "
                "As it's a relatively new library, "
                "it may not have been installed automatically by your package manager. "
                "To use NumbaLSODA, please see the PTtools documentation on how to install it manually."
            )
    except OSError as e:
        # NumbaLSODA requires an executable stack, which is not enabled by default on Linux 6.14.
        # https://github.com/Nicholaswogan/numbalsoda/issues/34
        _lib_path, _sep, _err_msg = str(e).partition(": ")
        if not _sep or _err_msg != "cannot enable executable stack as shared object requires: Invalid argument":
            raise e
        if shutil.which("execstack") is None:
            raise OSError(
                "NumbaLSODA requires an executable stack to run. To enable it, "
                "please install execstack with e.g. \"sudo apt install execstack\" and run this program again."
            ) from e
        subprocess.run(["execstack", "-c", _lib_path], check=False)
        import numbalsoda

if numbalsoda is None:
    if options.NUMBA_INTEGRATE:
        raise ImportError("Numba-jitted integration has been enabled, but NumbaLSODA is not available.")
    lsoda_sig = numba.types.void(
        numba.types.double,
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double))
else:
    lsoda_sig = numbalsoda.lsoda_sig

#: Numba version number
#: (The value shown in the PTtools documentation is the version the documentation has been built with.)
NUMBA_VERSION = tuple(int(val) for val in numba.__version__.split("."))
#: Whether the Numba version used is prone to segfaulting when profiled.
#: https://github.com/numba/numba/issues/3229
#: https://github.com/numba/numba/issues/3625
NUMBA_SEGFAULTING_PROFILERS: bool = NUMBA_VERSION < (0, 49, 0)
NUMBA_PYINSTRUMENT_INCOMPATIBLE_PYTHON_VERSION: bool = \
    sys.version_info.major > 3 or \
    (sys.version_info.major == 3 and sys.version_info.minor >= 11)

if NUMBA_OLD_STRUCTURE:
    logger.warning(
        "You are using an old Numba version, which has the old module structure. "
        "Please upgrade, as compatibility may break without notice.")
if NUMBA_SEGFAULTING_PROFILERS:
    logger.warning(
        "You are using an old Numba version, which is prone to segfaulting when profiled. Please upgrade.")

# For e.g. the cases where Numba does not understand a None as a default value
# TODO: Move this to utils
NAN_ARR: tp.Final[th.FloatArr1D] = np.array([np.nan], dtype=np.float64)
NAN_ARR.flags.writeable = False
# nan_arr.setflags(write=False)
