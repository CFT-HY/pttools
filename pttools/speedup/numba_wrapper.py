"""Wrapper for importing Numba libraries without version dependencies"""

# pylint: disable=unused-import

import logging

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

from . import options
OLD_NUMBALSODA = False
if options.NUMBA_DISABLE_JIT:
    # As of 0.3.3 NumbaLSODA can't be imported when Numba is disabled
    # pylint: disable=invalid-name
    numbalsoda = None
else:
    try:
        import numbalsoda
    except ImportError:
        try:
            import NumbaLSODA as numbalsoda
            OLD_NUMBALSODA = True
        except ImportError:
            numbalsoda = None

if numbalsoda is None:
    lsoda_sig = numba.types.void(
        numba.types.double,
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double))
else:
    lsoda_sig = numbalsoda.lsoda_sig

logger = logging.getLogger(__name__)

#: Numba version number
#: (The value shown in the documentation is the version the documentation has been built with.)
NUMBA_VERSION = tuple(int(val) for val in numba.__version__.split("."))
#: Whether the Numba version used is prone to segfaulting when profiled.
#: https://github.com/numba/numba/issues/3229
#: https://github.com/numba/numba/issues/3625
NUMBA_SEGFAULTING_PROFILERS = NUMBA_VERSION < (0, 49, 0)

if NUMBA_OLD_STRUCTURE:
    logger.warning(
        "You are using an old Numba version, which has the old module structure. "
        "Please upgrade, as compatibility may break without notice.")
if NUMBA_SEGFAULTING_PROFILERS:
    logger.warning(
        "You are using an old Numba version, which is prone to segfaulting when profiled. Please upgrade.")
if numbalsoda is None:
    logger.warning(
        "Could not import NumbaLSODA. "
        "As it's a relatively new library, it may not have been installed automatically by your package manager. "
        "To use NumbaLSODA, please see the PTtools documentation on how to install it manually."
    )
elif OLD_NUMBALSODA:
    logger.warning(
        "You are using an old version of NumbaLSODA. "
        "Please upgrade, as compatibility may break without notice.")
