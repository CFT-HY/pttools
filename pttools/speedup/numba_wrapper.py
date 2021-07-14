import logging

import numba
try:
    from numba.core.ccallback import CFunc
    from numba.core.dispatcher import Dispatcher
    from numba.core.registry import CPUDispatcher
    from numba.experimental import jitclass
    NUMBA_OLD_STRUCTURE = False
except ImportError:
    from numba import jitclass
    from numba.ccallback import CFunc
    from numba.dispatcher import Dispatcher
    from numba.targets.registry import CPUDispatcher
    NUMBA_OLD_STRUCTURE = True

logger = logging.getLogger(__name__)

NUMBA_VERSION = tuple(int(val) for val in numba.__version__.split("."))
# https://github.com/numba/numba/issues/3229
# https://github.com/numba/numba/issues/3625
NUMBA_SEGFAULTING_PROFILERS = NUMBA_VERSION < (0, 49, 0)

if NUMBA_OLD_STRUCTURE:
    logger.warning(
        "You are using an old Numba version, which has the old module structure. "
        "Please upgrade, as compatibility may break without notice.")
if NUMBA_SEGFAULTING_PROFILERS:
    logger.warning(
        "You are using an old Numba version, which is prone to segfaulting when profiled. Please upgrade.")
