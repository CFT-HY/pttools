import logging

try:
    from numba.core.ccallback import CFunc
    from numba.core.dispatcher import Dispatcher
    from numba.core.registry import CPUDispatcher
    NUMBA_IS_OLD = False
except ImportError:
    from numba.ccallback import CFunc
    from numba.dispatcher import Dispatcher
    from numba.targets.registry import CPUDispatcher
    NUMBA_IS_OLD = True

logger = logging.getLogger(__name__)
if NUMBA_IS_OLD:
    logger.warning("You are using an old version of Numba. Please upgrade, as compatibility may break without notice.")
