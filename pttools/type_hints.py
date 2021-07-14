"""Type hints for simplifying and unifying PTtools code"""

import logging
import typing as tp

try:
    from numba.core.registry import CPUDispatcher
    from numba.core.ccallback import CFunc
    NUMBA_IS_OLD = False
except ImportError:
    from numba.ccallback import CFunc
    from numba.targets.registry import CPUDispatcher
    NUMBA_IS_OLD = True
import numpy as np
import scipy.integrate as spi

logger = logging.getLogger(__name__)
if NUMBA_IS_OLD:
    logger.warning("You are using an old version of Numba. Please upgrade, as compatibility may break without notice.")

# Function and object types
NUMBA_FUNC = tp.Union[callable, CPUDispatcher]
ODE_SOLVER = tp.Union[spi.OdeSolver, tp.Type[spi.OdeSolver], tp.Type[spi.odeint], str]

# Numerical types
FLOAT_OR_ARR = tp.Union[float, np.ndarray]
FLOAT_OR_ARR_NUMBA = tp.Union[float, np.ndarray, NUMBA_FUNC]
INT_OR_ARR = tp.Union[int, np.ndarray]
