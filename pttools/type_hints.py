"""Type hints for simplifying and unifying PTtools code"""

import typing as tp

import numpy as np
import scipy.integrate as spi

from pttools.speedup.numba_wrapper import CPUDispatcher

# Function and object types
NUMBA_FUNC = tp.Union[callable, CPUDispatcher]
ODE_SOLVER = tp.Union[spi.OdeSolver, tp.Type[spi.OdeSolver], tp.Type[spi.odeint], str]

# Numerical types
FLOAT_OR_ARR = tp.Union[float, np.ndarray]
FLOAT_OR_ARR_NUMBA = tp.Union[float, np.ndarray, NUMBA_FUNC]
INT_OR_ARR = tp.Union[int, np.ndarray]
