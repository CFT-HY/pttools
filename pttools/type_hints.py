"""Type hints for simplifying and unifying PTtools code"""

import typing as tp

import numpy as np
import scipy.integrate as spi

from pttools.speedup.numba_wrapper import CPUDispatcher

# Function and object types
NumbaFunc = tp.Union[callable, CPUDispatcher]
ODESolver = tp.Union[spi.OdeSolver, tp.Type[spi.OdeSolver], tp.Type[spi.odeint], str]

# Numerical types
FloatOrArr = tp.Union[float, np.ndarray]
FloatOrArrNumba = tp.Union[float, np.ndarray, NumbaFunc]
IntOrArr = tp.Union[int, np.ndarray]
