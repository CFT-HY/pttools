"""Type hints for simplifying and unifying PTtools code"""

import typing as tp

import numpy as np
import scipy.integrate as spi

from pttools.speedup.numba_wrapper import CPUDispatcher

# Function and object types
#: Numba function
NumbaFunc = tp.Union[callable, CPUDispatcher]
#: ODE solver specifier
ODESolver = tp.Union[spi.OdeSolver, tp.Type[spi.OdeSolver], tp.Type[spi.odeint], str]

# Numerical types
#: Float list or a Numpy array
FloatListOrArr = tp.Union[tp.List[float], np.ndarray]
#: Float or a Numpy array
FloatOrArr = tp.Union[float, np.ndarray]
#: The return type of a Numba function that returns a float or a Numpy array
FloatOrArrNumba = tp.Union[float, np.ndarray, NumbaFunc]
#: Integer or a Numpy array
IntOrArr = tp.Union[int, np.ndarray]
