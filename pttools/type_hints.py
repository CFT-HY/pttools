"""Type hints for simplifying and unifying PTtools code"""

import ctypes
import typing as tp

import numba
from numba.core.registry import CPUDispatcher
import numpy as np
import scipy.integrate as spi

# This adds quite a bit of startup time when only the type hints are needed, and not the rest of PTtools.
# from pttools.speedup.numba_wrapper import CPUDispatcher

# Function and object types
#: Numba function
NumbaFunc = tp.Union[callable, CPUDispatcher]
#: ODE solver specifier
ODESolver = tp.Union[spi.OdeSolver, tp.Type[spi.OdeSolver], tp.Type[spi.odeint], str]

# Numerical types
#: Float list or a Numpy array
FloatListOrArr = tp.Union[tp.List[tp.Union[float, np.float64]], np.ndarray]
#: Float or a Numpy array
FloatOrArr = tp.Union[float, np.float64, np.ndarray]
#: The return type of a Numba function that returns a float or a Numpy array
FloatOrArrNumba = tp.Union[float, np.float64, np.ndarray, NumbaFunc]
#: Integer or a Numpy array
IntOrArr = tp.Union[int, np.ndarray]

#: Type of a cs2 function
CS2Fun = tp.Union[tp.Callable[[FloatOrArr, FloatOrArr], FloatOrArr], CPUDispatcher]
#: Numba type of a cs2 function
CS2FunScalarSig = numba.double(numba.double, numba.double)
#: Numba pointer to a cs2 function
CS2FunScalarPtr = numba.types.CPointer(CS2FunScalarSig)
#: ctypes type of a cs2 function
CS2CFunc = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)

# Other
Interpolation = tp.Literal["nearest", "linear", "cubic"]
