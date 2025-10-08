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
NumbaFunc = tp.Callable | CPUDispatcher
#: ODE solver specifier
ODESolver = spi.OdeSolver | type[spi.OdeSolver] | type[spi.odeint] | str

# Numerical types
# np.float64 is a subclass of float, so there is no need to specify it explicitly for scalars.
FloatArr = np.ndarray[tuple[int, ...], np.float64]
FloatArr1D = np.ndarray[tuple[int], np.float64]
# Float list or a Numpy array
# FloatListOrArr = list[tp.Union[float, np.float64]] | np.ndarray
#: Float or a Numpy array of floats
FloatOrArr = float | FloatArr
#: Float or a 1D Numpy array of floats
FloatOrArr1D = float | FloatArr1D
#: The return type of a Numba function that returns a float or a Numpy array
FloatOrArrNumba = float | np.ndarray | NumbaFunc
#: Integer or a Numpy array of integers
IntOrArr = int | np.ndarray[tuple[int], np.int_]

#: Type of a cs2 function
CS2Fun = tp.Callable[[FloatOrArr, FloatOrArr], FloatOrArr] | CPUDispatcher
#: Numba type of a cs2 function
CS2FunScalarSig = numba.double(numba.double, numba.double)
#: Numba pointer to a cs2 function
CS2FunScalarPtr = numba.types.CPointer(CS2FunScalarSig)
#: ctypes type of a cs2 function
CS2CFunc = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)

# Other
Interpolation = tp.Literal["nearest", "linear", "cubic"]
