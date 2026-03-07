"""Type hints for simplifying and unifying PTtools code"""

import ctypes
import typing as tp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numba
from numba.core.registry import CPUDispatcher
import numpy as np
from numpy.typing import NDArray
import scipy.integrate as spi

# This adds quite a bit of startup time when only the type hints are needed, and not the rest of PTtools.
# from pttools.speedup.numba_wrapper import CPUDispatcher

# -----
# Function and object types
# -----
type AxesArr1D = np.ndarray[tuple[int], np.dtype[Axes]]
type AxesArr2D = np.ndarray[tuple[int, int], np.dtype[Axes]]
type AxesArr3D = np.ndarray[tuple[int, int, int], np.dtype[Axes]]
type FigArr1D = np.ndarray[tuple[int], np.dtype[Figure]]
type FigArr2D = np.ndarray[tuple[int, int], np.dtype[Figure]]
#: Numba function
type NumbaFunc = tp.Callable | CPUDispatcher
#: ODE solver specifier
type ODESolver = spi.OdeSolver | type[spi.OdeSolver] | type[spi.odeint] | str

# -----
# Numerical types
# -----
type Bool = np.dtype[np.bool_]
type BoolArr = NDArray[np.bool_]
type BoolArr1D = np.ndarray[tuple[int], Bool]  # pylint: disable=invalid-name
type BoolArr2D = np.ndarray[tuple[int, int], Bool]  # pylint: disable=invalid-name
# np.float64 is a subclass of float, so for scalars specifying "float" is sufficient.
type Float64 = np.dtype[np.float64]
type FloatArr = NDArray[np.float64]
type FloatArr1D = np.ndarray[tuple[int], Float64]  # pylint: disable=invalid-name
type FloatArr1DOrList = FloatArr1D | list[float]
type FloatArr2D = np.ndarray[tuple[int, int], Float64]  # pylint: disable=invalid-name
type FloatArr3D = np.ndarray[tuple[int, int, int], Float64]  # pylint: disable=invalid-name
type FloatArr4D = np.ndarray[tuple[int, int, int, int], Float64]
# Float list or a Numpy array
# FloatListOrArr = list[tp.Union[float, Float64] | np.ndarray
#: Float or a Numpy array of floats
type FloatOrArr = float | FloatArr
#: Float or a 1D Numpy array of floats
type FloatOrArr1D = float | FloatArr1D
type FloatOrArr1D2D = FloatOrArr1D | FloatArr2D
#: The return type of Numba function that returns a float or a Numpy array
type FloatOrArrNumba = float | FloatArr | NumbaFunc
#: Integer or a Numpy array of integers
type Int = np.dtype[np.int_]
type IntArr1D = np.ndarray[tuple[int], Int]
type IntArr2D = np.ndarray[tuple[int, int], Int]
type IntOrArr = int | NDArray[np.int_]

type VWXi = tuple[FloatArr1D, FloatArr1D, FloatArr1D]

# -----
# CS2
# -----
#: Type of $c_s^2$ function$
type CS2Fun = tp.Callable[[FloatOrArr, FloatOrArr], FloatOrArr] | CPUDispatcher
#: Numba type of $c_s^2$ function$
CS2FunScalarSig = numba.double(numba.double, numba.double)
#: Numba pointer to a $c_s^2$ function$
type CS2FunScalarPtr = numba.types.CPointer  # CPointer(CS2FunScalarSig)
#: ctypes type of $c_s^2$ function$
CS2CFunc = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
#: Python type of $c_s^2$ function$
type CS2CFuncType = tp.Type[CS2CFunc]

# -----
# Other
# -----
type FSolveOutput = tuple[NDArray, dict, int, str]
type Interpolation = tp.Literal["nearest", "linear", "cubic"]
