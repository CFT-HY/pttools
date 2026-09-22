"""Type hints for simplifying and unifying PTtools code.

Many PTtools functions accept both floats and Numpy arrays, and return a float if all of their arguments are floats,
and an array if at least one of their arguments is an array.
Python has no dedicated syntax for such a dependency between the argument and return types,
but it can be expressed with a type parameter that is bound to this type:

.. code-block:: python

    @njit
    def v_shock[T: FloatOrArr](xi: T, cs: T | float = CS0) -> T:
        return (3 * xi ** 2 - cs ** 2) / (2 * xi)  # pyrefly: ignore[bad-return]

The type checker solves the type parameter as the join of the argument types,
which gives ``float`` when all of the arguments are floats and :py:data:`FloatArr` when they all are arrays.
The auxiliary arguments are declared as ``T | float`` instead of ``T``,
so that passing a float for them does not widen the solution of ``T``.
This is required for the type parameter to propagate through generic callers such as

.. code-block:: python

    @njit
    def v_shock_bag[T: FloatOrArr](xi: T) -> T:
        return v_shock(xi, CS0)

The type checker cannot infer that the arithmetic within the function body preserves ``T``,
and therefore the return values have to be either cast with ``typing.cast()``
or marked with a ``# pyrefly: ignore[bad-return]`` comment.
The comment is preferred, since ``typing.cast()`` is a function call
that Numba cannot compile and that adds overhead elsewhere.
Note that a ``# type: ignore[...]`` comment would silence *all* the errors on its line,
since Pyrefly does not recognise the error codes of Mypy,
whereas ``# pyrefly: ignore[bad-return]`` silences only the return type error.

There are two exceptions to this pattern:

- Pyrefly is currently unable to verify overrides of methods that have a ``T | float`` argument,
  so for the methods of the :py:class:`pttools.models.model.Model` classes
  the auxiliary arguments are declared as :py:data:`FloatOrArr` instead.
- ``numba.extending.overload()`` requires the typing function and its implementations
  to have identical parameter annotations,
  so the scalar and array implementations of an overloaded function keep the :py:data:`FloatOrArr` annotations,
  and only the public function that dispatches between them is made generic.
"""

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
# These are object arrays. Numpy typing has no way of expressing the element type of an object array,
# but declaring the element type here does give the correct types when the arrays are indexed.
# The ignore comments are needed, since Matplotlib objects are not subclasses of np.generic.
type AxesArr1D = np.ndarray[tuple[int], np.dtype[Axes]]  # pyrefly: ignore[bad-specialization]
type AxesArr2D = np.ndarray[tuple[int, int], np.dtype[Axes]]  # pyrefly: ignore[bad-specialization]
type AxesArr3D = np.ndarray[tuple[int, int, int], np.dtype[Axes]]  # pyrefly: ignore[bad-specialization]
type FigArr1D = np.ndarray[tuple[int], np.dtype[Figure]]  # pyrefly: ignore[bad-specialization]
type FigArr2D = np.ndarray[tuple[int, int], np.dtype[Figure]]  # pyrefly: ignore[bad-specialization]
#: Numba function
type NumbaFunc = tp.Callable | CPUDispatcher
#: ODE solver specifier
type ODESolver = spi.OdeSolver | type[spi.OdeSolver] | tp.Callable | str

# -----
# Numerical types
# -----
type Bool = np.dtype[np.bool_]
type BoolArr = NDArray[np.bool_]
type BoolArr1D = np.ndarray[tuple[int], Bool]
type BoolArr2D = np.ndarray[tuple[int, int], Bool]
# np.float64 is a subclass of float, so for scalars specifying "float" is sufficient.
type Float64 = np.dtype[np.float64]
#: Numpy array of floats
type FloatArr = NDArray[np.float64]
#: 1D Numpy array of floats
type FloatArr1D = np.ndarray[tuple[int], Float64]
type FloatArr1DOrList = FloatArr1D | list[float]
type FloatArr2D = np.ndarray[tuple[int, int], Float64]
type FloatArr3D = np.ndarray[tuple[int, int, int], Float64]
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
#: Type of $c_s^2$ function
type CS2Fun = tp.Callable[[FloatOrArr, FloatOrArr], FloatOrArr] | CPUDispatcher
#: Numba type of $c_s^2$ function
CS2FunScalarSig = numba.double(numba.double, numba.double)
#: Pointer to a $c_s^2$ function, i.e. the address of a Numba cfunc
type CS2FunScalarPtr = int
#: ctypes type of $c_s^2$ function
CS2CFunc = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
#: Python type of a $c_s^2$ ctypes function pointer instance,
#: as created by calling :py:data:`CS2CFunc`
type CS2CFuncType = tp.Callable[[float, float], float]

# -----
# Other
# -----
type FSolveOutput = tuple[NDArray, dict, int, str]
type Interpolation = tp.Literal["nearest", "linear", "cubic"]
