"""Numerical utilities."""

import math
import typing as tp

import numba
from numba.extending import overload
import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from pttools.speedup import njit, threads
import pttools.type_hints as th

#: Smallest float
EPS: tp.Final[np.float64] = np.nextafter(0, 1)
#: Powers of 2 up to a certain value, see :py:func:`pttools.speedup.threads.powers_of_2`.
#: It's implemented in :py:mod:`pttools.speedup`, as :py:mod:`pttools.speedup` cannot depend on :py:mod:`pttools.math`.
powers_of_2 = threads.powers_of_2


def finite_edge(
        func: tp.Callable[[float], float],
        x_finite: float,
        x_nan: float,
        n_iter: int = 60) -> float:
    """Find the edge of the region where the function is finite with bisection.

    The function should be finite at ``x_finite`` and non-finite (e.g. nan) at ``x_nan``,
    and have a single edge between them.

    :param func: function to evaluate
    :param x_finite: point where the function is finite
    :param x_nan: point where the function is not finite
    :param n_iter: number of bisection steps. Each step halves the distance to the edge.
    :return: the last point found, where the function is finite
    """
    for _ in range(n_iter):
        x_mid = (x_finite + x_nan) / 2
        if np.isfinite(func(x_mid)):
            x_finite = x_mid
        else:
            x_nan = x_mid
    return x_finite


def isclose_scalar(a: float, b: float, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    r"""Whether two scalars are equal within a tolerance.

    This gives the same results as :func:`numpy.isclose` with the same tolerances,
    $\lvert a - b \rvert \leq \text{atol} + \text{rtol} \lvert b \rvert$, including for infinities and nan,
    but is an order of magnitude faster for scalars,
    which matters in the functions that are called by the solvers in loops.

    This is a Python function, as calling a jitted function from Python is not faster,
    and with omitted default arguments it is about 100 times slower due to the overhead of the Numba dispatcher.
    Jitted functions can call this as well, as it's compiled for them with :func:`numba.extending.overload`.

    :param a: the first value
    :param b: the second value, to which the relative tolerance is applied
    :param rtol: relative tolerance
    :param atol: absolute tolerance
    :return: whether the values are close
    """
    # This handles the infinities of the same sign, for which a - b would be nan.
    if a == b:
        return True
    # As in NumPy, nan and unequal infinities are not close, even though e.g. |0 - inf| <= rtol * inf.
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return bool(abs(a - b) <= atol + rtol * abs(b))


@overload(isclose_scalar, jit_options={"nopython": True, "nogil": True})
def _isclose_scalar_numba(a: float, b: float, rtol: float = 1e-5, atol: float = 1e-8) -> th.NumbaFunc:
    """Compile :func:`isclose_scalar` for jitted functions."""
    return isclose_scalar


def is_nan_or_none(value: float | None = None) -> bool:
    """Determine if a value is NaN or None."""
    return value is None or np.isnan(value)


@tp.overload
def is_nan_or_outside(values: float, lower: float, upper: float, isclose: bool = False) -> bool: ...


@tp.overload
def is_nan_or_outside(
        values: tuple[float, ...],
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        isclose: bool = False) -> bool: ...


def is_nan_or_outside(
        values: float | tuple[float, ...],
        lower: float | tuple[float, ...],
        upper: float | tuple[float, ...],
        isclose: bool = False) -> bool:
    r"""Whether a value is nan or outside its bounds, or any of several values is.

    The bounds are inclusive, i.e. a value equal to a bound is inside the bounds, unless ``isclose=True``.
    Infinite bounds are allowed, e.g. ``upper=np.inf`` for values that have only a lower bound.

    Several values are given as tuples of floats, which are faster to pass to the compiled functions than arrays.
    The values and the bounds of a tuple must all be floats, e.g. ``0.`` instead of ``0``,
    as otherwise they would have to be compiled for a heterogeneous tuple.

    :param values: a value or a tuple of values
    :param lower: the lower bound or a tuple of the lower bounds of the values
    :param upper: the upper bound or a tuple of the upper bounds of the values
    :param isclose: whether a value that is close to a finite bound according to :func:`isclose_scalar`
        is also considered to be outside the bounds
    :return: whether the value or any of the values is nan or outside its bounds
    """
    # The compiled functions are called with all the arguments,
    # as omitting default arguments makes the calls of jitted functions from Python a lot slower.
    if isinstance(values, tuple):
        return _is_nan_or_outside_tuple(
            values, tp.cast(tuple[float, ...], lower), tp.cast(tuple[float, ...], upper), isclose)
    return _is_nan_or_outside_scalar(values, tp.cast(float, lower), tp.cast(float, upper), isclose)


@overload(is_nan_or_outside, jit_options={"nopython": True, "nogil": True})
def _is_nan_or_outside_numba(
        values: float | tuple[float, ...],
        lower: float | tuple[float, ...],
        upper: float | tuple[float, ...],
        isclose: bool = False) -> th.NumbaFunc:
    """Compile :func:`is_nan_or_outside` for jitted functions, choosing the implementation by the argument types.

    The Python function is not compiled itself, as it's faster to call from Python than a jitted function.
    The implementations must have the same signature as this function, including the type hints.
    """
    if isinstance(values, numba.types.BaseTuple):
        def impl_tuple(
                values: float | tuple[float, ...],
                lower: float | tuple[float, ...],
                upper: float | tuple[float, ...],
                isclose: bool = False) -> bool:
            return _is_nan_or_outside_tuple(values, lower, upper, isclose)  # pyrefly: ignore[bad-argument-type]
        return impl_tuple

    def impl_scalar(
            values: float | tuple[float, ...],
            lower: float | tuple[float, ...],
            upper: float | tuple[float, ...],
            isclose: bool = False) -> bool:
        return _is_nan_or_outside_scalar(values, lower, upper, isclose)  # pyrefly: ignore[bad-argument-type]
    return impl_scalar


@njit(cache=True)
def _is_nan_or_outside_scalar(value: float, lower: float, upper: float, isclose: bool) -> bool:
    """See :func:`is_nan_or_outside`."""
    if np.isnan(value) or value < lower or value > upper:
        return True
    if not isclose:
        return False
    return (np.isfinite(lower) and isclose_scalar(value, lower)) \
        or (np.isfinite(upper) and isclose_scalar(value, upper))


@njit(cache=True)
def _is_nan_or_outside_tuple(
        values: tuple[float, ...],
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        isclose: bool) -> bool:
    """See :func:`is_nan_or_outside`."""
    if len(lower) != len(values) or len(upper) != len(values):
        raise ValueError("The values and their bounds must have the same lengths.")
    # A loop is used instead of any() with a generator, as this is compiled by Numba.
    for i in range(len(values)):  # noqa: SIM110
        if _is_nan_or_outside_scalar(values[i], lower[i], upper[i], isclose):
            return True
    return False


def rel_diff_arr[T: NDArray](x: T, y: T) -> T:
    """Relative differences of two arrays."""
    if not np.count_nonzero(y):
        return np.full_like(x, np.nan)
    nonzero = y != 0
    return np.abs(x - y)[nonzero] / np.abs(y[nonzero])


def rel_diff_scalar(x: float, y: float) -> float:
    """Relative difference of two scalars."""
    if y == 0:
        return np.nan
    return abs(x - y) / y


def running_mean(values: th.FloatArr1D, window: int) -> th.FloatArr1D:
    r"""Running mean with mirrored boundaries.

    :param values: the values
    :param window: number of points in the window, which must be odd
    :return: the running mean
    """
    return savgol_filter(values, window_length=window, polyorder=0, mode="mirror")
