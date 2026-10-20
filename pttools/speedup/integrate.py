r"""Numba overloads for :external:py:mod:`scipy.integrate`.

Numba requires that the parameters of an overload typing function and of the implementation it returns
are identical, including their type annotations.
Therefore, the parameters are annotated with :data:`typing.Any`.
"""

import typing as tp

import numba
from numba.extending import overload
import numpy as np
import scipy.integrate

from pttools.speedup.jit import njit
import pttools.type_hints as th


@njit(cache=True, nogil=True)
def simpson_h(y: th.FloatArr1D, h: th.FloatArr1D) -> float:
    r"""Composite Simpson's rule for a 1D array with the given intervals.

    This is the algorithm of :external:py:func:`scipy.integrate.simpson`.

    The rule for irregularly spaced data is applied to each pair of intervals
    $[x_{i}, x_{i+2}]$ with $h_0 = x_{i+1} - x_i$ and $h_1 = x_{i+2} - x_{i+1}$:
    $$\int_{x_i}^{x_{i+2}} y dx \approx \frac{h_0 + h_1}{6} \left[
    \left( 2 - \frac{h_1}{h_0} \right) y_i
    + \frac{(h_0 + h_1)^2}{h_0 h_1} y_{i+1}
    + \left( 2 - \frac{h_0}{h_1} \right) y_{i+2} \right].$$
    For an even number of points, the last interval is integrated with the correction of Cartwright,
    $$\int_{x_{N-2}}^{x_{N-1}} y dx \approx \alpha y_{N-1} + \beta y_{N-2} - \eta y_{N-3},$$
    $$\alpha = \frac{2h_1^2 + 3h_0 h_1}{6(h_0 + h_1)}, \quad
    \beta = \frac{h_1^2 + 3h_0 h_1}{6h_0}, \quad
    \eta = \frac{h_1^3}{6h_0(h_0 + h_1)},$$
    where $h_0$ and $h_1$ are the last two intervals.
    For two points, the trapezoidal rule is used, and for a single point the integral is zero.

    :param y: values of the integrand
    :param h: intervals between the sample points, $h_i = x_{i+1} - x_i$
    :return: the integral
    """
    n = y.size
    if n < 2:  # noqa: PLR2004
        return 0.
    if n == 2:  # noqa: PLR2004
        return 0.5 * h[0] * (y[0] + y[1])
    # The pairs of intervals cover the points 0 ... n-1 for odd n, and 0 ... n-2 for even n.
    stop = n - 2 if n % 2 else n - 3
    result = 0.
    for i in range(0, stop, 2):
        h0 = h[i]
        h1 = h[i + 1]
        hsum = h0 + h1
        result += hsum / 6. * (
            y[i] * (2. - h1 / h0)
            + y[i + 1] * hsum * hsum / (h0 * h1)
            + y[i + 2] * (2. - h0 / h1)
        )
    if n % 2 == 0:
        h0 = h[-2]
        h1 = h[-1]
        alpha = (2 * h1**2 + 3 * h0 * h1) / (6 * (h1 + h0))
        beta = (h1**2 + 3 * h0 * h1) / (6 * h0)
        eta = h1**3 / (6 * h0 * (h0 + h1))
        result += alpha * y[-1] + beta * y[-2] - eta * y[-3]
    return result


@njit(cache=True)
def _validate_axis(axis: int) -> None:
    """Validate the axis of a 1D array. The error type is the same as in SciPy."""
    if axis not in (-1, 0):
        raise IndexError("Only 1D arrays are supported, so axis must be -1 or 0.")


def simpson_x(y: tp.Any, x: tp.Any = None, dx: tp.Any = 1.0, axis: tp.Any = -1) -> float:
    """Implementation of :external:py:func:`scipy.integrate.simpson` for 1D arrays with sample points."""
    _validate_axis(axis)
    if x.size != y.size:
        raise ValueError("x and y must have the same size.")
    return simpson_h(y, np.diff(x))


def simpson_dx(y: tp.Any, x: tp.Any = None, dx: tp.Any = 1.0, axis: tp.Any = -1) -> float:
    """Implementation of :external:py:func:`scipy.integrate.simpson` for 1D arrays with uniform spacing."""
    _validate_axis(axis)
    return simpson_h(y, np.full(max(y.size - 1, 0), float(dx)))


@overload(scipy.integrate.simpson, jit_options={"nopython": True, "nogil": True})
def simpson(y: tp.Any, x: tp.Any = None, dx: tp.Any = 1.0, axis: tp.Any = -1) -> tp.Callable:
    r"""Numba overload of :external:py:func:`scipy.integrate.simpson` for 1D arrays.

    See :py:func:`simpson_h` for the integration rule.
    Unlike in SciPy, ``dx`` and ``axis`` can also be given as positional arguments.

    :param y: values of the integrand, a 1D array
    :param x: sample points. If not given, the points are spaced uniformly by ``dx``.
    :param dx: spacing of the sample points, if ``x`` is not given
    :param axis: axis of integration, which must be -1 or 0 for a 1D array
    :return: the implementation
    """
    if not isinstance(y, numba.types.Array) or y.ndim != 1:
        raise numba.core.errors.TypingError("Only 1D arrays are supported for y.")
    if x is None or isinstance(x, (numba.types.NoneType, numba.types.Omitted)):
        return simpson_dx
    if not isinstance(x, numba.types.Array) or x.ndim != 1:
        raise numba.core.errors.TypingError("Only 1D arrays are supported for x.")
    return simpson_x
