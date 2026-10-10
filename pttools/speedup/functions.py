"""Faster or Numba-jitted versions of library functions."""

import numpy as np

# from pttools.speedup.options import NUMBA_NESTED_PARALLELISM
from pttools.speedup.jit import njit
import pttools.type_hints as th
from pttools.type_hints import FloatArr, FloatArr1D


@njit(cache=True)
def gradient[T: FloatArr](f: T) -> T:
    """Numba version of :func:`np.gradient`."""
    if f.ndim > 1:
        raise NotImplementedError

    out = np.empty_like(f)
    out[1:-1] = (f[2:] - f[:-2]) / 2.
    out[0] = f[1] - f[0]
    out[-1] = f[-1] - f[-2]
    return out


# @njit(parallel=options.NUMBA_NESTED_PARALLELISM)
@njit(cache=True)
def logspace(start: float, stop: float, num: int, base: float = 10.) -> th.FloatArr1D:
    """Numba version of :func:`numpy.logspace`."""
    return base ** np.linspace(start, stop, num)


@njit(cache=True)
def resample_log(x: FloatArr1D, nx: int) -> FloatArr1D:
    """Resample a variable over a logarithmic range."""
    return logspace(np.log10(x.min()), np.log10(x.max()), num=nx)


@njit(cache=True)
def simpson(y: FloatArr1D, x: FloatArr1D) -> float:
    r"""Numba version of :external:py:func:`scipy.integrate.simpson` for 1D arrays with sample points.

    The composite Simpson's rule for irregularly spaced data is applied to each pair of intervals
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
    where $h_0$ and $h_1$ are the last two intervals, as in SciPy.

    :param y: values of the integrand
    :param x: sample points
    :return: the integral
    """
    n = y.size
    if n < 2:  # noqa: PLR2004
        return 0.
    if n == 2:  # noqa: PLR2004
        return 0.5 * (x[1] - x[0]) * (y[0] + y[1])
    # The pairs of intervals cover the points 0 ... n-1 for odd n, and 0 ... n-2 for even n.
    stop = n - 2 if n % 2 else n - 3
    result = 0.
    for i in range(0, stop, 2):
        h0 = x[i + 1] - x[i]
        h1 = x[i + 2] - x[i + 1]
        hsum = h0 + h1
        result += hsum / 6. * (
            y[i] * (2. - h1 / h0)
            + y[i + 1] * hsum * hsum / (h0 * h1)
            + y[i + 2] * (2. - h0 / h1)
        )
    if n % 2 == 0:
        h0 = x[-2] - x[-3]
        h1 = x[-1] - x[-2]
        alpha = (2 * h1**2 + 3 * h0 * h1) / (6 * (h1 + h0))
        beta = (h1**2 + 3 * h0 * h1) / (6 * h0)
        eta = h1**3 / (6 * h0 * (h0 + h1))
        result += alpha * y[-1] + beta * y[-2] - eta * y[-3]
    return result
