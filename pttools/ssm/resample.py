r"""Resampling of the fluid profiles of the Sound Shell Model to uniform grids of $\xi$.

The resampled profiles are used for the sine transforms, see :py:mod:`pttools.ssm.sin_transform`.
"""

import logging

import numpy as np

from pttools.speedup import njit
from pttools.ssm import const
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)


@njit(cache=True, nogil=True)
def resample_merged_uniform_xi(
        xi: th.FloatArr1D,
        f: th.FloatArr1D,
        n_xi: int = const.DEFAULT_N_XI_SSM) -> tuple[th.FloatArr1D, th.FloatArr1D]:
    r"""
    Merge a uniform grid of $\xi$ into the points $(\xi,f)$.

    Unlike :py:func:`resample_uniform_xi`, this keeps the original points.
    Therefore the discontinuities at the wall and at the shock are preserved exactly,
    as are the features that are narrower than the spacing of the uniform grid,
    such as the thin shell between the wall and the shock of a hybrid
    with $v_\text{wall}$ close to the Chapman-Jouguet speed $v_\text{CJ}$.
    The points of the uniform grid of n_xi points in range [0,1] fill the gaps between the original points.
    Their values are interpolated linearly between the neighbouring original points,
    and the points that coincide with original points are dropped.

    :param xi: $\xi$, which should be in ascending order,
        but may have duplicates at discontinuities and small numerical deviations from the ascending order
    :param f: function values $f$ at the points $\xi$
    :param n_xi: number of points in the uniform grid
    :return: merged $\xi$ and $f$
    """
    xi_uni = np.linspace(0, 1-1/n_xi, n_xi)
    xi_re = np.empty(xi.size + n_xi)
    f_re = np.empty(xi.size + n_xi)
    # Index of the next free slot in the output arrays
    i_re = 0
    # Index of the next uniform point that has not yet been added or dropped
    j = 0
    # Walk through the original points in their order, and before each original point xi[i],
    # add the uniform points that are below it. This is a merge of two sorted arrays,
    # but the original points are never reordered or dropped,
    # so the duplicate points of a discontinuity stay in their order (e.g. behind and ahead of the wall).
    for i in range(xi.size):
        while j < n_xi and xi_uni[j] <= xi[i]:
            # A uniform point that coincides with an original point is dropped,
            # as its value would be ambiguous at a discontinuity, and the original point is kept anyway.
            if xi_uni[j] < xi[i]:
                xi_re[i_re] = xi_uni[j]
                if i == 0 or xi[i] <= xi[i-1]:
                    # For i == 0 there is no original point below,
                    # so the value of the first original point is used, i.e. constant extrapolation as in np.interp.
                    # The condition xi[i] <= xi[i-1] only guards against division by zero,
                    # as the uniform points up to xi[i-1] have already been consumed for the previous original point.
                    f_re[i_re] = f[i]
                else:
                    # Linear interpolation between the neighbouring original points xi[i-1] < xi_uni[j] < xi[i].
                    # These are the same values as from np.interp(xi_uni, xi, f) for ascending xi,
                    # but the interval is known from the merge, so that no search is needed,
                    # and a discontinuity represented by duplicate points cannot be mixed up.
                    # The weight is in (0, 1) by construction, and the clipping only guards against rounding.
                    t = min(max((xi_uni[j] - xi[i-1]) / (xi[i] - xi[i-1]), 0.), 1.)
                    f_re[i_re] = f[i-1] + t * (f[i] - f[i-1])
                i_re += 1
            j += 1
        # The original point is copied as is.
        xi_re[i_re] = xi[i]
        f_re[i_re] = f[i]
        i_re += 1
    # Uniform points beyond the last original point get its value, i.e. constant extrapolation at the end,
    # as in np.interp. For fluid shells this does not happen, as their last point is at xi = 1.
    while j < n_xi:
        xi_re[i_re] = xi_uni[j]
        f_re[i_re] = f[-1]
        i_re += 1
        j += 1
    return xi_re[:i_re], f_re[:i_re]


@njit(cache=True, nogil=True)
def resample_uniform_xi(
        xi: th.FloatArr1D,
        f: th.FloatArr1D,
        n_xi: int = const.DEFAULT_N_XI_SSM) -> tuple[th.FloatArr1D, th.FloatArr1D]:
    r"""
    Provide uniform resample of function defined by $(x,y) = (\xi,f)$.

    Returns f interpolated and the uniform grid of n_xi points in range [0,1].

    :param xi: $\xi$
    :param f: function values $f$ at the points $\xi$
    :param n_xi: number of interpolated points
    """
    xi_re = np.linspace(0, 1-1/n_xi, n_xi)
    return xi_re, np.interp(xi_re, xi, f)  # pyrefly: ignore[bad-return]
