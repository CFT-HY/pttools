"""Approximate sine transform for the Sound Shell Model"""

import logging

import numba
import numpy as np

import pttools.type_hints as th

logger = logging.getLogger(__name__)


@numba.njit(cache=True)
def envelope(
        xi: th.FloatArr1D,
        f: th.FloatArr1D,
        v_wall: float | None = None,
        v_sh: float | None = None) -> th.FloatArr1D:
    r"""
    Helper function for :func:`sin_transform_approx`.
    Assumes that

    - $\max(v)$ is achieved at a discontinuity (bubble wall)
    - $f(\xi)$ finishes at a discontinuity (shock)
    - at least the first element of $f$ is zero

    xi1: last zero value of f,
    xi_w: position of maximum f (wall)
    x12: last non-zero value of f (def) or 1st zero after wall (det)
    f1: value just before xi1
    f_m: value just before wall
    f_p: value just after wall
    f2: (at shock, or after wall)

    :param: xi: $\xi$
    :param f: function values $f$ at the points $\xi$
    :param v_wall: wall speed
    :param v_sh: shock speed
    :return: array of $\xi$, $f$ pairs "outlining" function $f$
    """
    # if v_wall is None or v_sh is None:
    #     with numba.objmode:
    #         logger.warning(
    #             "Please give v_wall and v_sh to envelope(). "
    #             "They will be needed in the future for finding the discontinuities."
    #         )

    xi_nonzero = xi[np.nonzero(f)]
    xi1 = np.min(xi_nonzero)
    xi2 = np.max(xi_nonzero)
    ind1 = np.where(xi == xi1)[0][0]  # where returns tuple, first element array
    ind2 = np.where(xi == xi2)[0][0]
    f1 = f[ind1 - 1]  # in practice, f1 is always zero, or very close, so could drop.
    xi1 = xi[ind1 - 1]  # line up f1 and xi1

    i_max_f = np.argmax(f)
    f_max = f[i_max_f]
    xi_w = xi[i_max_f]  # max f always at wall

    # This indexing fix has changed test_pow_specs.py results a bit
    if i_max_f + 1 == f.shape[0]:
        df_at_max = f[i_max_f] - f[i_max_f - 2]
        with numba.objmode:
            logger.warning("i_max_f is at the end of f. df_at_max will be calculated for the previous values.")
    else:
        df_at_max = f[i_max_f + 1] - f[i_max_f - 1]

    # print(ind1, ind2, [xi1,f1], [xi_w, f_max])

    if df_at_max > 0:
        # Deflagration or hybrid, ending in shock.
        f_m = f[i_max_f - 1]
        f_p = f_max
        f2 = f[ind2]
    else:
        # Detonation, nothing beyond wall
        f_m = f_max
        f_p = 0
        f2 = 0

    return np.array([
        [xi1, xi_w, xi_w, xi2],
        [f1, f_m, f_p, f2]
    ])


@numba.njit(cache=True)
def sin_transform_approx(
        z: th.FloatOrArr,
        xi: th.FloatArr1D,
        f: th.FloatArr1D,
        v_wall: float | None = None,
        v_sh: float | None = None) -> th.FloatArr1D:
    r"""
    Approximate sin transform of $f(\xi)$.
    For values $f_a$ and $f_b$, we have
    $$
    \int_{\xi_a}^{\xi_b} d\xi f(\xi) \sin(z \xi) \to
    - \frac{1}{z} \left(f_b \cos(z \xi_b) - f_a \cos(z \xi_a)\right) + O(1/z^2)
    $$
    as $z \to \infty$.
    Function assumed piecewise continuous in intervals $[\xi_1, \xi_w]$ and
    $[\xi_w,\xi_2]$.

    :param z: Fourier transform variable (any shape)
    :param xi: $\xi$
    :param f: function values at the points $\xi$, same shape as $\xi$
    """
    # Old versions of Numba don't support unpacking 2D arrays
    # [[xi1, xi_w, _, xi2], [f1, f_m, f_p, f2]] = envelope(xi, f)
    envelope_arr = envelope(xi, f, v_wall=v_wall, v_sh=v_sh)
    [xi1, xi_w, _, xi2] = envelope_arr[0, :]
    [f1, f_m, f_p, f2] = envelope_arr[1, :]

    integral = -(f2 * np.cos(z * xi2) - f_p * np.cos(z * xi_w)) / z
    integral += -(f_m * np.cos(z * xi_w) - f1 * np.cos(z * xi1)) / z
    return integral
