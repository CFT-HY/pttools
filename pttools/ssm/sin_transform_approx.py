"""Approximate sine transform for the Sound Shell Model."""

import logging

import numba
import numpy as np

from pttools.speedup import njit
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)


@njit(cache=True, nogil=True)
def envelope(
        xi: th.FloatArr1D,
        f: th.FloatArr1D,
        v_wall: float | None = None,
        v_sh: float | None = None) -> th.FloatArr1D:
    r"""
    Helper function for :func:`sin_transform_approx`.

    Assumes that:

    - $f(\xi)$ has a discontinuity at the wall
    - $f(\xi)$ finishes at a discontinuity (shock for deflagrations and hybrids, wall for detonations)
    - at least the first element of $f$ is zero

    xi1: last zero value of f,
    xi_w: wall
    xi2: shock (def, hyb) or wall (det)
    f1: value just before xi1
    f_m: value just before wall
    f_p: value just after wall
    f2: value just before shock (def, hyb), or zero (det)

    If $v_\text{wall}$ and $v_\text{sh}$ are given, the discontinuities are located with them
    (see :func:`_envelope_wall_shock`).
    Otherwise, the wall is assumed to be at $\max(f)$ (see :func:`_envelope_max`).
    This fails, if the wall and the shock are not resolved by the points $\xi$,
    or if $f$ is not at its maximum just behind or just ahead of the wall,
    e.g. for $\lambda$ of a hybrid with $v_\text{wall}$ close to $v_\text{CJ}$,
    for which the thin shell between the wall and the shock has the largest values of $f$.

    :param xi: $\xi$
    :param f: function values $f$ at the points $\xi$
    :param v_wall: wall speed
    :param v_sh: shock speed, which should be equal to $v_\text{wall}$ for detonations
    :return: array of $\xi$, $f$ pairs "outlining" function $f$
    """
    xi_nonzero = xi[np.nonzero(f)]
    xi1 = np.min(xi_nonzero)
    ind1 = np.where(xi == xi1)[0][0]  # where returns tuple, first element array
    f1 = f[ind1 - 1]  # in practice, f1 is always zero, or very close, so could drop.
    xi1 = xi[ind1 - 1]  # line up f1 and xi1

    # The conditions are nested so that Numba can prune the branches for None.
    if v_wall is not None:  # noqa: SIM102
        if v_sh is not None:  # noqa: SIM102
            if np.isfinite(v_wall) and np.isfinite(v_sh):
                return _envelope_wall_shock(xi, f, xi1=xi1, f1=f1, v_wall=v_wall, v_sh=v_sh)
    return _envelope_max(xi, f, xi1=xi1, f1=f1)


@njit(cache=True, nogil=True)
def _envelope_max(xi: th.FloatArr1D, f: th.FloatArr1D, xi1: float, f1: float) -> th.FloatArr1D:
    """Envelope of $f$ assuming that the wall is at the maximum of $f$, see :func:`envelope`."""
    xi2 = np.max(xi[np.nonzero(f)])
    ind2 = np.where(xi == xi2)[0][0]

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


@njit(cache=True, nogil=True)
def _envelope_wall_shock(
        xi: th.FloatArr1D,
        f: th.FloatArr1D,
        xi1: float,
        f1: float,
        v_wall: float,
        v_sh: float) -> th.FloatArr1D:
    r"""Envelope of $f$ with the discontinuities at $v_\text{wall}$ and $v_\text{sh}$, see :func:`envelope`.

    The values next to the discontinuities are taken from the points closest to them,
    following the conventions of the fluid shells of :py:class:`pttools.bubble.bubble.Bubble`:
    a detonation has one point at the wall with the value behind the wall,
    a hybrid has two points at the wall with the values behind and ahead of the wall,
    and a subsonic deflagration has one point at the wall with the value ahead of the wall.
    The value just behind the shock is that of the last point within $\xi \leq v_\text{sh}$.
    If there are no points between the wall and the shock, the shell is assumed to be infinitely thin.
    """
    i_behind = -1
    i_ahead = -1
    i_at_first = -1
    i_at_last = -1
    i_sh = -1
    for i in range(xi.size):
        if xi[i] < v_wall:
            i_behind = i
        elif xi[i] == v_wall:
            if i_at_first < 0:
                i_at_first = i
            i_at_last = i
        else:
            if i_ahead < 0:
                i_ahead = i
            if xi[i] <= v_sh:
                i_sh = i

    if v_sh <= v_wall:
        # Detonation, nothing beyond wall
        f_m = f[i_at_first] if i_at_first >= 0 else f[i_behind]
        return np.array([
            [xi1, v_wall, v_wall, v_wall],
            [f1, f_m, 0., 0.]
        ])

    # Deflagration or hybrid, ending in shock
    if i_at_last > i_at_first:
        f_m = f[i_at_first]
        f_p = f[i_at_last]
    elif i_at_first >= 0:
        f_m = f[i_behind]
        f_p = f[i_at_first]
    else:
        f_m = f[i_behind]
        f_p = f[i_ahead] if i_sh >= 0 else f_m
    # The last point at the wall is ahead of the wall, and the shell may have no points beyond it.
    f2 = f[i_sh] if i_sh >= 0 else f_p
    return np.array([
        [xi1, v_wall, v_wall, v_sh],
        [f1, f_m, f_p, f2]
    ])


@njit(cache=True, nogil=True)
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
