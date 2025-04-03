"""Numerical utilities for SSMtools."""

import logging
import typing as tp

import numba
from numba.extending import overload
import numba.types
import numpy as np

import pttools.type_hints as th
from . import const

logger = logging.getLogger(__name__)


@numba.njit
def envelope(xi: np.ndarray, f: np.ndarray, v_wall: float = None, v_sh: float = None) -> np.ndarray:
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


@numba.njit
def resample_uniform_xi(
        xi: np.ndarray,
        f: th.FloatOrArr,
        n_xi: int = const.NPTDEFAULT[0]) -> tp.Tuple[np.ndarray, th.FloatOrArr]:
    r"""
    Provide uniform resample of function defined by $(x,y) = (\xi,f)$.
    Returns f interpolated and the uniform grid of n_xi points in range [0,1].

    :param xi: $\xi$
    :param f: function values $f$ at the points $\xi$
    :param n_xi: number of interpolated points
    """
    xi_re = np.linspace(0, 1-1/n_xi, n_xi)
    return xi_re, np.interp(xi_re, xi, f)


def _sin_transform_scalar(
        z: th.FloatOrArr,
        xi: np.ndarray,
        f: np.ndarray,
        z_st_thresh: float = const.Z_ST_THRESH,
        v_wall: float = None,
        v_sh: float = None) -> th.FloatOrArrNumba:
    if z <= z_st_thresh:
        array = f * np.sin(z * xi)
        integral = np.trapezoid(array, xi)
    else:
        integral = sin_transform_approx(z, xi, f, v_wall=v_wall, v_sh=v_sh)
    return integral


def _sin_transform_arr(
        z: th.FloatOrArr,
        xi: np.ndarray,
        f: np.ndarray,
        z_st_thresh: float = const.Z_ST_THRESH,
        v_wall: float = None,
        v_sh: float = None) -> th.FloatOrArrNumba:
    lo = np.where(z <= z_st_thresh)
    z_lo = z[lo]
    # Integrand of the sine transform
    # This computation is O(len(z_lo) * len(xi)) = O(n^2)
    # array_lo = f * np.sin(np.outer(z_lo, xi))
    # For each z, integrate f * sin(z*xi) over xi
    # integral: np.ndarray = np.trapezoid(array_lo, xi)
    integral = sin_transform_core(xi, f, z_lo)

    if len(lo) < len(z):
        z_hi = z[np.where(z > z_st_thresh - const.DZ_ST_BLEND)]
        I_hi = sin_transform_approx(z_hi, xi, f, v_wall=v_wall, v_sh=v_sh)

        if len(z_hi) + len(z_lo) > len(z):
            # If there are elements in the z blend range, then blend
            hi_blend = np.where(z_hi <= z_st_thresh)
            z_hi_blend = z_hi[hi_blend]
            lo_blend = np.where(z_lo > z_st_thresh - const.DZ_ST_BLEND)
            z_blend_max = np.max(z_hi_blend)
            z_blend_min = np.min(z_hi_blend)
            if z_blend_max > z_blend_min:
                s = (z_hi_blend - z_blend_min) / (z_blend_max - z_blend_min)
            else:
                s = 0.5 * np.ones_like(z_hi_blend)
            frac = 3 * s ** 2 - 2 * s ** 3
            integral[lo_blend] = I_hi[hi_blend] * frac + integral[lo_blend] * (1 - frac)

        integral = np.concatenate((integral[lo], I_hi[z_hi > z_st_thresh]))

    # if len(integral) != len(z):
    #     raise RuntimeError

    return integral


def sin_transform(
        z: th.FloatOrArr,
        xi: np.ndarray,
        f: np.ndarray,
        z_st_thresh: float = const.Z_ST_THRESH,
        v_wall: float = None,
        v_sh: float = None) -> th.FloatOrArrNumba:
    r"""
    sin transform of $f(\xi)$, Fourier transform variable z.
    For z > z_st_thresh, use approximation rather than doing the integral.
    Interpolate between  z_st_thresh - dz_blend < z < z_st_thresh.

    Without the approximations this function would compute
    $$\hat{f}(z) =  f(\xi) \int_{{\xi}_\text{min}}^{{\xi}_\text{max}} \sin(z \xi) d\xi$$.

    Used in :gw_pt_ssm:`\ ` eq. 4.5, 4.8

    :param z: Fourier transform variable (any shape)
    :param xi: $\xi$ points over which to integrate
    :param f: function values at the points $\xi$, same shape as $\xi$
    :param z_st_thresh: for $z$ values above z_sh_tresh, use approximation rather than doing the integral.
    :param v_wall: wall speed
    :param v_sh: shock speed
    :return: sine transformed values $\hat{f}(z)$
    """
    if isinstance(z, float):
        return _sin_transform_scalar(z, xi, f, z_st_thresh, v_wall=v_wall, v_sh=v_sh)
    if isinstance(z, np.ndarray):
        return _sin_transform_arr(z, xi, f, z_st_thresh, v_wall=v_wall, v_sh=v_sh)
    raise NotImplementedError


@overload(sin_transform, jit_options={"parallel": True})
def _sin_transform_numba(
        z: th.FloatOrArr,
        xi: np.ndarray,
        f: np.ndarray,
        z_st_thresh: float = const.Z_ST_THRESH,
        v_wall: float = None,
        v_sh: float = None) -> th.FloatOrArrNumba:
    if isinstance(z, numba.types.Float):
        return _sin_transform_scalar
    if isinstance(z, numba.types.Array):
        return _sin_transform_arr
    raise NotImplementedError


@numba.njit(parallel=True)
def sin_transform_core(t: np.ndarray, f: np.ndarray, freq: np.ndarray) -> np.ndarray:
    r"""
    The `sine transform <https://en.wikipedia.org/wiki/Sine_and_cosine_transforms>`_
    for multiple values of $\omega$ without any approximations.
    Computes the following for each angular frequency $\omega$.
    $$\hat{f}(\omega) = \int_{{t}_\text{min}}^{{t}_\text{max}} f(t) \sin(\omega t) dt$$

    :param t: variable of the real space ($t$ or $x$)
    :param f: function values at the points $t$
    :param freq: frequencies $\omega$
    :return: value of the sine transformed function at each angular frequency $\omega$
    """
    integral = np.zeros_like(freq)
    # pylint: disable=not-an-iterable
    for i in numba.prange(freq.size):
        integrand = f * np.sin(freq[i] * t)
        # If you get Numba errors here, ensure that t is contiguous.
        # This can be achieved with the use of t.copy() in the data pipeline leading to this function.
        integral[i] = np.trapezoid(integrand, t)
    return integral


@numba.njit
def sin_transform_approx(
        z: th.FloatOrArr, xi: np.ndarray, f: np.ndarray,
        v_wall: float = None, v_sh: float = None) -> np.ndarray:
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


def sin_transform_old(z: th.FloatOrArr, xi: np.ndarray, v: np.ndarray) -> th.FloatOrArr:
    r"""
    Old sin transform of $v(\xi)$

    .. deprecated:: 0.0.1

    :param z: Fourier transform variable (any shape)
    :param xi: $\xi$
    :param v: wall speed $v$, same shape as $\xi$
    """
    logger.warning("sin_transform_old is deprecated")
    if isinstance(z, np.ndarray):
        array = np.sin(np.outer(z, xi)) * v
        integral = np.trapezoid(array, xi)
    else:
        array = v * np.sin(z * xi)
        integral = np.trapezoid(array, xi)

    return integral
