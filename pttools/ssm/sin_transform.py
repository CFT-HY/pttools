"""Sine transform for the Sound Shell Model"""

import numba
from numba.extending import overload
import numba.types
import numpy as np

from pttools.ssm import const
from pttools.ssm.sin_transform_approx import sin_transform_approx
import pttools.type_hints as th


def _sin_transform_scalar(
        z: th.FloatOrArr,
        xi: th.FloatArr1D,
        f: th.FloatArr1D,
        z_st_thresh: float = const.Z_ST_THRESH,
        v_wall: float | None = None,
        v_sh: float | None = None,
        parallel: bool = True) -> th.FloatOrArr:
    if z <= z_st_thresh:
        array = f * np.sin(z * xi)
        integral = np.trapezoid(array, xi)
    else:
        integral = sin_transform_approx(z=z, xi=xi, f=f, v_wall=v_wall, v_sh=v_sh)
    return integral


def _sin_transform_arr(
        z: th.FloatOrArr,
        xi: th.FloatArr1D,
        f: th.FloatArr1D,
        z_st_thresh: float = const.Z_ST_THRESH,
        v_wall: float | None = None,
        v_sh: float | None = None,
        parallel: bool = True) -> th.FloatArr1D:
    lo = np.where(z <= z_st_thresh)
    z_lo = z[lo]
    # Integrand of the sine transform
    # This computation is O(len(z_lo) * len(xi)) = O(n^2)
    # array_lo = f * np.sin(np.outer(z_lo, xi))
    # For each z, integrate f * sin(z*xi) over xi
    # integral = np.trapezoid(array_lo, xi)
    integral = sin_transform_core(t=xi, f=f, freq=z_lo) if parallel else sin_transform_core_single(t=xi, f=f, freq=z_lo)

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
        xi: th.FloatArr1D,
        f: th.FloatArr1D,
        z_st_thresh: float = const.Z_ST_THRESH,
        v_wall: float | None = None,
        v_sh: float | None = None,
        parallel: bool = True) -> th.FloatOrArr:
    r"""
    Sine transform $\hat{f}(z)$ of $f(\xi)$

    For z > z_st_thresh, use approximation rather than doing the integral.
    Interpolate between  z_st_thresh - dz_blend < z < z_st_thresh.

    Without the approximations this function would compute
    $$\hat{f}(z) =  f(\xi) \int_{{\xi}_\text{min}}^{{\xi}_\text{max}} \sin(z \xi) d\xi$$.

    Used in :gw_pt_ssm:`\ ` eq. 4.5, 4.8

    Todo: It may be computationally more efficient to re-sample the points to a uniform grid and then use FFT.

    :param z: Fourier transform variable (any shape)
    :param xi: $\xi$ points over which to integrate
    :param f: function values at the points $\xi$, same shape as $\xi$
    :param z_st_thresh: for $z$ values above z_sh_tresh, use approximation rather than doing the integral.
    :param v_wall: wall speed
    :param v_sh: shock speed
    :return: sine transformed values $\hat{f}(z)$ (same size as $z$)
    """
    if isinstance(z, float):
        return _sin_transform_scalar(z=z, xi=xi, f=f, z_st_thresh=z_st_thresh, v_wall=v_wall, v_sh=v_sh)
    if isinstance(z, np.ndarray):
        return _sin_transform_arr(z=z, xi=xi, f=f, z_st_thresh=z_st_thresh, v_wall=v_wall, v_sh=v_sh, parallel=parallel)
    raise NotImplementedError


@overload(sin_transform, jit_options={"parallel": True, "nogil": True})
def _sin_transform_numba(
        z: th.FloatOrArr,
        xi: th.FloatArr1D,
        f: th.FloatArr1D,
        z_st_thresh: float = const.Z_ST_THRESH,
        v_wall: float | None = None,
        v_sh: float | None = None,
        parallel: bool = True) -> th.NumbaFunc:
    if isinstance(z, numba.types.Float):
        return _sin_transform_scalar
    if isinstance(z, numba.types.Array):
        return _sin_transform_arr
    raise NotImplementedError


def _sin_transform_core(t: th.FloatArr1D, f: th.FloatArr1D, freq: th.FloatArr1D) -> th.FloatArr1D:
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


sin_transform_core = numba.njit(parallel=True, nogil=True, cache=True)(_sin_transform_core)
sin_transform_core_single = numba.njit(nogil=True, cache=True)(_sin_transform_core)
