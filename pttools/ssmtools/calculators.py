"""Numerical utilities for ssmtools"""

import typing as tp

import numba
import numba.types
import numpy as np

import pttools.type_hints as th
from . import const


@numba.njit
def sin_transform_old(z: th.FLOAT_OR_ARR, xi: np.ndarray, v: np.ndarray):
    """
    sin transform of v(xi), Fourier transform variable z.
    xi and v arrays of the same shape, z can be an array of a different shape.
    """

    if isinstance(z, np.ndarray):
        array = np.sin(np.outer(z, xi)) * v
        integral = np.trapz(array, xi)
    else:
        array = v * np.sin(z * xi)
        integral = np.trapz(array, xi)

    return integral


@numba.njit
def envelope(xi: np.ndarray, f: np.ndarray) -> tp.Tuple[tp.List[float], tp.List[float]]:
    """
    Returns lists of xi, f pairs "outlining" function f.
    Helper function for sin_transform_approx.
    Assumes that
    - max(v) is acheived at a discontinuity (bubble wall)
    - f(xi) finishes at a discontinuity (shock)
    - at least the first element of f is zero

    xi1:  last zero value of f,             f1, value just before xi1
    xi_w: positiom of maximum f (wall)      f_m, value just before wall
                                            f_p, value just after wall
    x12:  last non-zero value of f (def)    f2 (at shock, or after wall)
          or 1st zero after wall (det)
    """

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

    df_at_max = f[i_max_f + 1] - f[i_max_f - 1]

    #    print(ind1, ind2, [xi1,f1], [xi_w, f_max])

    if df_at_max > 0:
        # deflagration or hybrid, ending in shock.
        f_m = f[i_max_f - 1]
        f_p = f_max
        f2 = f[ind2]

    else:
        # detonation, nothing beyond wall
        f_m = f_max
        f_p = 0
        f2 = 0

    xi_list = [xi1, xi_w, xi_w, xi2]
    f_list = [f1, f_m, f_p, f2]

    return xi_list, f_list


@numba.njit
def sin_transform_approx(z: th.FLOAT_OR_ARR, xi: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Approximate sin transform of f(xi), Fourier transform variable z.
    xi and f arrays of the same shape, z can be an array of a different shape.
    values $f_a$ and $f_b$, we have
    $$
    \\int_{\\xi_a}^{\\xi_b} d\\xi f(\\xi) \\sin(z \\xi) \\to
    - \\frac{1}{z} \\left(f_b \\cos(z \\xi_b) - f_a \\cos(z \\xi_a)\\right) + O(1/z^2)
    $$
    as $z \\to \\infty$.
    Function assumed piecewise continuous in intervals $[\\xi_1, \\xi_w]$ and
    $[\\xi_w,\\xi_2]$.
    """
    [xi1, xi_w, _, xi2], [f1, f_m, f_p, f2] = envelope(xi, f)
    integral = -(f2 * np.cos(z * xi2) - f_p * np.cos(z * xi_w)) / z
    integral += -(f_m * np.cos(z * xi_w) - f1 * np.cos(z * xi1)) / z
    return integral


@numba.njit
def _sin_transform_scalar(z: float, xi: np.ndarray, f: np.ndarray, z_st_thresh: float = const.Z_ST_THRESH) -> float:
    if z <= z_st_thresh:
        array = f * np.sin(z * xi)
        integral = np.trapz(array, xi)
    else:
        integral = sin_transform_approx(z, xi, f)
    return integral


@numba.njit(parallel=True)
def sin_transform_core(x: np.ndarray, f: np.ndarray, freq: np.ndarray):
    integral = np.zeros_like(freq)
    for i in numba.prange(freq.size):
        integrand = f * np.sin(freq[i] * x)
        integral[i] = np.trapz(integrand, x)
    return integral


@numba.njit(parallel=True)
def _sin_transform_arr(
        z: np.ndarray, xi: np.ndarray, f: np.ndarray, z_st_thresh: float = const.Z_ST_THRESH) -> np.ndarray:
    lo = np.where(z <= z_st_thresh)
    z_lo = z[lo]
    # Integrand of the sine transform
    # This computation is O(len(z_lo) * len(xi)) = O(n^2)
    # array_lo = f * np.sin(np.outer(z_lo, xi))
    # For each z, integrate f * sin(z*xi) over xi
    # integral: np.ndarray = np.trapz(array_lo, xi)
    integral = sin_transform_core(xi, f, z_lo)

    if len(lo) < len(z):
        z_hi = z[np.where(z > z_st_thresh - const.DZ_ST_BLEND)]
        I_hi = sin_transform_approx(z_hi, xi, f)

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


@numba.generated_jit(nopython=True)
def sin_transform(z: th.FLOAT_OR_ARR, xi: np.ndarray, f: np.ndarray, z_st_thresh: float = const.Z_ST_THRESH):
    """
    sin transform of f(xi), Fourier transform variable z.
    xi and f arrays of the same shape, z can be an array of a different shape.
    For z > z_st_thresh, use approximation rather than doing the integral.
    Interpolate between  z_st_thresh - dz_blend < z < z_st_thresh.
    """
    if isinstance(z, numba.types.Float):
        return _sin_transform_scalar
    elif isinstance(z, numba.types.Array):
        return _sin_transform_arr
    else:
        raise NotImplementedError


@numba.njit
def resample_uniform_xi(
        xi: np.ndarray,
        f: th.FLOAT_OR_ARR,
        nxi: int = const.NPTDEFAULT[0]) -> tp.Tuple[np.ndarray, th.FLOAT_OR_ARR]:
    """
    Provide uniform resample of function defined by (x,y) = (xi,f).
    Returns f interpolated and the uniform grid of nxi points in range [0,1]
    """
    xi_re = np.linspace(0, 1-1/nxi, nxi)
    return xi_re, np.interp(xi_re, xi, f)
