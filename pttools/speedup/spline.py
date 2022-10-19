"""
Spline interpolation utilities

These are implemented manually, as SciPy libraries don't expose the interfaces of the Fortran functions.
Only their wrappers written in C are exposed, but those expect Python objects and therefore aren't callable
from Numba without the use of object mode.
"""

# import ctypes as ct
# import glob
# import os
import typing as tp

import numba
from numba.extending import overload
import numpy as np
import scipy.interpolate

from pttools.speedup import fitpack

# interpolate_dir = os.path.dirname(os.path.abspath(scipy.interpolate.fitpack.__file__))
# fitpack_files = glob.glob(os.path.join(interpolate_dir, "_fitpack.*.so"))
# if len(fitpack_files) < 1:
#     raise FileNotFoundError("Fitpack was not found")
# fitpack = ct.CDLL(os.path.join(interpolate_dir, fitpack_files[0]))

# f_splev = fitpack.splev_
# f_splev.argtypes = [
#     ct.POINTER(ct.c_double),
#     ct.POINTER(ct.c_int),
#     ct.POINTER(ct.c_double),
#     ct.POINTER(ct.c_int),
#     ct.POINTER(ct.c_int),
#     ct.POINTER(ct.c_double),
#     ct.POINTER(ct.c_double),
#     ct.POINTER(ct.c_int),
#     ct.POINTER(ct.c_int),
#     ct.POINTER(ct.c_double),
#     ct.POINTER(ct.c_int)
# ]
# f_splev.restype = None
#
# f_splder = fitpack.splder_
# f_splder.argtypes = [
#     ct.POINTER(ct.c_double),
#     ct.POINTER(ct.c_int),
#     ct.POINTER(ct.c_double),
#     ct.POINTER(ct.c_int),
#     ct.POINTER(ct.c_double),
#     ct.POINTER(ct.c_double),
#     ct.POINTER(ct.c_int),
#     ct.POINTER(ct.c_int),
#     ct.POINTER(ct.c_int)
# ]
# f_splder.restype = None


# @overload(scipy.interpolate.splev)
def splev(x: np.ndarray, tck: tp.Tuple[np.ndarray, np.ndarray, int], der: int = 0, ext: int = 0):
    """
    Modified from :external:py:func:`scipy.interpolate.splev`.
    See the SciPy documentation for details.

    :param x: 1D array
    :param tck: Tuple of spline parameters as given by scipy.interpolate.splrep()
    :param der: order of derivative to be computed
    :param ext: Extrapolation: 0 = extrapolate, 1 = return 0, 2 = raise ValueError, 3 = return the boundary value
    """
    t, c, k = tck

    if c.ndim > 1:
        raise ValueError("Parametric interpolation is not supported on Numba")
    # try:
    #     c[0][0]
    #     parametric = True
    # except Exception:
    #     parametric = False
    # if parametric:
    #     return list(map(lambda c, x=x, t=t, k=k, der=der:
    #                     splev(x, [t, c, k], der, ext), c))
    # else:
    if not (0 <= der <= k):
        raise ValueError("0<=der=%d<=k=%d must hold" % (der, k))
    if ext not in (0, 1, 2, 3):
        raise ValueError("ext = %s not in (0, 1, 2, 3) " % ext)

    # x = asarray(x)
    shape = x.shape
    # x = atleast_1d(x).ravel()

    y, ier = fitpack_spl_(x, der, t, c, k, ext)

    if ier == 10:
        raise ValueError("Invalid input data")
    if ier == 1:
        raise ValueError("Found x value not in the domain")
    if ier:
        raise TypeError("An error occurred")

    return y.reshape(shape)


@numba.njit
def splev_linear_core(xp: float, t: np.ndarray, c: np.ndarray, ext: int) -> float:
    if xp < t[0]:
        if ext == 0:
            a = (c[1] - c[0]) / (t[2] - t[1])
            return c[0] + a * (xp - t[0])
        if ext == 1:
            return 0
        if ext == 2:
            raise ValueError("Extrapolating is disabled")
        if ext == 3:
            return c[0]
        raise ValueError("Invalid ext")
    for j in range(t.size - 2):
        if xp < t[j + 2]:
            a = (c[j + 1] - c[j]) / (t[j + 2] - t[j + 1])
            return c[j] + a * (xp - t[j + 1])
    # If the upper boundary is exceeded
    else:
        if ext == 0:
            a = (c[-3] - c[-4]) / (t[-2] - t[-3])
            return c[-3] + a * (xp - t[-2])
        if ext == 1:
            return 0
        if ext == 2:
            raise ValueError("Extrapolating is disabled")
        if ext == 3:
            return c[-3]
        raise ValueError("Invalid ext")


@numba.njit
def splev_linear_validate(k: int, der: int) -> None:
    if k != 1:
        print("Got k = ", k)
        raise NotImplementedError("Only linear interpolation is implemented at the moment")
    if der != 0:
        raise NotImplementedError("Derivatives are not yet implemented")


def splev_linear_arr(x, tck: tp.Tuple[np.ndarray, np.ndarray, int], der: int = 0, ext: int = 0):
    t, c, k = tck
    splev_linear_validate(k, der)

    y = np.empty_like(x)
    for i, xp in enumerate(x):
        y[i] = splev_linear_core(xp, t, c, ext)

    return y


def splev_linear_scalar(x, tck: tp.Tuple[np.ndarray, np.ndarray, int], der: int = 0, ext: int = 0):
    t, c, k = tck
    splev_linear_validate(k, der)
    return splev_linear_core(x, t, c, ext)


@overload(scipy.interpolate.splev)
def splev_linear(x, tck: tp.Tuple[np.ndarray, np.ndarray, int], der: int = 0, ext: int = 0):
    """
    :param x: float or 1D array
    :param tck: Tuple of spline parameters as given by scipy.interpolate.splrep()
    :param der: order of derivative to be computed
    :param ext: Extrapolation: 0 = extrapolate, 1 = return 0, 2 = raise ValueError, 3 = return the boundary value
    """
    if isinstance(x, numba.types.Float):
        return splev_linear_scalar
    return splev_linear_arr


# @numba.njit
def fitpack_spl_(x: np.ndarray, nu: int, t: np.ndarray, c: np.ndarray, k: int, e: int):
    """
    Numba implementation of the
    `SciPy C wrapper for spline interpolation <https://github.com/scipy/scipy/blob/main/scipy/interpolate/src/_fitpackmodule.c>`_.

    :param x: points to interpolate at
    :param t: knots
    :param nu: order of the derivative to be taken
    :param c: B-spline coefficients
    :param k: degree of the spline
    :param e: whether to extend the spline beyond the knot data
    """
    # ier = ct.c_int()
    #
    # m = ct.c_int(x.shape[0])
    # n = ct.c_int(t.shape[0])
    m = x.shape[0]
    n = t.shape[0]
    wrk = np.zeros((n,))

    # y = np.empty((1, m))
    y = np.zeros((m,))

    # c_e = ct.c_int(e)
    # c_k = ct.c_int(k)
    # c_nu = ct.c_int(nu)

    if nu:
        # f_splder(
        #     t.ctypes.data, ct.byref(n), c.ctypes.data, ct.byref(c_k), ct.byref(c_nu),
        #     x.ctypes.data, y.ctypes.data, ct.byref(m), ct.byref(c_e), wrk.ctypes.data, ct.byref(ier))
        ier = fitpack.splder(t, n, c, k, nu, x, y, m, e, wrk)
    else:
        # f_splev(
        #     t.ctypes.data, ct.byref(n), c.ctypes.data, ct.byref(c_k),
        #     x.ctypes.data, y.ctypes.data, ct.byref(m), ct.byref(c_e), ct.byref(ier))
        ier = fitpack.splev(t, n, c, k, x, y, m, e)

    return y, ier
