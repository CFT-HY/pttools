# import ctypes as ct
# import glob
# import os
import typing as tp

# import numba
# from numba.extending import overload
import numpy as np
# import scipy.interpolate

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
    Modified from scipy.interpolate.splev

    :param x: 1D array
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


# @numba.njit
def fitpack_spl_(x: np.ndarray, nu: int, t: np.ndarray, c: np.ndarray, k: int, e: int):
    """
    https://github.com/scipy/scipy/blob/main/scipy/interpolate/src/_fitpackmodule.c
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


def spline():
    # from tests import utils

    x = np.linspace(0, 2*np.pi, 20)
    x2 = np.linspace(0, 2*np.pi, 40)
    y = np.sin(x)
    spl = scipy.interpolate.splrep(x, y, s=0)
    ref = scipy.interpolate.splev(x2, spl)
    data = splev(x2, spl)

    import matplotlib.pyplot as plt
    plt.plot(x2, data, label="data")
    plt.plot(x2, ref, label="ref", ls=":")
    plt.legend()
    plt.show()

    # utils.assert_allclose(data, ref)


if __name__ == "__main__":
    spline()

# TODO
# - redo splev()
