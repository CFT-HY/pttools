"""Additional definitions for Numba-jitting functions from other libraries"""

import logging

import numba
from numba.extending import overload
import numpy as np

from . import numba_wrapper

logger = logging.getLogger(__name__)


def do_nothing(x):
    return x


if numba_wrapper.NUMBA_VERSION < (0, 49, 0):
    logger.warning("Overloading numpy.flipud for old Numba")

    @overload(np.flipud, jit_options={"nopython": True})
    def np_flip_ud(arr: np.ndarray):
        def impl(arr: np.ndarray) -> np.ndarray:
            # Copying may be necessary to avoid problems with the memory layout of the array
            # return arr[::-1, ...].copy()
            return arr[::-1, ...]
        return impl


@overload(np.all, jit_options={"nopython": True})
def np_all(x):
    """Overload of :external:py:func:`numpy.all` for booleans

    This seems not to be used properly in Numba 0.60.0.
    """
    if isinstance(x, numba.types.Boolean):
        return do_nothing
    if isinstance(x, numba.types.Number):
        return bool


def np_all_fix(x):
    return np.all(x)


@overload(np_all_fix, jit_options={"nopython": True})
def np_all_fix_scalar(x):
    if isinstance(x, numba.types.Boolean):
        return do_nothing
    if isinstance(x, numba.types.Number):
        return bool
    return np_all_fix


@overload(np.any, jit_options={"nopython": True})
def np_any(x):
    """Overload of :external:py:func:`numpy.any` for booleans and scalars"""
    if isinstance(x, numba.types.Boolean):
        return do_nothing
    if isinstance(x, numba.types.Number):
        return bool


# @overload(np.asanyarray, jit_options={"nopython": True})
# def asanyarray(arr: np.ndarray):
#     if isinstance(arr, numba.types.Array):
#         def func(arr: np.ndarray):
#             return arr
#         return func
#     raise NotImplementedError
#
#
# @overload(np.ndim, jit_options={"nopython": True})
# def ndim(val):
#     if isinstance(val, numba.types.Number):
#         def func(val):
#             return 0
#         return func
#     if isinstance(val, numba.types.Array):
#         def func(val):
#             return val.ndim
#         return func
#     raise NotImplementedError
