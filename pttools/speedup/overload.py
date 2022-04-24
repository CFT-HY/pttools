import logging

import numba
from numba.extending import overload
import numpy as np

from . import numba_wrapper

logger = logging.getLogger(__name__)


if numba_wrapper.NUMBA_VERSION < (0, 49, 0):
    logger.warning("Overloading numpy.flipud for old Numba")

    @overload(np.flipud)
    def np_flip_ud(arr: np.ndarray):
        def impl(arr: np.ndarray) -> np.ndarray:
            # Copying may be necessary to avoid problems with the memory layout of the array
            # return arr[::-1, ...].copy()
            return arr[::-1, ...]
        return impl


@overload(np.any)
def np_any(a):
    """Overload of :external:py:func:`numpy.any` for booleans and scalars."""
    if isinstance(a, numba.types.Boolean):
        def func(a):
            return a

        return func
    if isinstance(a, numba.types.Number):
        def func(a):
            return bool(a)

        return func

# @overload(np.asanyarray)
# def asanyarray(arr: np.ndarray):
#     if isinstance(arr, numba.types.Array):
#         def func(arr: np.ndarray):
#             return arr
#         return func
#     raise NotImplementedError
#
#
# @overload(np.ndim)
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
