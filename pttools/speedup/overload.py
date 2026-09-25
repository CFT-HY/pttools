"""Additional definitions for Numba-jitting functions from other libraries.

Numba requires that the parameters of an overload typing function and of the implementation it returns
are identical, including their type annotations.
Therefore, the parameters that receive Numba types in the typing functions
and values in the implementations are annotated with :data:`typing.Any`.
"""

import logging
import typing as tp

import numba
from numba.extending import overload
import numpy as np

from pttools.speedup import numba_wrapper

logger: logging.Logger = logging.getLogger(__name__)


def do_nothing(x: tp.Any) -> tp.Any:
    """Do nothing."""
    return x


def is_nonzero(x: tp.Any) -> bool:
    """Truth value of a scalar number, as given by :external:py:func:`numpy.all` and :external:py:func:`numpy.any`."""
    return x != 0


if numba_wrapper.NUMBA_VERSION < (0, 49, 0):
    logger.warning("Overloading numpy.flipud for old Numba")

    @overload(np.flipud, jit_options={"nopython": True})
    def np_flip_ud(arr: np.ndarray) -> tp.Callable[[np.ndarray], np.ndarray]:
        def impl(arr: np.ndarray) -> np.ndarray:
            # Copying may be necessary to avoid problems with the memory layout of the array
            # return arr[::-1, ...].copy()
            return arr[::-1, ...]
        return impl


@overload(np.all, jit_options={"nopython": True})
def np_all(x: tp.Any) -> tp.Callable | None:
    """Overload of :external:py:func:`numpy.all` for booleans and scalars.

    This seems not to be used properly in Numba 0.60.0.
    For other types, Numba's own implementation is used.
    """
    if isinstance(x, numba.types.Boolean):
        return do_nothing
    if isinstance(x, numba.types.Number):
        return is_nonzero
    return None


def np_all_fix(x: tp.Any) -> np.bool:
    """A fix for overloading :external:py:func:`numpy.all`."""
    return np.all(x)


@overload(np_all_fix, jit_options={"nopython": True})
def np_all_fix_scalar(x: tp.Any) -> tp.Callable:
    """Overload of :external:py:func:`numpy.all` for booleans and scalars."""
    if isinstance(x, numba.types.Boolean):
        return do_nothing
    if isinstance(x, numba.types.Number):
        return is_nonzero
    return np_all_fix


@overload(np.any, jit_options={"nopython": True})
def np_any(x: tp.Any) -> tp.Callable | None:
    """Overload of :external:py:func:`numpy.any` for booleans and scalars.

    For other types, Numba's own implementation is used.
    """
    if isinstance(x, numba.types.Boolean):
        return do_nothing
    if isinstance(x, numba.types.Number):
        return is_nonzero
    return None


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
