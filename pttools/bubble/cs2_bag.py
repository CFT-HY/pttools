"""Speed of sound for the Bag Model."""

import numba
from numba.extending import overload
import numpy as np

from pttools.bubble import const
from pttools.bubble.cs2 import cs2_to_ptr
from pttools.bubble.phase import Phase
from pttools.speedup import njit
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr

NUMBA_CACHE_CS2_BAG: bool = True
"""Whether to cache the Numba-compiled $c_s^2$ functions

This is enabled by default, since these functions are not expected to change between runs.
"""


@njit(cache=NUMBA_CACHE_CS2_BAG)
def cs2_bag_multi[T: FloatOrArr](
        w: T,
        phase: th.FloatOrArr) -> T:
    r"""Sound speed squared, $c_s^2=\frac{1}{3}$.
    :notes:`\ `, p. 37,
    :rel_hydro_book:`\ `, eq. 2.207.
    """
    return np.ones_like(w) * np.ones_like(phase) / 3.


@njit(cache=NUMBA_CACHE_CS2_BAG)
def cs2_bag_neg[T: FloatOrArr](w: T, phase: th.FloatOrArr) -> T:
    return - cs2_bag_multi(w, phase)  # pyrefly: ignore[bad-return]


def _cs2_bag_scalar(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
    """The scalar versions of the bag functions have to be compiled to cfuncs if jitting is disabled,
    as otherwise the cfunc version of the differential cannot be created.
    """
    return const.CS0_2


@numba.cfunc(th.CS2FunScalarSig, cache=NUMBA_CACHE_CS2_BAG)
def cs2_bag_scalar_cfunc(w: float, phase: Phase) -> float:
    return const.CS0_2


@njit(cache=NUMBA_CACHE_CS2_BAG)
def cs2_bag_temp[T: FloatOrArr](temp: T, phase: th.FloatOrArr) -> T:
    return cs2_bag_multi(temp, phase)


def _cs2_bag_arr(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatArr:
    return np.full_like(w, const.CS0_2)


def cs2_bag[T: FloatOrArr](w: T, phase: th.FloatOrArr) -> T:
    r"""
    Speed of sound squared in Bag model, equal to $\frac{1}{3}$, independent of enthalpy $w$.

    :notes:`\ `, p. 37,
    :rel_hydro_book:`\ `, eq. 2.207

    :param w: enthalpy $w$
    :param phase: phase $\phi$
    :return: speed of sound squared $c_s^2$
    """
    if isinstance(w, float):
        return cs2_bag_scalar(w, phase)  # pyrefly: ignore[bad-return]
    if isinstance(w, np.ndarray):
        return cs2_bag_arr(w, phase)  # pyrefly: ignore[bad-return]
    raise TypeError(f"Unknown type for w: {type(w)}")


# The Numba caching of the overload implementations is disabled, as their cache files collide
# with those of the njit-compiled versions of the same functions, which results in segmentation faults.
@overload(cs2_bag, jit_options={"nopython": True})
def cs2_bag_numba(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
    if isinstance(w, numba.types.Float):
        return _cs2_bag_scalar
    if isinstance(w, numba.types.Array):
        return _cs2_bag_arr
    raise TypeError(f"Unknown type for w: {type(w)}")


#: Pointer to the scalar $c_s^2$ function of the Bag Model
CS2_BAG_SCALAR_PTR: th.CS2FunScalarPtr = cs2_to_ptr(cs2_bag_scalar_cfunc)
CS2ScalarCType = cs2_bag_scalar_cfunc.ctypes
cs2_bag_scalar = njit(cache=NUMBA_CACHE_CS2_BAG)(_cs2_bag_scalar)
cs2_bag_arr = njit(cache=NUMBA_CACHE_CS2_BAG)(_cs2_bag_arr)
