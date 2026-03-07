"""Speed of sound for the Bag Model"""

import numba
from numba.extending import overload
import numpy as np

import pttools.type_hints as th
from pttools.bubble.phase import Phase
from pttools.bubble import const


NUMBA_CACHE_CS2_BAG: bool = True
"""Whether to cache the Numba-compiled $c_s^2$ functions

This is enabled by default, since these functions are not expected to change between runs.
"""


@numba.njit(cache=NUMBA_CACHE_CS2_BAG)
def cs2_bag_multi(
        w: th.FloatOrArr,
        phase: th.FloatOrArr) -> th.FloatOrArr:
    r"""Sound speed squared, $c_s^2=\frac{1}{3}$.
    :notes:`\ `, p. 37,
    :rel_hydro_book:`\ `, eq. 2.207
    """
    return np.ones_like(w) * np.ones_like(phase) / 3.


@numba.njit(cache=NUMBA_CACHE_CS2_BAG)
def cs2_bag_neg(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
    return - cs2_bag_multi(w, phase)


# pylint: disable=unused-argument
def _cs2_bag_scalar(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
    """The scalar versions of the bag functions have to be compiled to cfuncs if jitting is disabled,
    as otherwise the cfunc version of the differential cannot be created.
    """
    return const.CS0_2


@numba.cfunc(th.CS2FunScalarSig, cache=NUMBA_CACHE_CS2_BAG)
# pylint: disable=unused-argument
def cs2_bag_scalar_cfunc(w: float, phase: Phase) -> float:
    return const.CS0_2

@numba.njit(cache=NUMBA_CACHE_CS2_BAG)
def cs2_bag_temp(temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
    return cs2_bag_multi(temp, phase)

# pylint: disable=unused-argument
def _cs2_bag_arr(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
    return np.full_like(w, const.CS0_2)


def cs2_bag(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Speed of sound squared in Bag model, equal to $\frac{1}{3}$, independent of enthalpy $w$.

    :notes:`\ `, p. 37,
    :rel_hydro_book:`\ `, eq. 2.207

    :param w: enthalpy $w$
    :param phase: phase $\phi$
    :return: speed of sound squared $c_s^2$
    """
    if isinstance(w, float):
        return cs2_bag_scalar(w, phase)
    if isinstance(w, np.ndarray):
        return cs2_bag_arr(w, phase)
    raise TypeError(f"Unknown type for w: {type(w)}")


@overload(cs2_bag, jit_options={"nopython": True, "cache": NUMBA_CACHE_CS2_BAG})
def cs2_bag_numba(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
    if isinstance(w, numba.types.Float):
        return _cs2_bag_scalar
    if isinstance(w, numba.types.Array):
        return _cs2_bag_arr
    raise TypeError(f"Unknown type for w: {type(w)}")


CS2_BAG_SCALAR_PTR: int = cs2_bag_scalar_cfunc.address
CS2ScalarCType = cs2_bag_scalar_cfunc.ctypes
cs2_bag_scalar = numba.njit(cache=NUMBA_CACHE_CS2_BAG)(_cs2_bag_scalar)
cs2_bag_arr = numba.njit(cache=NUMBA_CACHE_CS2_BAG)(_cs2_bag_arr)
