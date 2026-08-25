"""Phase of the field that undergoes the phase transition"""

import enum

import numba
from numba.extending import overload
import numpy as np

import pttools.type_hints as th


@enum.unique
class Phase(float, enum.Enum):
    """Phase of the field that undergoes the phase transition

    In general the phase not binary, but a real number.
    Therefore, it is implemented as a float rather than bool."""
    # Do not change these values without also checking the model cs2 functions.
    # These are floats instead of integers to ensure that the Numba functions don't have to be compiled twice.
    SYMMETRIC = 0.
    BROKEN = 1.


def _get_phase_scalar(xi: th.FloatOrArr, v_wall: float) -> th.FloatOrArr:
    return Phase.BROKEN if xi < v_wall else Phase.SYMMETRIC


def _get_phase_arr(xi: th.FloatOrArr, v_wall: float) -> th.FloatOrArr:
    phase = np.zeros_like(xi)
    phase[np.where(xi < v_wall)] = Phase.BROKEN.value
    return phase


def get_phase(xi: th.FloatOrArr, v_wall: float) -> th.FloatOrArr:
    r"""
    Returns array indicating phase of system.
    in symmetric phase $(\xi > v_w)$, phase = 0
    in broken phase $(\xi < v_w)$, phase = 1

    :return: phase
    """
    if isinstance(xi, float):
        return _get_phase_scalar(xi, v_wall)
    if isinstance(xi, np.ndarray):
        return _get_phase_arr(xi, v_wall)
    raise TypeError(f"Unknown type for {type(xi)}")


@overload(get_phase, jit_options={"nopython": True})
def _get_phase_numba(xi: th.FloatOrArr, v_wall: float) -> th.NumbaFunc:
    if isinstance(xi, numba.types.Float):
        return _get_phase_scalar
    if isinstance(xi, numba.types.Array):
        if not xi.ndim:
            return _get_phase_scalar
        return _get_phase_arr
    raise TypeError(f"Unknown type for {type(xi)}")
