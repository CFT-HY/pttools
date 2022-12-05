"""Useful functions for finding the properties of a solution."""

import numba.types
import numpy as np

import pttools.type_hints as th
from .boundary import Phase
from . import relativity


@numba.njit
def find_v_index(xi: np.ndarray, v_target: float) -> int:
    r"""
    The first array index of $\xi$ where value is just above $v_\text{target}$.
    If no xi > v_target is found, returns 0.
    """
    return np.argmax(xi >= v_target)


def find_phase(xi: np.ndarray, v_wall: float) -> np.ndarray:
    i_wall = find_v_index(xi, v_wall)
    # This presumes that Phase.SYMMETRIC = 0
    phase = np.zeros_like(xi)
    phase[:i_wall] = Phase.BROKEN
    return phase


@numba.vectorize
def v_max_behind(xi: th.FloatOrArr, cs: float):
    r"""Maximum fluid velocity behind the wall.
    Given by the condition $\mu(\xi, v) = c_s$.
    This results in:
    $$ v_\text{max} = \frac{c_s-\xi}{c_s \xi - 1} $$

    Requires that the sound speed is a constant!

    :param xi: $\xi$
    :param cs: $c_s$, speed of sound behind the wall (=in the broken phase)
    :return: $v_\text{max,behind}$
    """
    return relativity.lorentz(xi, cs)
