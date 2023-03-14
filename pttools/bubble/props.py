"""Useful functions for finding the properties of a solution."""

import numba.types
import numpy as np

import pttools.type_hints as th
from .boundary import Phase, SolutionType
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


def v_and_w_from_solution(v: np.ndarray, w: np.ndarray, xi: np.ndarray, v_wall: float, sol_type: SolutionType):
    i_wall = np.argmax(v)
    i_wall_w = np.argmax(w)
    if i_wall != i_wall_w:
        raise ValueError("The wall is not at the same index in v and w")
    vw = xi[i_wall]
    if not np.isclose(vw, v_wall):
        raise ValueError(f"v_wall={v_wall}, computed v_wall={vw}")

    # The direction in the change of values depends on the solution type
    if sol_type == SolutionType.DETON:
        i_wall += 1

    vp = v[i_wall]
    if vp > v_wall:
        raise ValueError(f"Cannot have vp > v_wall, got vp={vp}, v_wall={v_wall}")
    vm = v[i_wall-1]
    vp_tilde = relativity.lorentz(v_wall, vp)
    if np.isnan(vp_tilde) or vp_tilde < 0:
        raise ValueError(f"vp={vp}, vp_tilde={vp_tilde}")
    vm_tilde = relativity.lorentz(v_wall, vm)
    if np.isnan(vm_tilde) or vm_tilde < 0:
        raise ValueError(f"vm={vm}, vm_tilde={vm_tilde}")
    wp = w[i_wall]
    wm = w[i_wall-1]

    if sol_type == SolutionType.DETON:
        if wp > wm:
            raise ValueError("Got wp > wm for a detonation")
    else:
        if wp < wm:
            raise ValueError("Got wp < wm for a deflagration or hybrid")

    wn = w[-1]
    return vp, vm, vp_tilde, vm_tilde, wp, wm, wn
