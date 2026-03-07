"""Useful functions for finding the properties of a solution."""

import numba.types
import numpy as np

from pttools.bubble.phase import Phase
from pttools.bubble.solution_type import SolutionType
from pttools.bubble import relativity
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr


def find_phase(xi: th.FloatArr1D, v_wall: float) -> th.FloatArr1D:
    r"""Get the phase at each given $\xi$ value"""
    # Todo: Replace this with pttools.bubble.phase.get_phase
    i_wall = find_v_index(xi, v_wall)
    # This presumes that Phase.SYMMETRIC = 0
    phase = np.zeros_like(xi)
    if i_wall == 0:
        return phase
    phase[:i_wall-1] = Phase.BROKEN
    # Fix for detonations
    if np.isclose(xi[i_wall], v_wall):
        phase[i_wall-1] = Phase.BROKEN
    return phase


@numba.njit
def find_v_index(xi: th.FloatArr, v_target: float) -> int:
    r"""
    The first array index of $\xi$ where value is just above $v_\text{target}$.
    If no xi > v_target is found, returns 0.
    """
    return np.argmax(xi >= v_target)


@numba.njit
def v_max_behind[T: FloatOrArr](xi: T, cs: float) -> T:
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


def v_and_w_from_solution(
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        sol_type: SolutionType) \
        -> tuple[float, float, float, float, float, float, float, float]:
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
    wm_sh: float = w[np.argmax(np.flip(w) > wn)]
    return vp, vm, vp_tilde, vm_tilde, wp, wm, wn, wm_sh
