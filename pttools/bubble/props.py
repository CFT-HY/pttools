"""Useful functions for finding the properties of a solution."""

import numpy as np

from pttools.bubble import relativity
from pttools.bubble.phase import Phase
from pttools.bubble.solution_type import SolutionType
from pttools.speedup import njit
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr


def find_phase(xi: th.FloatArr1D, v_wall: float, sol_type: SolutionType | None = None) -> th.FloatArr1D:
    r"""Get the phase at each given $\xi$ value.

    The points with $\xi < {v}_\text{wall}$ are in the broken phase,
    and the points with $\xi > {v}_\text{wall}$ are in the symmetric phase.
    For the points with $\xi = {v}_\text{wall}$, the phase is determined by the structure of the profile.
    If there are several points at $\xi = {v}_\text{wall}$, as in hybrids,
    the last of them is in the symmetric phase and the others in the broken phase.
    If there is a single point at $\xi = {v}_\text{wall}$, it is the end of the rarefaction wave in detonations
    and therefore in the broken phase,
    and the start of the shell in subsonic deflagrations and therefore in the symmetric phase.
    As these cannot be distinguished based on $\xi$ alone,
    the single point is considered to be in the symmetric phase unless ``sol_type`` is a detonation.

    :param xi: $\xi$ values of the profile in increasing order
    :param v_wall: ${v}_\text{wall}$, wall speed
    :param sol_type: type of the solution
    :return: phase
    """
    # This presumes that Phase.SYMMETRIC = 0
    phase = np.zeros_like(xi)
    if not np.any(xi >= v_wall):
        phase[:] = Phase.BROKEN
        return phase
    i_wall = find_v_index(xi, v_wall)
    phase[:i_wall] = Phase.BROKEN
    # The points at the wall are set to exactly v_wall by the solvers,
    # so the tolerance has to be tight to not include the neighbouring points.
    n_at_wall = 0
    while i_wall + n_at_wall < xi.size and np.isclose(xi[i_wall + n_at_wall], v_wall, rtol=1e-12, atol=0):
        n_at_wall += 1
    if n_at_wall > 1:
        phase[i_wall:i_wall + n_at_wall - 1] = Phase.BROKEN
    elif n_at_wall == 1 and sol_type == SolutionType.DETON:
        phase[i_wall] = Phase.BROKEN
    return phase


@njit(cache=True)
def find_v_index(xi: th.FloatArr, v_target: float) -> int:
    r"""
    The first array index of $\xi$ where value is just above $v_\text{target}$.

    If no xi > v_target is found, returns 0.
    """
    return int(np.argmax(xi >= v_target))


@njit
def v_max_behind[T: FloatOrArr](xi: T, cs: T | float) -> T:
    r"""Maximum fluid velocity behind the wall.

    Given by the condition $\mu(\xi, v) = c_s$.
    This results in:
    $${v}_\text{max} = \frac{c_s-\xi}{c_s \xi - 1}$$.

    This requires that the sound speed is a constant.

    :param xi: $\xi$
    :param cs: $c_s$, speed of sound behind the wall (=in the broken phase)
    :return: $v_\text{max,behind}$
    """
    return relativity.lorentz(xi=xi, v=cs)


def v_and_w_from_solution(
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        sol_type: SolutionType) \
        -> tuple[float, float, float, float, float, float, float, float]:
    r"""Get the fluid velocities and enthalpies at the wall and at the shock from a solved fluid profile.

    The wall is located at the maximum of $v$ and $w$.

    :param v: $v$, fluid velocity
    :param w: $w$, enthalpy
    :param xi: $\xi$
    :param v_wall: ${v}_\text{wall}$, wall speed
    :param sol_type: solution type
    :return: $v_{+}, v_{-}, \tilde{v}_+, \tilde{v}_-, w_{+}, w_{-}, w_n, w_{-,sh}$
    :raises ValueError: if the profile is inconsistent with the given $v_\text{wall}$ and solution type
    """
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
    elif wp < wm:
        raise ValueError("Got wp < wm for a deflagration or hybrid")

    wn = w[-1]
    wm_sh: float = w[np.argmax(np.flip(w) > wn)]
    return vp, vm, vp_tilde, vm_tilde, wp, wm, wn, wm_sh
