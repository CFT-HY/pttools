"""Useful quantities for deciding the type of a transition"""

import logging
import typing as tp

import numba
import numpy as np
from scipy.optimize import fminbound

import pttools.type_hints as th
from . import alpha as alpha_tools
from . import boundary
from . import const
from pttools.bubble.chapman_jouguet import v_chapman_jouguet
if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


@numba.njit
def identify_solution_type(v_wall: float, alpha_n: float, exit_on_error: bool = False) -> boundary.SolutionType:
    """
    Determines wall type from wall speed and global strength parameter.
    solution_type = [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    # v_wall = wall velocity, alpha_n = relative trace anomaly at nucleation temp outside shell
    sol_type = boundary.SolutionType.ERROR  # Default
    if alpha_n < alpha_tools.alpha_n_max_detonation(v_wall):
        # Must be detonation
        sol_type = boundary.SolutionType.DETON
    else:
        if alpha_n < alpha_tools.alpha_n_max_deflagration(v_wall):
            if v_wall <= const.CS0:
                sol_type = boundary.SolutionType.SUB_DEF
            else:
                sol_type = boundary.SolutionType.HYBRID

    if sol_type == boundary.SolutionType.ERROR and exit_on_error:
        with numba.objmode:
            logger.error(f"No solution for v_wall = %s, alpha_n = %s", v_wall, alpha_n)
        raise RuntimeError("No solution for given v_wall, alpha_n")

    return sol_type


def identify_solution_type_beyond_bag(
        model: "Model",
        v_wall: float,
        alpha_n: float,
        wn_guess: float,
        wm_guess: float
        ) -> boundary.SolutionType:

    wn = model.w_n(alpha_n, wn_guess)
    v_cj = v_chapman_jouguet(model, alpha_n, wn_guess=wn, wm_guess=wm_guess)

    if is_surely_detonation(v_wall, v_cj):
        return boundary.SolutionType.DETON
    if is_surely_sub_def(v_wall, model, wn):
        return boundary.SolutionType.SUB_DEF
    if cannot_be_detonation(v_wall, v_cj) and cannot_be_sub_def(v_wall, model, wn):
        return boundary.SolutionType.HYBRID
    logger.warning(f"Could not determine solution type for {model.name} with v_wall={v_wall}, alpha_n={alpha_n}")
    return boundary.SolutionType.UNKNOWN


def cannot_be_sub_def(v_wall: float, model: "Model", wn: float):
    """If the wall is certainly hypersonic, it cannot be a subsonic deflagration."""
    if v_wall**2 > max_cs2_inside_def(model, wn):
        return True
    return False


def is_surely_sub_def(v_wall: float, model: "Model", wn: float):
    r"""If v_wall < cs_b for all w in [0, wn], then it is certainly a deflagration."""
    if v_wall**2 < min_cs2_inside_sub_def(model, wn):
        return True
    return False


def is_surely_detonation(v_wall: float, v_cj: float) -> float:
    r"""If $v_w > v_{CJ}$, it is certainly a detonation"""
    return v_wall > v_cj


def cannot_be_detonation(v_wall: float, v_cj: float) -> float:
    r"""If $v_w < v_{CJ}, it cannot be a detonation"""
    return v_wall < v_cj


def max_cs2_inside_def(model: "Model", wn: float = 1, allow_fail: bool = False) -> float:
    r"""If the wall speed $v_w > c_s(w) \forall w \in [0, w_n]$,
    then the wall is certainly hypersonic and therefore the solution cannot be a subsonic deflagration."""
    def func(w):
        return -model.cs2(w, boundary.Phase.BROKEN)

    sol = fminbound(func, x1=0, x2=wn, full_output=True)
    cs2 = -sol[1]
    if sol[2]:
        msg = f"Could not find max_cs2_inside_def. Using wn={wn}, max_cs2_inside_def={cs2}. Iterations: {sol[3]}"
        logger.error(msg)
        if not allow_fail:
            raise RuntimeError(msg)
    return cs2


def min_cs2_inside_sub_def(model: "Model", wn: float = 1) -> float:
    r"""If the wall speed $v_w < c_s(w) \forall w \in [0, w_n]$,
    then the wall is certainly subsonic and therefore the solution is certainly a subsonic deflagration."""
    def func(w):
        return model.cs2(w, boundary.Phase.BROKEN)

    sol = fminbound(func, x1=0, x2=wn)
    return sol[0]

# -----


@numba.njit
def identify_solution_type_alpha_plus(v_wall: float, alpha_p: float) -> boundary.SolutionType:
    r"""
    Determines wall type from wall speed $v_\text{wall}$ and at-wall strength parameter $\alpha_+$.

    :param v_wall: $v_\text{wall}$
    :param alpha_p: $\alpha_+$
    :return: solution type [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    # TODO: Currently this is for the bag model only
    if v_wall <= const.CS0:
        sol_type = boundary.SolutionType.SUB_DEF
    else:
        if alpha_p < alpha_tools.alpha_plus_max_detonation(v_wall):
            sol_type = boundary.SolutionType.DETON
            if alpha_tools.alpha_plus_min_hybrid(v_wall) < alpha_p < 1/3:
                with numba.objmode:
                    logger.warning((
                        "Hybrid and detonation both possible for v_wall = {}, alpha_plus = {}. "
                        "Choosing detonation.").format(v_wall, alpha_p))
        else:
            sol_type = boundary.SolutionType.HYBRID

    if alpha_p > 1/3 and sol_type != boundary.SolutionType.DETON:
        with numba.objmode:
            logger.error("No solution for for v_wall = {}, alpha_plus = {}".format(v_wall, alpha_p))
        sol_type = boundary.SolutionType.ERROR

    return sol_type


@numba.njit
def max_speed_deflag(alpha_p: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Maximum speed for a deflagration: speed where wall and shock are coincident.
    May be greater than 1, meaning that hybrids exist for all wall speeds above cs.
    $\alpha_+ < \frac{1}{3}$, but $\alpha_n$ unbounded above.

    :param alpha_p: $\alpha_+$
    """
    return 1 / (3 * boundary.v_plus(const.CS0, alpha_p, boundary.SolutionType.SUB_DEF))


@numba.njit
def min_speed_deton(alpha: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Minimum speed for a detonation (Jouguet speed).
    Equivalent to $v_+(cs_0,\alpha)$.
    Note that $\alpha_+ = \alpha_n$ for detonation.

    :param alpha: $\alpha$
    """
    return (const.CS0 / (1 + alpha)) * (1 + np.sqrt(alpha * (2. + 3. * alpha)))
