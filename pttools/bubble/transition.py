"""Useful quantities for deciding the type of a transition"""

import logging
import typing as tp

import numba

import pttools.type_hints as th
from . import alpha as alpha_tools
from .boundary import Phase, SolutionType, v_plus
from . import const
if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


@numba.njit
def identify_solution_type_bag(v_wall: float, alpha_n: float, exit_on_error: bool = False) -> SolutionType:
    """
    Determines wall type from wall speed and global strength parameter.
    solution_type = [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    if alpha_n < alpha_tools.alpha_n_max_detonation_bag(v_wall):
        return SolutionType.DETON
    else:
        if alpha_n < alpha_tools.alpha_n_max_deflagration_bag(v_wall):
            if v_wall <= const.CS0:
                return SolutionType.SUB_DEF
            return SolutionType.HYBRID
        # elif v_wall > const.CS0 and alpha_n < alpha_tools.alpha_n_max_hybrid_bag(v_wall):
        #     with numba.objmode:
        #         logger.warning(
        #             "Using an untested way to identify the solution as a hybrid with v_wall=%s, alpha_n=%s",
        #             v_wall, alpha_n
        #         )
        #     return SolutionType.HYBRID

    if exit_on_error:
        with numba.objmode:
            logger.error(f"No solution for v_wall=%s, alpha_n=%s", v_wall, alpha_n)
        raise RuntimeError("No solution for given v_wall, alpha_n")

    return SolutionType.ERROR


def validate_solution_type(
        model: "Model",
        v_wall: float,
        alpha_n: float,
        sol_type: SolutionType,
        wn: float = None,
        wn_guess: float = None,
        wm_guess: float = None) -> SolutionType:
    if sol_type is None or sol_type is SolutionType.UNKNOWN:
        sol_type = model.solution_type(
            v_wall=v_wall, alpha_n=alpha_n, wn=wn, wn_guess=wn_guess, wm_guess=wm_guess
        )
    if sol_type in [SolutionType.UNKNOWN, SolutionType.ERROR]:
        msg = \
            "Could not determine solution type automatically for " \
            f"model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. " \
            f"Got sol_type={sol_type}. Please choose it manually."
        logger.error(msg)
        raise ValueError(msg)
    return sol_type


def cannot_be_detonation(v_wall: float, v_cj: float) -> float:
    r"""If $v_w < v_{CJ}$, it cannot be a detonation"""
    return v_wall < v_cj


def cannot_be_sub_def(model: "Model", v_wall: float, wn: float) -> bool:
    r"""If the wall speed $v_w > c_{sb}(w) \forall w \in [0, w_n]$,
    then the wall is certainly hypersonic in the broken phase and must have fluid movement inside the wall
    to satisfy the boundary conditions. Therefore, the solution cannot be a subsonic deflagration."""
    cs2_max, w_max = model.cs2_max(wn, Phase.BROKEN)
    return v_wall**2 > cs2_max


def is_surely_detonation(v_wall: float, v_cj: float) -> float:
    r"""If $v_w > v_{CJ}$, it is certainly a detonation"""
    return v_wall > v_cj


def is_surely_sub_def(model: "Model", v_wall: float, wn: float) -> bool:
    r"""If the wall speed $v_w < c_{sb}(w) \forall w \in [0, w_n]$,
    then the wall is certainly subsonic in the broken phase,
    and therefore the solution is certainly a subsonic deflagration."""
    cs2_min, w_min = model.cs2_min(wn, Phase.BROKEN)
    return v_wall**2 < cs2_min


@numba.njit
def identify_solution_type_alpha_plus(v_wall: float, alpha_p: float) -> SolutionType:
    r"""
    Determines wall type from wall speed $v_\text{wall}$ and at-wall strength parameter $\alpha_+$.

    :param v_wall: $v_\text{wall}$
    :param alpha_p: $\alpha_+$
    :return: solution type [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    # TODO: Currently this is for the bag model only
    if v_wall <= const.CS0:
        sol_type = SolutionType.SUB_DEF
    else:
        if alpha_p < alpha_tools.alpha_plus_max_detonation_bag(v_wall):
            sol_type = SolutionType.DETON
            if alpha_tools.alpha_plus_min_hybrid(v_wall) < alpha_p < 1/3:
                with numba.objmode:
                    logger.warning((
                        "Hybrid and detonation both possible for v_wall = {}, alpha_plus = {}. "
                        "Choosing detonation.").format(v_wall, alpha_p))
        else:
            sol_type = SolutionType.HYBRID

    if alpha_p > 1/3 and sol_type != SolutionType.DETON:
        with numba.objmode:
            logger.error("No solution for for v_wall = {}, alpha_plus = {}".format(v_wall, alpha_p))
        sol_type = SolutionType.ERROR

    return sol_type


@numba.njit
def max_speed_deflag(alpha_p: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Maximum speed for a deflagration: speed where wall and shock are coincident.
    May be greater than 1, meaning that hybrids exist for all wall speeds above cs.
    $\alpha_+ < \frac{1}{3}$, but $\alpha_n$ unbounded above.

    :param alpha_p: $\alpha_+$
    """
    return 1 / (3 * v_plus(const.CS0, alpha_p, SolutionType.SUB_DEF))
