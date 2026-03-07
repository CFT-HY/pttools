"""Solution types of relativistic combustion for the Bag Model"""

import logging

import numba

import pttools.bubble.alpha as alpha_tools
from pttools.bubble import const
from pttools.bubble.solution_type import SolutionType

logger = logging.getLogger(__name__)


@numba.njit
def identify_solution_type_bag(v_wall: float, alpha_n: float, exit_on_error: bool = False) -> SolutionType:
    """
    Determines wall type from wall speed and global strength parameter.
    solution_type = [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    if alpha_n < alpha_tools.alpha_n_max_detonation_bag(v_wall):
        return SolutionType.DETON
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
            logger.error("No solution for v_wall=%s, alpha_n=%s", v_wall, alpha_n)
        raise RuntimeError("No solution for given v_wall, alpha_n")

    return SolutionType.ERROR


@numba.njit
def identify_solution_type_alpha_plus_bag(v_wall: float, alpha_p: float) -> SolutionType:
    r"""
    Determines wall type from wall speed $v_\text{wall}$ and at-wall strength parameter $\alpha_+$.

    :param v_wall: $v_\text{wall}$
    :param alpha_p: $\alpha_+$
    :return: solution type [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    if v_wall <= const.CS0:
        sol_type = SolutionType.SUB_DEF
    else:
        if alpha_p < alpha_tools.alpha_plus_max_detonation_bag(v_wall):
            sol_type = SolutionType.DETON
            if alpha_tools.alpha_plus_min_hybrid(v_wall) < alpha_p < 1/3:
                with numba.objmode:
                    logger.warning(
                        "Hybrid and detonation both possible for v_wall=%s, alpha_plus=%s. "
                        "Choosing detonation.",
                        v_wall, alpha_p
                    )
        else:
            sol_type = SolutionType.HYBRID

    if alpha_p > 1/3 and sol_type != SolutionType.DETON:
        with numba.objmode:
            logger.error(
                "No solution for for v_wall=%s, alpha_plus=%s",
                v_wall, alpha_p
            )
        sol_type = SolutionType.ERROR

    return sol_type
