"""Solution types of relativistic combustion for the Bag Model"""

import logging

import numba

import pttools.bubble.alpha as alpha_tools
from pttools.bubble.const import ALPHA_PLUS_MAX_DEF, CS0
from pttools.bubble.solution_type import SolutionType
import pttools.type_hints as th
from pttools.speedup import njit
from pttools.bubble.integrate import FluidIntegrateMethod
from pttools.speedup.differential import DifferentialPointer

logger = logging.getLogger(__name__)


@njit
def identify_solution_type_bag(
        v_wall: float,
        alpha_n: float,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_fun: th.CS2Fun,
        exit_on_error: bool = False) -> SolutionType:
    """
    Determines wall type from wall speed and global strength parameter.
    solution_type = [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    if alpha_n < alpha_tools.alpha_n_max_detonation_bag(v_wall):
        return SolutionType.DETON
    if alpha_n < alpha_tools.alpha_n_max_deflagration_bag(
            v_wall, df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, cs2_fun=cs2_fun):
        if v_wall <= CS0:
            return SolutionType.SUB_DEF
        return SolutionType.HYBRID
    # elif v_wall > CS0 and alpha_n < alpha_tools.alpha_n_max_hybrid_bag(v_wall):
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


@njit
def identify_solution_type_alpha_plus_bag(v_wall: float, alpha_plus: float) -> SolutionType:
    r"""
    Determines wall type from wall speed $v_\text{wall}$ and at-wall strength parameter $\alpha_+$.

    :param v_wall: $v_\text{wall}$
    :param alpha_plus: $\alpha_+$
    :return: solution type [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    if v_wall <= CS0:
        sol_type = SolutionType.SUB_DEF
    else:
        if alpha_plus < alpha_tools.alpha_plus_max_detonation_bag(v_wall):
            sol_type = SolutionType.DETON
            if alpha_tools.alpha_plus_min_hybrid(v_wall) < alpha_plus < ALPHA_PLUS_MAX_DEF:
                with numba.objmode:
                    logger.warning(
                        "Both hybrid and detonation are possible for v_wall=%s, alpha_plus=%s. "
                        "Choosing detonation.",
                        v_wall, alpha_plus
                    )
        else:
            sol_type = SolutionType.HYBRID

    if alpha_plus > ALPHA_PLUS_MAX_DEF and sol_type != SolutionType.DETON:
        with numba.objmode:
            logger.error(
                "No solution for for v_wall=%s, alpha_plus=%s",
                v_wall, alpha_plus
            )
        sol_type = SolutionType.ERROR

    return sol_type
