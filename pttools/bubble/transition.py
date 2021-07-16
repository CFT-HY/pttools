"""Useful quantities for deciding type of transition"""

import logging

import numba
import numpy as np

import pttools.type_hints as th
from . import alpha as alpha_tools
from . import boundary
from . import const

logger = logging.getLogger(__name__)


@numba.njit
def min_speed_deton(alpha: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Minimum speed for a detonation (Jouguet speed).
    Equivalent to $v_+(cs_0,\alpha)$.
    Note that $\alpha_+ = \alpha_n$ for detonation.
    """
    return (const.CS0 / (1 + alpha)) * (1 + np.sqrt(alpha * (2. + 3. * alpha)))


@numba.njit
def max_speed_deflag(alpha_p: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Maximum speed for a deflagration: speed where wall and shock are coincident.
    May be greater than 1, meaning that hybrids exist for all wall speeds above cs.
    $alpha_plus < \frac{1}{3}$, but $\alpha_n$ unbounded above.
    """
    return 1/(3 * boundary.v_plus(const.CS0, alpha_p, boundary.SolutionType.SUB_DEF))


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

    if (sol_type == boundary.SolutionType.ERROR) & exit_on_error:
        with numba.objmode:
            logger.error(f"No solution for v_wall = %s, alpha_n = %s", v_wall, alpha_n)
        raise RuntimeError("No solution for given v_wall, alpha_n")

    return sol_type


@numba.njit
def identify_solution_type_alpha_plus(v_wall: float, alpha_p: float) -> boundary.SolutionType:
    """
    Determines wall type from wall speed and at-wall strength parameter.
    solution_type = [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
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
