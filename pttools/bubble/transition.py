"""Useful quantities for deciding type of transition"""

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


# -----
# TODO: Untested
# -----

def identify_solution_type_beyond_bag(
        v_wall: float,
        alpha_n: float,
        model: "Model",
        wp: float = 1) -> boundary.SolutionType:

    if is_surely_detonation(v_wall, alpha_n, model):
        return boundary.SolutionType.DETON
    if is_surely_sub_def(v_wall, alpha_n, model, wp):
        return boundary.SolutionType.SUB_DEF
    return boundary.SolutionType.UNKNOWN


def is_surely_sub_def(v_wall: float, alpha_n: float, model: "Model", wn: float = 1):
    r"""If v_wall < cs_b for all w in [0, wn], then it is certainly a deflagration"""
    if v_wall**2 < max_cs2_inside_sub_def(model, wn):
        return True
    return False


def is_surely_detonation(v_wall: float, alpha_n: float, model: "Model") -> float:
    r"""If $v_w > v_{CJ}$, it is certainly a detonation"""
    v_cj = v_chapman_jouguet(alpha_n, model)
    if v_wall > v_cj:
        return True
    return False


def max_cs2_inside_sub_def(model: "Model", wn: float = 1) -> float:
    r"""If the wall speed $v_w < c_s(w) \forall w \in [0, w_n]$,
    then the wall is certainly subsonic and therefore the solution is certainly a subsonic deflagration."""
    def func(w):
        return -model.cs2(w, boundary.Phase.BROKEN)

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
