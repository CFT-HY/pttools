"""Junction conditions for the Bag Model

.. plot:: fig/vm_vp_plane.py
"""

import numba

from pttools.bubble import const
from pttools.bubble.relativity import lorentz
from pttools.bubble.solution_type import SolutionType
from pttools.bubble.v_plus import v_plus
from pttools.bubble.v_minus import v_minus


@numba.njit
def fluid_speeds_at_wall_bag(
        v_wall: float,
        alpha_plus: float,
        sol_type: SolutionType) -> tuple[float, float, float, float]:
    r"""
    Solves fluid speed boundary conditions at the wall to obtain
    the fluid speeds both in the universe (plasma frame): $v_+$ and $v_+$
    and in the wall frame: $\tilde{v}_+, \tilde{v}_-$.

    Bag model only.

    The abbreviations are: fluid speed (vf) just behind (m=minus) and just ahead (p=plus) of wall,
    in wall (_w) and plasma/universe (_p) frames.

    TODO: add a validity check for v_minus

    :param v_wall: $v_\text{wall}$
    :param alpha_plus: $\alpha_+$
    :param sol_type: solution type
    :return: $\tilde{v}_+,\tilde{v}_-,v_+,v_-$
    """
    if v_wall > 1:
        # Todo: better error handling and logging
        # with numba.objmode:
        #     logger.error("v_wall > 1: v_wall = %s", v_wall)
        raise ValueError(f"Got v_wall = {v_wall} > 1")

    # print("max_speed_deflag(alpha_plus)=", max_speed_deflag(alpha_plus))
    # if v_wall < max_speed_deflag(alpha_plus) and v_wall <= cs and alpha_p <= 1/3.:
    if sol_type == SolutionType.SUB_DEF.value:
        # For clarity these are defined here in the same order as returned
        vfp_w = v_plus(v_wall, alpha_plus, sol_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
        vfm_w = v_wall  # Fluid velocity just behind the wall in wall frame (v-)
        vfp_p = lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        vfm_p = lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
    elif sol_type == SolutionType.HYBRID.value:
        vfp_w = v_plus(const.CS0, alpha_plus, sol_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
        vfm_w = const.CS0  # Fluid velocity just behind the wall in plasma frame (hybrid)
        vfp_p = lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        vfm_p = lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
    elif sol_type == SolutionType.DETON.value:
        vfp_w = v_wall  # Fluid velocity just ahead of the wall in wall frame (v+)
        vfm_w = v_minus(v_wall, alpha_plus)  # Fluid velocity just behind the wall in wall frame (v-)
        vfp_p = lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        vfm_p = lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
    else:
        # Todo: better error handling and logging
        # with numba.objmode:
        #     logger.error("Unknown sol_type: %s", sol_type)
        raise ValueError(f"Unknown sol_type={sol_type}")

    return vfp_w, vfm_w, vfp_p, vfm_p
