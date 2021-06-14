"""Functions for calculating the properties of the bubble boundaries"""

import enum
import typing as tp

import numpy as np

import pttools.type_hints as th
from . import const
from . import relativity


@enum.unique
class SolutionType(str, enum.Enum):
    DETON = "Detonation"
    ERROR = "Error"
    HYBRID = "Hybrid"
    SUB_DEF = "Subsonic deflagration"
    UNKNOWN = "Unknown"


def v_plus(vm: th.FLOAT_OR_ARR, ap: th.FLOAT_OR_ARR, sol_type: SolutionType) -> th.FLOAT_OR_ARR:
    """
    Wall frame fluid speed v_plus ahead of the wall, as a function of
    vm = v_minus - fluid speed v_plus behind the wall
    ap = alpha_plus - strength parameter at wall
    solution_type - Detonation, Deflagration, Hybrid
    """
    X = vm + 1. / (3 * vm)
    if sol_type == SolutionType.DETON:
        b = 1.
    else:
        b = -1.
    return_value = (0.5 / (1 + ap)) * (X + b * np.sqrt(X ** 2 + 4. * ap ** 2 + (8. / 3.) * ap - (4. / 3.)))

    if isinstance(return_value, np.ndarray):
        return_value[np.where(isinstance(return_value, complex))] = np.nan
    else:
        if isinstance(return_value, complex):
            return_value = np.nan

    return return_value


def v_minus(vp: th.FLOAT_OR_ARR, ap: th.FLOAT_OR_ARR, sol_type: SolutionType = SolutionType.DETON) -> th.FLOAT_OR_ARR:
    """
    Wall frame fluid speed v_minus behind the wall, as a function of
    vp = v_plus - fluid speed v_plus behind the wall
    ap = alpha_plus - strength parameter at wall
    sol_type - Detonation, Deflagration, Hybrid
    """
    vp2 = vp ** 2
    Y = vp2 + 1. / 3.
    Z = (Y - ap * (1. - vp2))
    X = (4. / 3.) * vp2

    if sol_type == SolutionType.DETON:
        b = +1.
    else:
        b = -1.

    return_value = (0.5 / vp) * (Z + b * np.sqrt(Z ** 2 - X))

    if isinstance(return_value, np.ndarray):
        return_value[np.where(isinstance(return_value, complex))] = np.nan
    else:
        if isinstance(return_value, complex):
            return_value = np.nan

    return return_value


def fluid_speeds_at_wall(
        v_wall: float,
        alpha_p: th.FLOAT_OR_ARR,
        sol_type: SolutionType) -> tp.Tuple[float, float, float, float]:
    """
    Solves fluid speed boundary conditions at the wall, returning
        vfp_w, vfm_w, vfp_p, vfm_p
    Fluid speed vf? just behind (?=m) and just ahead (?=p) of wall,
    in wall (_w) and plasma (_p) frames.
    """
    if v_wall <= 1:
        # print( "max_speed_deflag(alpha_p)= ", max_speed_deflag(alpha_p))
        #     if v_wall < max_speed_deflag(alpha_p) and v_wall <= cs and alpha_p <= 1/3.:
        if sol_type == SolutionType.SUB_DEF:
            vfm_w = v_wall  # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(v_wall, alpha_p, sol_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        elif sol_type == SolutionType.HYBRID:
            vfm_w = const.CS0  # Fluid velocity just behind the wall in plasma frame (hybrid)
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(const.CS0, alpha_p, sol_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        elif sol_type == SolutionType.DETON:
            vfm_w = v_minus(v_wall, alpha_p)  # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_wall  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        else:
            raise ValueError(f"Unknown sol_type: {sol_type}")
    else:
        raise ValueError(f"v_wall > 1: v_wall = {v_wall}")

    return vfp_w, vfm_w, vfp_p, vfm_p


def enthalpy_ratio(v_m: th.FLOAT_OR_ARR, v_p: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Ratio of enthalpies behind (w_- ) and ahead (w_+) of a shock or
    transition front, w_-/w_+. Uses conservation of momentum in moving frame.
    """
    return relativity.gamma2(v_m) * v_m / (relativity.gamma2(v_p) * v_p)
