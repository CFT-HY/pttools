"""Functions for calculating the properties of the bubble boundaries"""

import enum
import logging
import typing as tp

import numba
import numpy as np

import pttools.type_hints as th
from . import const
from . import relativity

logger = logging.getLogger(__name__)


@enum.unique
class SolutionType(str, enum.Enum):
    """There are three different types of relativistic combustion.
    For further details, please see chapter 7.2 and figure 14
    of the :notes:`lecture notes <>`.

    .. plot:: fig/relativistic_combustion.py
    """
    DETON = "Detonation"
    ERROR = "Error"
    HYBRID = "Hybrid"
    SUB_DEF = "Subsonic deflagration"
    UNKNOWN = "Unknown"


@numba.njit
def _v_plus_scalar(vm: float, ap: float, sol_type: SolutionType) -> float:
    X = vm + 1. / (3 * vm)
    b = 1. if sol_type == SolutionType.DETON.value else -1.
    return (0.5 / (1 + ap)) * (X + b * np.sqrt(X ** 2 + 4. * ap ** 2 + (8. / 3.) * ap - (4. / 3.)))
    # if np.imag(ret):
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_plus. This is deprecated. "
    #             "Check the types of the arguments.")
    #     return np.nan
    # return ret


@numba.njit
def _v_plus_arr(vm: th.FLOAT_OR_ARR, ap: th.FLOAT_OR_ARR, sol_type: SolutionType) -> np.ndarray:
    X = vm + 1. / (3 * vm)
    b = 1. if sol_type == SolutionType.DETON.value else -1.
    return (0.5 / (1 + ap)) * (X + b * np.sqrt(X ** 2 + 4. * ap ** 2 + (8. / 3.) * ap - (4. / 3.)))
    # complex_inds = np.where(np.imag(ret))
    # if np.any(complex_inds):
    #     ret[np.where(np.imag(ret))] = np.nan
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_plus. This is deprecated. "
    #             "Check the types of the arguments.")
    # return np.real(ret)


@numba.generated_jit(nopython=True)
def v_plus(vm: float, ap: th.FLOAT_OR_ARR, sol_type: SolutionType) -> th.FLOAT_OR_ARR_NUMBA:
    r"""
    Wall frame fluid speed $v_+$ ahead of the wall

    :param vm: $v_-$, fluid speed $v_+$ behind the wall
    :param ap: $\alpha_+$, strength parameter at the wall
    :param sol_type: Detonation, Deflagration, Hybrid
    :return: $v_+$, fluid speed ahead of the wall
    """
    # TODO: add support for having both arguments as arrays
    if isinstance(vm, numba.types.Float) and isinstance(ap, numba.types.Float):
        return _v_plus_scalar
    if isinstance(vm, numba.types.Array) != isinstance(ap, numba.types.Array):
        return _v_plus_arr
    if isinstance(vm, float) and isinstance(ap, float):
        return _v_plus_scalar(vm, ap, sol_type)
    if isinstance(vm, np.ndarray) != isinstance(ap, np.ndarray):
        return _v_plus_arr(vm, ap, sol_type)
    raise TypeError(f"Unknown argument types: vm = {type(vm)}, ap = {type(ap)}")


@numba.njit
def _v_minus_scalar(vp: float, ap: float, sol_type: SolutionType):
    vp2 = vp ** 2
    Y = vp2 + 1. / 3.
    Z = (Y - ap * (1. - vp2))
    X = (4. / 3.) * vp2
    b = 1. if sol_type == SolutionType.DETON.value else -1
    return (0.5 / vp) * (Z + b * np.sqrt(Z ** 2 - X))
    # if np.imag(ret):
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_minus. This is deprecated. "
    #             "Check the types of the arguments.")
    #     return np.nan
    # return ret


@numba.njit
def _v_minus_arr(vp: th.FLOAT_OR_ARR, ap: th.FLOAT_OR_ARR, sol_type: SolutionType):
    vp2 = vp ** 2
    Y = vp2 + 1. / 3.
    Z = (Y - ap * (1. - vp2))
    X = (4. / 3.) * vp2
    b = 1. if sol_type == SolutionType.DETON.value else -1
    return (0.5 / vp) * (Z + b * np.sqrt(Z ** 2 - X))
    # complex_inds = np.where(np.imag(ret))
    # if np.any(complex_inds):
    #     ret[np.where(np.imag(ret))] = np.nan
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_minus. This is deprecated. "
    #             "Check the types of the arguments.")
    # return ret


@numba.generated_jit(nopython=True)
def v_minus(
    vp: th.FLOAT_OR_ARR,
    ap: th.FLOAT_OR_ARR,
    sol_type: SolutionType = SolutionType.DETON) -> th.FLOAT_OR_ARR_NUMBA:
    r"""
    Wall frame fluid speed $v_-$ behind the wall

    :param vp: $v_-$, fluid speed $v_+$ behind the wall
    :param ap: $\alpha_+$, strength parameter at the wall
    :param sol_type: Detonation, Deflagration, Hybrid
    :return: $v_+$, fluid speed behind the wall
    """
    # TODO: add support for having both arguments as arrays
    # sol_type is a string enum, which would complicate the use of numba.guvectorize
    if isinstance(vp, numba.types.Float) and isinstance(ap, numba.types.Float):
        return _v_minus_scalar
    if isinstance(vp, numba.types.Array) != isinstance(ap, numba.types.Array):
        return _v_minus_arr
    if isinstance(vp, float) and isinstance(ap, float):
        return _v_minus_scalar(vp, ap, sol_type)
    if isinstance(vp, np.ndarray) != isinstance(ap, np.ndarray):
        return _v_minus_arr(vp, ap, sol_type)
    raise TypeError(f"Unknown argument types: vp = {type(vp)}, ap = {type(ap)}")


@numba.njit
def fluid_speeds_at_wall(
        v_wall: float,
        alpha_p: th.FLOAT_OR_ARR,
        sol_type: SolutionType) -> tp.Tuple[float, float, float, float]:
    """
    Solves fluid speed boundary conditions at the wall.
    Fluid speed vf? just behind (?=m) and just ahead (?=p) of wall,
    in wall (_w) and plasma (_p) frames.
    :return vfp_w, vfm_w, vfp_p, vfm_p
    """
    if v_wall <= 1:
        # print( "max_speed_deflag(alpha_p)= ", max_speed_deflag(alpha_p))
        #     if v_wall < max_speed_deflag(alpha_p) and v_wall <= cs and alpha_p <= 1/3.:
        if sol_type == SolutionType.SUB_DEF.value:
            vfm_w = v_wall  # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(v_wall, alpha_p, sol_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        elif sol_type == SolutionType.HYBRID.value:
            vfm_w = const.CS0  # Fluid velocity just behind the wall in plasma frame (hybrid)
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(const.CS0, alpha_p, sol_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        elif sol_type == SolutionType.DETON.value:
            vfm_w = v_minus(v_wall, alpha_p)  # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_wall  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        else:
            with numba.objmode:
                logger.error("Unknown sol_type: %s", sol_type)
            raise ValueError("Unknown sol_type")
    else:
        with numba.objmode:
            logger.error("v_wall > 1: v_wall = %s", v_wall)
        raise ValueError("v_wall > 1")

    return vfp_w, vfm_w, vfp_p, vfm_p


@numba.njit
def enthalpy_ratio(v_m: th.FLOAT_OR_ARR, v_p: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Ratio of enthalpies behind ($w_-$) and ahead $(w_+)$ of a shock or
    transition front, $w_-/w_+$. Uses conservation of momentum in moving frame.
    """
    return relativity.gamma2(v_m) * v_m / (relativity.gamma2(v_p) * v_p)
