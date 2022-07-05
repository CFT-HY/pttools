"""Functions for calculating the properties of the bubble boundaries

.. plot:: fig/vm_vp_plane.py
"""

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
class Phase(float, enum.Enum):
    """In general the phase is a scalar variable (a real number), and therefore also these values are floats."""
    # Do not change these values without also checking the model cs2 functions.
    SYMMETRIC = 0.
    BROKEN = 1.


@enum.unique
class SolutionType(str, enum.Enum):
    """There are three different types of relativistic combustion.
    For further details, please see chapter 7.2 and figure 14
    of the :notes:`lecture notes <>`.

    .. plot:: fig/relativistic_combustion.py
    """

    #: In a detonation the fluid outside the bubble is at rest and the wall moves at a supersonic speed.
    DETON = "Detonation"

    #: This value is used to inform, that determining the type of the
    #: relativistic combustion failed.
    ERROR = "Error"

    #: In the hybrid case the wall speed is supersonic and the fluid is moving both ahead and behind the wall.
    HYBRID = "Hybrid"

    #: In a subsonic deflagration the fluid is at rest inside the bubble,
    #: and the wall moves at a subsonic speed.
    SUB_DEF = "Subsonic deflagration"

    #: This value is used, when the type of the relativistic combustion is not yet determined.
    UNKNOWN = "Unknown"


@numba.njit
def enthalpy_ratio(v_m: th.FloatOrArr, v_p: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Ratio of enthalpies behind ($w_-$) and ahead $(w_+)$ of a shock or
    transition front, $w_-/w_+$. Uses conservation of momentum in moving frame.

    $$\frac{\gamma^2 (v_m) v_m}{\gamma^2 (v_p) v_p}$$

    :param v_m: $v_-$
    :param v_p: $v_+$
    :return: enthalpy ratio
    """
    return relativity.gamma2(v_m) * v_m / (relativity.gamma2(v_p) * v_p)


@numba.njit
def fluid_speeds_at_wall(
        v_wall: float,
        alpha_p: th.FloatOrArr,
        sol_type: SolutionType) -> tp.Tuple[float, float, float, float]:
    r"""
    Solves fluid speed boundary conditions at the wall to obtain
    the fluid speeds both in the universe (plasma frame): $v_+$ and $v_+$
    and in the wall frame: $\tilde{v}_+, \tilde{v}_-$.

    The abbreviations are: fluid speed (vf) just behind (m=minus) and just ahead (p=plus) of wall,
    in wall (_w) and plasma/universe (_p) frames.

    TODO: add a validity check for v_minus

    :param v_wall: $v_\text{wall}$
    :param alpha_p: $\alpha_+$
    :param sol_type: solution type
    :return: $v_+,v_-,\tilde{v}_+,\tilde{v}_-$
    """
    if v_wall <= 1:
        # print( "max_speed_deflag(alpha_p)= ", max_speed_deflag(alpha_p))
        #     if v_wall < max_speed_deflag(alpha_p) and v_wall <= cs and alpha_p <= 1/3.:
        if sol_type == SolutionType.SUB_DEF.value:
            # For clarity these are defined here in the same order as returned
            vfp_w = v_plus(v_wall, alpha_p, sol_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfm_w = v_wall  # Fluid velocity just behind the wall in wall frame (v-)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
        elif sol_type == SolutionType.HYBRID.value:
            vfp_w = v_plus(const.CS0, alpha_p, sol_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfm_w = const.CS0  # Fluid velocity just behind the wall in plasma frame (hybrid)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
        elif sol_type == SolutionType.DETON.value:
            vfp_w = v_wall  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfm_w = v_minus(v_wall, alpha_p)  # Fluid velocity just behind the wall in wall frame (v-)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
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
def _v_minus_scalar(vp: float, ap: float, sol_type: SolutionType) -> float:
    vp2 = vp ** 2
    Y = vp2 + 1. / 3.
    Z = (Y - ap * (1. - vp2))
    X = (4. / 3.) * vp2
    b = 1. if sol_type == SolutionType.DETON.value else -1
    return (0.5 / vp) * (Z + b * np.sqrt(Z ** 2 - X))

    # Handling of complex return values for scalars
    # if np.imag(ret):
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_minus. This is deprecated. "
    #             "Check the types of the arguments.")
    #     return np.nan
    # return ret


@numba.njit
def _v_minus_arr(vp: np.ndarray, ap: float, sol_type: SolutionType) -> np.ndarray:
    ret = np.empty_like(vp)
    for i, v in enumerate(vp):
        ret[i] = _v_minus_scalar(v, ap, sol_type)
    return ret

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
        vp: th.FloatOrArr,
        ap: float,
        sol_type: SolutionType = SolutionType.DETON) -> th.FloatOrArrNumba:
    r"""
    Wall frame fluid speed $\tilde{v}_-$ behind the wall
    $$\tilde{v}_- = \frac{1}{2} \left[
    \left( (1 + \alpha_+)\tilde{v}_+ + \frac{1 - 3\alpha_+}{3 \tilde{v}_+} \right)
    \pm
    \sqrt{ \left( (1 + \alpha_+)\tilde{v}_+ + \frac{1 - 3\alpha_+}{3 \tilde{v}_+} \right)^2 - \frac{4}{3} }
    \right]$$
    :gw_pt_ssm:`\ `, eq. B.7

    Positive sign is for detonations and negative for others.

    :param vp: $\tilde{v}_+$, fluid speed ahead of the wall
    :param ap: $\alpha_+$, strength parameter at the wall
    :param sol_type: Detonation, Deflagration, Hybrid
    :return: $\tilde{v}_-$, fluid speed behind the wall
    """
    # TODO: add support for having both arguments as arrays
    # sol_type is a string enum, which would complicate the use of numba.guvectorize
    if isinstance(vp, numba.types.Float):
        return _v_minus_scalar
    if isinstance(vp, numba.types.Array):
        return _v_minus_arr
    if isinstance(vp, float):
        return _v_minus_scalar(vp, ap, sol_type)
    if isinstance(vp, np.ndarray):
        return _v_minus_arr(vp, ap, sol_type)
    raise TypeError(f"Unknown argument types: vp = {type(vp)}, ap = {type(ap)}")


@numba.njit
def _v_plus_scalar(vm: float, ap: float, sol_type: SolutionType) -> float:
    X = vm + 1. / (3 * vm)
    b = 1. if sol_type == SolutionType.DETON.value else -1.
    vp = (0.5 / (1 + ap)) * (X + b * np.sqrt(X ** 2 + 4. * ap ** 2 + (8. / 3.) * ap - (4. / 3.)))
    # Fluid must flow through the wall from the outside to the inside of the bubble.
    return vp if vp >= 0 else np.nan

    # Handling of complex return values for scalars
    # if np.imag(ret):
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_plus. This is deprecated. "
    #             "Check the types of the arguments.")
    #     return np.nan
    # return ret


@numba.njit
def _v_plus_arr(vm: np.ndarray, ap: float, sol_type: SolutionType) -> np.ndarray:
    ret = np.empty_like(vm)
    for i, v in enumerate(vm):
        ret[i] = _v_plus_scalar(v, ap, sol_type)
    return ret

    # complex_inds = np.where(np.imag(ret))
    # if np.any(complex_inds):
    #     ret[np.where(np.imag(ret))] = np.nan
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_plus. This is deprecated. "
    #             "Check the types of the arguments.")
    # return np.real(ret)


@numba.generated_jit(nopython=True)
def v_plus(vm: th.FloatOrArr, ap: float, sol_type: SolutionType) -> th.FloatOrArrNumba:
    r"""
    Wall frame fluid speed $\tilde{v}_+$ ahead of the wall
    $$\tilde{v}_+ = \frac{1}{2(1 + \alpha_+)}
    \left[
    \left( \frac{1}{3 \tilde{v}_-} + \tilde{v}_- \right)
    \pm
    \sqrt{ \left( \frac{1}{3\tilde{v}_-} - \tilde{v}_- \right)^2 + 4\alpha_+^2 + \frac{8}{3} \alpha_+}
    \right]$$
    :gw_pt_ssm:`\ `, eq. B.6,
    :notes:`\ `, eq. 7.27.
    The equations in both sources are equivalent by moving a factor of 2.

    Positive sign is for detonations and negative for others.

    :param vm: $\tilde{v}_-$, fluid speed behind the wall
    :param ap: $\alpha_+$, strength parameter at the wall
    :param sol_type: Detonation, Deflagration, Hybrid
    :return: $\tilde{v}_+$, fluid speed ahead of the wall
    """
    # TODO: add support for having both arguments as arrays
    # sol_type is a string enum, which would complicate the use of numba.guvectorize
    if isinstance(vm, numba.types.Float):
        return _v_plus_scalar
    if isinstance(vm, numba.types.Array):
        return _v_plus_arr
    if isinstance(vm, float):
        return _v_plus_scalar(vm, ap, sol_type)
    if isinstance(vm, np.ndarray):
        return _v_plus_arr(vm, ap, sol_type)
    raise TypeError(f"Unknown argument types: vm = {type(vm)}, ap = {type(ap)}")
