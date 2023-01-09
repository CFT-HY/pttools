"""Functions for calculating the properties of the bubble boundaries

.. plot:: fig/vm_vp_plane.py
"""

import enum
import logging
import typing as tp

import numba
import numpy as np
from scipy.optimize import fsolve

if tp.TYPE_CHECKING:
    from pttools.models.model import Model
import pttools.type_hints as th
from . import const
from . import relativity

logger = logging.getLogger(__name__)


@enum.unique
class Phase(float, enum.Enum):
    """In general the phase is a scalar variable (a real number), and therefore also these values are floats."""
    # Todo: Move this to a separate file.
    # Do not change these values without also checking the model cs2 functions.
    # These are floats instead of integers to ensure that the Numba functions don't have to be compiled twice.
    SYMMETRIC = 0.
    BROKEN = 1.


@enum.unique
class SolutionType(str, enum.Enum):
    r"""There are three different types of relativistic combustion.
    For further details, please see chapter 7.2 and figure 14
    of :notes:`\ `.

    .. plot:: fig/relativistic_combustion.py
    """
    # Todo: Move this to transition.py
    # Todo: Should the strong and weak branches of the solutions (vplus, vminus signs) be distinquished here?

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
    transition front, $w_-/w_+$.
    Uses conservation of momentum in moving frame.

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

    Bag model only!

    The abbreviations are: fluid speed (vf) just behind (m=minus) and just ahead (p=plus) of wall,
    in wall (_w) and plasma/universe (_p) frames.

    TODO: add a validity check for v_minus

    :param v_wall: $v_\text{wall}$
    :param alpha_p: $\alpha_+$
    :param sol_type: solution type
    :return: $\tilde{v}_+,\tilde{v}_-,v_+,v_-$
    """
    if v_wall > 1:
        with numba.objmode:
            logger.error("v_wall > 1: v_wall = %s", v_wall)
        raise ValueError("v_wall > 1")

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

    return vfp_w, vfm_w, vfp_p, vfm_p


@numba.njit
def _get_phase_scalar(xi: float, v_w: float) -> float:
    return Phase.BROKEN if xi < v_w else Phase.SYMMETRIC


@numba.njit
def _get_phase_arr(xi: np.ndarray, v_w: float) -> np.ndarray:
    ph = np.zeros_like(xi)
    ph[np.where(xi < v_w)] = Phase.BROKEN.value
    return ph


@numba.generated_jit(nopython=True)
def get_phase(xi: th.FloatOrArr, v_w: float) -> th.FloatOrArrNumba:
    r"""
    Returns array indicating phase of system.
    in symmetric phase $(\xi > v_w)$, phase = 0
    in broken phase $(\xi < v_w)$, phase = 1

    :return: phase
    """
    if isinstance(xi, numba.types.Float):
        return _get_phase_scalar
    if isinstance(xi, numba.types.Array):
        if not xi.ndim:
            return _get_phase_scalar
        return _get_phase_arr
    if isinstance(xi, float):
        return _get_phase_scalar(xi, v_w)
    if isinstance(xi, np.ndarray):
        return _get_phase_arr(xi, v_w)
    raise TypeError(f"Unknown type for {type(xi)}")


def junction_conditions_deviation(vp: th.FloatOrArr, vm: th.FloatOrArr, ap: th.FloatOrArr) -> th.FloatOrArr:
    r"""Deviation from the combined junction conditions
    $$\Delta = \left( \frac{1}{\tilde{v}_-} + 3\tilde{v}_- \right) \tilde{v}_+ - 3(1 + \alpha_+) \tilde{v}_+^2 - \alpha_+ + 1$$
    """
    dev = (1/vm + 3*vm)*vp - 3*(1 + ap)*vp**2 + 3*ap - 1
    if not np.allclose(dev, 0):
        if np.isscalar(dev):
            logger.error(f"Non-zero deviation from junction conditions: {dev} for vp={vp}, vm={vm}, ap={ap}")
        else:
            logger.error(f"Non-zero deviation from junction conditions")
    return dev


def junction_conditions_solvable(params: np.ndarray, model: "Model", v1: float, w1: float, phase1: float, phase2: float):
    """Get the deviation from both boundary conditions simultaneously."""
    v2 = params[0]
    w2 = params[1]
    return np.array([
        junction_condition_deviation1(v1, w1, v2, w2),
        junction_condition_deviation2(
            v1, w1, model.p(w1, phase1),
            v2, w2, model.p(w2, phase2)
        )
    ])


@numba.njit
def junction_condition_deviation1(
        v1: th.FloatOrArr, w1: th.FloatOrArr,
        v2: th.FloatOrArr, w2: th.FloatOrArr) -> th.FloatOrArr:
    r"""Deviation from the first junction condition
    $$w_- \tilde{\gamma}_-^2 \tilde{v}_- - w_+ \tilde{\gamma}_-^2 \tilde{v}_+$$
    :notes:`\ `, eq. 7.22
    """
    return w1 * relativity.gamma2(v1) * v1 - w2 * relativity.gamma2(v2) * v2


@numba.njit
def junction_condition_deviation2(
        v1: th.FloatOrArr, w1: th.FloatOrArr, p1: th.FloatOrArr,
        v2: th.FloatOrArr, w2: th.FloatOrArr, p2: th.FloatOrArr
    ):
    # Todo: This docstring causes the error "ERROR: Unknown target name: "p"" with Sphinx.
    r"""Deviation from the second junction condition
    $$w_1 \tilde{\gamma}_1^2 \tilde{v}_1^2 + p_1 - w_2 \tilde{\gamma}_2^2 \tilde{v}_2^2 - p_2$$
    :notes:`\ `, eq. 7.22
    """
    return w1 * relativity.gamma2(v1) * v1**2 + p1 - w2 * relativity.gamma2(v2) * v2**2 - p2


def solve_junction(
        model: "Model",
        v1: float,
        w1: float,
        phase1: Phase,
        phase2: Phase,
        v2_guess: float,
        w2_guess: float,
        allow_failure: bool = False) -> tp.Tuple[float, float]:
    """Model-independent junction condition solver
    Velocities are in the wall frame!
    """
    if np.isclose(v1, 0) or np.isclose(v1, 1) \
            or np.isclose(v2_guess, 0) or np.isclose(v2_guess, 1) \
            or np.isclose(w1, 0) or np.isclose(w2_guess, 0):
        logger.warning(
            "Invalid input for junction solver. "
            f"Got: v1={v1}, w1={w1}, v2_guess={v2_guess}, w2_guess={w2_guess}")
        return np.nan, np.nan

    # if w2_guess is None:
    #     from . import chapman_jouguet
    #     w2_guess = 0.5*chapman_jouguet.wm_chapman_jouguet(model, w1)
    # if v2_guess is None:
    #     v2_guess = 0.5*np.sqrt(model.cs2(w2_guess, Phase.BROKEN))
    # w2_guess = 1.05*w1
    # v2_guess = 0.05
    # vp, vm, vp_tilde, vm_tilde = fluid_speeds_at_wall(v1, alpha_p=model.alpha_n(w1), sol_type=SolutionType.DETON)
    # v2_guess = vm_tilde
    # w2_guess = 1.234 * w1
    # v2_guess = 0.818
    # print("v2_guess, w2_guess, w1:", v2_guess, w2_guess, w1)
    # v2_guess = 0.1
    # wn = 1 -> wm = 1.25
    # Enthalpy ratio wm = 1.234 wn

    sol = fsolve(
        junction_conditions_solvable,
        x0=np.array([v2_guess, w2_guess]),
        args=(model, v1, w1, phase1, phase2),
        full_output=True
    )
    v2 = sol[0][0]
    w2 = sol[0][1]
    if sol[2] != 1:
        msg = \
            f"Boundary solution was not found for v1={v1}, w1={w1}, model={model.name}. " + \
            f"Using v2={v2}, w2={w2}. " + \
            ("" if (0 < v2 < 1) else "This is unphysical! ") + \
            f"Reason: {sol[3]}"
        logger.error(msg)
        if not allow_failure:
            return np.nan, np.nan
            # logger.error("ERROR")
            # raise ValueError(msg)

    devs = junction_conditions_solvable(np.array([v2, w2]), model, v1, w1, phase1, phase2)
    # print(f"v1w={v1}, v2w={v2}, w1={w1}, w2={w2}, dev={devs}")
    if not np.allclose(devs, np.zeros(2)):
        logger.error("The boundary solver gave a solution that deviates from the boundary conditions with: %s", devs)
    return v2, w2


@numba.njit
def _v_minus_scalar(
        vp: float,
        ap: float,
        sol_type: SolutionType,
        strong_branch: bool,
        debug: bool) -> float:
    # Fluid must flow through the wall from the outside to the inside of the bubble.
    if vp < 0:
        return np.nan
    # This has probably been written like this for numerical stability
    vp2 = vp ** 2
    y = vp2 + 1. / 3.
    z = (y - ap * (1. - vp2))
    x = (4. / 3.) * vp2
    sqrt_arg = z**2 - x

    # Way 2
    # x = (1 + ap)*vp + (1 - 3*ap)/(3*vp)
    # sqrt_arg = x**2 - 4/3

    if debug and sqrt_arg < 0:
        with numba.objmode:
            logger.error("Cannot compute vm, got imaginary result with: vp=%s, ap=%s in sqrt_arg=%s", vp, ap, sqrt_arg)
        return np.nan

    # Finding the solution type automatically does not work in the general case
    # if sol_type is None:
    #     b = 1. if vp < 1/np.sqrt(3) else -1
    # else:

    b = 1. if sol_type == SolutionType.DETON.value else -1
    c = -1 if strong_branch else 1
    return (0.5 / vp) * (z + b * c * np.sqrt(sqrt_arg))

    # Way 2
    # return 0.5 * (x + b*np.sqrt(sqrt_arg))

    # Handling of complex return values for scalars
    # if np.imag(ret):
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_minus. This is deprecated. "
    #             "Check the types of the arguments.")
    #     return np.nan
    # return ret


@numba.njit(parallel=True)
def _v_minus_arr(
        vp: np.ndarray,
        ap: float,
        sol_type: SolutionType,
        strong_branch: bool,
        debug: bool) -> np.ndarray:
    ret = np.empty_like(vp)
    for i in numba.prange(vp.size):
        ret[i] = _v_minus_scalar(vp[i], ap, sol_type, strong_branch, debug)
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
        sol_type: SolutionType = SolutionType.DETON,
        strong_branch: bool = False,
        debug: bool = False) -> th.FloatOrArrNumba:
    r"""
    Fluid speed $\tilde{v}_-$ behind the wall in the wall frame
    $$\tilde{v}_- = \frac{1}{2} \left[
    \left( (1 + \alpha_+)\tilde{v}_+ + \frac{1 - 3\alpha_+}{3 \tilde{v}_+} \right)
    \pm
    \sqrt{ \left( (1 + \alpha_+)\tilde{v}_+ + \frac{1 - 3\alpha_+}{3 \tilde{v}_+} \right)^2 - \frac{4}{3} }
    \right]$$
    :gw_pt_ssm:`\ `, eq. B.7

    Positive sign is for detonations,
    which corresponds to $\tilde{v}_+ < \frac{1}{\sqrt{3}}$ in the bag model.
    TODO Check that this is actually the case.

    :param vp: $\tilde{v}_+$, fluid speed ahead of the wall
    :param ap: $\alpha_+$, strength parameter at the wall
    :param sol_type: Detonation, Deflagration, Hybrid (assumed detonation if not given)
    :return: $\tilde{v}_-$, fluid speed behind the wall
    """
    # TODO: add support for having both arguments as arrays
    if isinstance(vp, numba.types.Float):
        return _v_minus_scalar
    if isinstance(vp, numba.types.Array):
        return _v_minus_arr
    if isinstance(vp, float):
        return _v_minus_scalar(vp, ap, sol_type, strong_branch, debug)
    if isinstance(vp, np.ndarray):
        return _v_minus_arr(vp, ap, sol_type, strong_branch, debug)
    raise TypeError(f"Unknown argument types: vp = {type(vp)}, ap = {type(ap)}")


@numba.njit
def _v_plus_scalar(vm: float, ap: float, sol_type: SolutionType, debug: bool) -> float:
    x = vm + 1. / (3 * vm)
    # Finding the SolutionType automatically does not work in the general case
    # if sol_type is None:
    #     b = 1. if vm > 1/np.sqrt(3) else -1.
    # else:
    b = 1. if sol_type == SolutionType.DETON.value else -1.
    # Fluid must flow through the wall from the outside to the inside of the bubble.
    if b == -1 and ap > 1/3:
        if debug:
            with numba.objmode:
                logger.error("v_plus would be negative for a deflagration with ap > 1/3, got ap=%s", ap)
        return np.nan

    return (0.5 / (1 + ap)) * (x + b * np.sqrt(x ** 2 + 4. * ap ** 2 + (8. / 3.) * ap - (4. / 3.)))
    # if vp < 0:
    #     with numba.objmode:
    #         logger.error(
    #             f"Cannot compute v_plus, got negative result: {vp}. "
    #             "THIS SHOULD NOT HAPPEN. Earlier checks should have caught this."
    #         )
    #     return np.nan
    # return vp

    # Handling of complex return values for scalars
    # if np.imag(ret):
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_plus. This is deprecated. "
    #             "Check the types of the arguments.")
    #     return np.nan
    # return ret


@numba.njit(parallel=True)
def _v_plus_arr(vm: np.ndarray, ap: float, sol_type: SolutionType, debug: bool) -> np.ndarray:
    ret = np.empty_like(vm)
    for i in numba.prange(vm.size):
        ret[i] = _v_plus_scalar(vm[i], ap, sol_type, debug)
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
def v_plus(vm: th.FloatOrArr, ap: float, sol_type: SolutionType, debug: bool = True) -> th.FloatOrArrNumba:
    r"""
    Fluid speed $\tilde{v}_+$ ahead of the wall in the wall frame
    $$\tilde{v}_+ = \frac{1}{2(1 + \alpha_+)}
    \left[
    \left( \frac{1}{3 \tilde{v}_-} + \tilde{v}_- \right)
    \pm
    \sqrt{ \left( \frac{1}{3\tilde{v}_-} - \tilde{v}_- \right)^2 + 4\alpha_+^2 + \frac{8}{3} \alpha_+}
    \right]$$
    :gw_pt_ssm:`\ `, eq. B.6,
    :notes:`\ `, eq. 7.27.
    The equations in both sources are equivalent by moving a factor of 2.

    Positive sign is for detonations,
    which corresponds to $\tilde{v}_- > \frac{1}{\sqrt{3}}$ in the bag model.

    :param vm: $\tilde{v}_-$, fluid speed behind the wall
    :param ap: $\alpha_+$, strength parameter at the wall
    :param sol_type: Detonation, Deflagration, Hybrid
    :return: $\tilde{v}_+$, fluid speed ahead of the wall
    """
    # TODO: add support for having both arguments as arrays
    if isinstance(vm, numba.types.Float):
        return _v_plus_scalar
    if isinstance(vm, numba.types.Array):
        return _v_plus_arr
    if isinstance(vm, float):
        return _v_plus_scalar(vm, ap, sol_type, debug)
    if isinstance(vm, np.ndarray):
        return _v_plus_arr(vm, ap, sol_type, debug)
    raise TypeError(f"Unknown argument types: vm = {type(vm)}, ap = {type(ap)}")


def v_plus_limit(ap: th.FloatOrArr, sol_type: SolutionType) -> th.FloatOrArr:
    r"""Limit for the values that $\tilde{v}_+$ can have.

    TODO this is the Chapman-Jouguet speed, not a separate limit!

    $$\frac{1}{1+\alpha_+} \left( \frac{1}{\sqrt{3}} \pm \sqrt{\alpha_+ ( \alpha_+ + \frac{2}{3})} \right)
    """
    b = 1 if sol_type == SolutionType.DETON.value else -1
    return 1/(1 + ap) * (1/np.sqrt(3) + b*np.sqrt(ap*(ap+2/3)))


def v_plus_off_limits(vp: float, ap: float, sol_type: SolutionType):
    if sol_type == SolutionType.DETON.value:
        return vp < v_plus_limit(ap, sol_type)
    return vp > v_plus_limit(ap, sol_type)


def w2_junction(v1: th.FloatOrArr, w1: th.FloatOrArr, v2: th.FloatOrArr) -> th.FloatOrArr:
    r"""Get $w_-$ from the junction condition 1
    $$w_1 = w_2 \frac{\tilde{\gamma}_2^2 \tilde{v}_2}{\tilde{\gamma}_1^2 \tilde{v}_1}$$
    :notes:`\ `, eq. 7.22
    """
    wm = w1 * relativity.gamma2(v1) * v1 / (relativity.gamma2(v2) * v2)
    # wm > wp for detonations
    # if wm > wp:
    #     logger.warning(f"wm_junction resulted in wm > wp: vp={vp}, wp={wp}, vm={vm}, wm={wm}")
    return wm
