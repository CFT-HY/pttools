"""
Chapman-Jouguet speed
"""

import logging
import typing as tp

import numba
import numpy as np
from scipy.optimize import fsolve

from pttools.bubble import const
from pttools.bubble import boundary
from pttools.bubble.boundary import Phase, SolutionType
from pttools.bubble.relativity import gamma2
# This would cause a circular import
# from pttools.models.bag import BagModel
if tp.TYPE_CHECKING:
    from pttools.models.model import Model
import pttools.type_hints as th

logger = logging.getLogger(__name__)


# def gen_wn_solvable(model: "Model", alpha_n: float):
#     def wn_solvable(params: np.ndarray) -> float:
#         r"""This function is zero when $w_n$ corresponds to the given $\alpha_n$"""
#         wn = params[0]
#         # return model.theta(wn, Phase.SYMMETRIC) - model.theta(wn, Phase.BROKEN) - 3/4 * wn * alpha_n
#         return model.alpha_n(wn) - alpha_n
#     return wn_solvable


def wm_solvable(params: np.ndarray, model: "Model", wn: float):
    wm_param = params[0]
    vm2 = model.cs2(wm_param, Phase.BROKEN)
    vm = np.sqrt(vm2)
    ap = model.alpha_plus(wp=wn, wm=wm_param)
    vp = boundary.v_plus(vm, ap, sol_type=SolutionType.DETON)
    # print(f"vm={vm}, ap={ap}, vp={vp}")

    # What was this?
    # return wm_param - wn * vp / (1 - vp**2) * (1 - vm2) / vm

    return wm_param**2 + wn * gamma2(vp) * vp * (vm2 - 1)


# def chapman_jouguet_solvable(params: np.ndarray, model: "Model", wn: float, wm_guess: float):
#     v_wall = params[0]
#     vm_guess = np.sqrt(model.cs2(wm_guess, Phase.BROKEN))
#     _, _, vm, wm = boundary.solve_boundary(
#         v_wall, wn, SolutionType.SUB_DEF, model, vm_guess=vm_guess, wm_guess=wm_guess)
#     return vm - np.sqrt(model.cs2(wm, Phase.BROKEN))
#
#
# def chapman_jouguet_vm_solvable(params: np.ndarray, model: "Model", vp: float, wp: float):
#     """Not useful, as we don't know vp."""
#     vm = params[0]
#     wm = wp * gamma2(vp) * vp / (gamma2(vm) * vm)
#     cs = np.sqrt(model.cs2(wm, Phase.BROKEN))
#     return cs - vm


def wm_vw_solvable(params: np.ndarray, model: "Model", vp: float, wp: float):
    r"""$$\Delta_\text{junc1}(w_-)$$ for detonations"""
    wm = params[0]
    vm = boundary.v_minus(vp, model.alpha_plus(wp, wm), SolutionType.DETON)
    return boundary.junction_condition_deviation1(vp, wp, vm, wm)


def wm_vw(wm_guess: float, model: "Model", vp: float, wp: float):
    """$$w_-(v_w)$$"""
    sol = fsolve(wm_vw_solvable, x0=np.array([wm_guess]), args=(model, vp, wp), full_output=True)
    wm = sol[0][0]
    if sol[2] != 1:
        logger.error(
            f"wm(vw) solution was not found for model={model.name}, vp={vp}, wp={wp}, wm_guess={wm_guess}. "
            f"Using wm(vw)={wm}. "
            f"Reason: {sol[3]}"
        )
    return wm


def v_chapman_jouguet_solvable(params: np.ndarray, model: "Model", wp: float, wm_guess: float = None):
    vp = params[0]
    # If a guess is not provided, use the bag model value.
    wm_guess = boundary.wm_junction(vp, wp, const.CS0) if wm_guess is None else wm_guess
    wm = wm_vw(wm_guess, model, vp, wp)
    vm = boundary.v_minus(vp, model.alpha_plus(wp, wm))
    cs = model.cs2(wm, Phase.BROKEN)
    return vm - cs


def v_chapman_jouguet_new(
        model: "Model",
        alpha_n: float,
        wn_guess: float = 1,
        wm_guess: float = 1,
        extra_output: bool = False,
        analytical: bool = True) -> tp.Union[float, tp.Tuple[float, float, float]]:
    if analytical and model.DEFAULT_NAME == "bag":
        return v_chapman_jouguet_bag(alpha_plus=alpha_n)

    wn = model.w_n(alpha_n, wn_guess=wn_guess)
    v_cj_guess = v_chapman_jouguet_bag(alpha_plus=alpha_n)
    sol = fsolve(
        v_chapman_jouguet_solvable,
        x0=np.array([v_cj_guess]),
        args=(model, wn),
        full_output=True
    )
    v_cj = sol[0][0]
    if sol[2] != 1:
        logger.error(
            f"v_cj solution was not found for alpha_n={alpha_n}, model={model.name}, wn_guess={wn_guess}. "
            f"Using v_cj={v_cj}. "
            f"Reason: {sol[3]}"
        )
    return v_cj


# def v_chapman_jouguet_old2(
#         model: "Model",
#         alpha_n: float,
#         wn_guess: float = 1,
#         wm_guess: float = 1,
#         extra_output: bool = False,
#         analytical: bool = True) -> tp.Union[float, tp.Tuple[float, float, float]]:
#     if analytical and model.DEFAULT_NAME == "bag":
#         return v_chapman_jouguet_bag(alpha_plus=alpha_n)
#
#     wn = model.w_n(alpha_n, wn_guess=wn_guess)
#     vm_guess = model.cs2(wm_guess, Phase.BROKEN)
#     vm = fsolve(chapman_jouguet_vm_solvable, x0=np.array([vm_guess]), args=(model, vp, wn))


# def v_chapman_jouguet_old2(
#         model: "Model",
#         alpha_n: float,
#         wn_guess: float = 1,
#         wm_guess: float = 1,
#         extra_output: bool = False,
#         analytical: bool = True) -> tp.Union[float, tp.Tuple[float, float, float]]:
#     if analytical and model.DEFAULT_NAME == "bag":
#         return v_chapman_jouguet_bag(alpha_plus=alpha_n)
#
#     v_cj_guess = 0.5
#     # v_cj_guess = v_chapman_jouguet_old(model, alpha_n)
#     # return v_cj_guess
#
#     wn = model.w_n(alpha_n, wn_guess=wn_guess)
#     sol = fsolve(chapman_jouguet_solvable, x0=np.array([v_cj_guess]), args=(model, wn, wm_guess), full_output=True)
#     v_cj = sol[0][0]
#     if sol[2] != 1:
#         logger.error(
#             f"v_cj solution was not found for alpha_n={alpha_n}, model={model.name}, wn_guess={wn_guess}. "
#             f"Using v_cj={v_cj}. "
#             f"Reason: {sol[3]}"
#         )
#     return v_cj


def v_chapman_jouguet(
        model: "Model",
        alpha_n: float,
        wn_guess: float = 1,
        wm_guess: float = 2,
        extra_output: bool = False,
        analytical: bool = True) -> tp.Union[float, tp.Tuple[float, float, float]]:
    """Chapman-Jouguet speed

    This is the minimum wall speed for detonations.
    This produces incorrect results for some reason.

    (Based on the handwritten notes by Hindmarsh on 2022-06-15.)
    """
    if analytical and model.DEFAULT_NAME == "bag":
        return v_chapman_jouguet_bag(alpha_plus=alpha_n)

    # wn_sol = fsolve(gen_wn_solvable(model, alpha_n), x0=np.array([wn_guess]), full_output=True)
    # wn: float = wn_sol[0][0]
    # if wn_sol[2] != 1:
    #     logger.error(
    #         f"w_n solution was not found for alpha_n={alpha_n}, model={model.name}, wn_guess={wn_guess}. "
    #         f"Using w_n={wn}. "
    #         f"Reason: {wn_sol[3]}")

    wn = model.w_n(alpha_n, wn_guess=wn_guess)

    # Get wm
    # For detonations wn = wp

    wm_sol = fsolve(wm_solvable, x0=np.array([wm_guess]), args=(model, wn), full_output=True)
    wm: float = wm_sol[0][0]
    if wm_sol[2] != 1:
        logger.error(
            f"w_- solution was not found for alpha_n={alpha_n}, model={model.name}, wm_guess={wm_guess}. "
            f"Using w_-={wm}. "
            f"Reason: {wm_sol[3]}")

    # Compute vp with wp, wm & vm
    vm_cj2 = model.cs2(wm, Phase.BROKEN)
    vm_cj = np.sqrt(vm_cj2)
    ap_cj = model.alpha_plus(wn, wm)
    v_cj = boundary.v_plus(vm_cj, ap_cj, sol_type=SolutionType.DETON)
    if extra_output:
        return v_cj, vm_cj, ap_cj
    return v_cj


@numba.njit
def v_chapman_jouguet_bag(alpha_plus: th.FloatOrArr) -> th.FloatOrArr:
    r"""Chapman-Jouguet speed for the bag model

    $\alpha_n$ can be given instead of $\alpha_+$, as
    "The two definitions of the transition strength coincide
    only in the case of detonations within the bag model."
    :notes:` \` p. 40

    $$v_{CJ}(\alpha_+) = \frac{1}{\sqrt{3}} \frac{1 + \sqrt{\alpha_+ + 3 \alpha_+^2}}{1 + \alpha_+}$$
    It should be noted that $v_{CJ} \in [0, 1] \forall \alpha_+ >= 0$.
    """
    return 1/np.sqrt(3) * (1 + np.sqrt(alpha_plus + 3*alpha_plus**2)) / (1 + alpha_plus)
