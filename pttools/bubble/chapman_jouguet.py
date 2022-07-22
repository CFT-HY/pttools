"""
Chapman-Jouguet speed
"""

import logging
import typing as tp

import numpy as np
from scipy.optimize import fsolve

from pttools.bubble.boundary import Phase, SolutionType, v_plus
from pttools.bubble.relativity import gamma2
# This would cause a circular import
# from pttools.models.bag import BagModel
if tp.TYPE_CHECKING:
    from pttools.models.model import Model
import pttools.type_hints as th

logger = logging.getLogger(__name__)


def gen_wn_solvable(model: "Model", alpha_n: float):
    def wn_solvable(params: np.ndarray) -> float:
        r"""This function is zero when $w_n$ corresponds to the given $\alpha_n$"""
        wn = params[0]
        return model.theta(wn, Phase.SYMMETRIC) - model.theta(wn, Phase.BROKEN) - 3/4 * wn * alpha_n
    return wn_solvable


def v_chapman_jouguet(
        model: "Model",
        alpha_n: float,
        wn_guess: float = 1,
        wm_guess: float = 0.5,
        extra_output: bool = False,
        bag_analytical: bool = True) -> tp.Union[float, tp.Tuple[float, float, float]]:
    """Chapman-Jouguet speed

    This is the minimum wall speed for detonations.
    TODO: Not tested yet

    (Based on the handwritten notes by Hindmarsh on 2022-06-15.)
    """
    # if bag_analytical and isinstance(model, BagModel):
    if bag_analytical and model.DEFAULT_NAME == "bag":
        return v_chapman_jouguet_bag(alpha_plus=alpha_n)

    wn_sol = fsolve(gen_wn_solvable(model, alpha_n), x0=np.array([wn_guess]), full_output=True)
    wn: float = wn_sol[0][0]
    if wn_sol[2] != 1:
        logger.error(
            f"w_n solution was not found for alpha_n={alpha_n}, model={model.name}, wn_guess={wn_guess}. "
            f"Using w_n={wn}. "
            f"Reason: {wn_sol[3]}")

    # Get wm
    # For detonations wn = wp
    def wm_solvable(params: np.ndarray):
        wm_param = params[0]
        vm = model.cs2(wm_param, Phase.BROKEN)
        ap = model.alpha_plus(wp=wn, wm=wm_param)
        vp = v_plus(vm, ap, sol_type=SolutionType.DETON)
        return wm_param**2 + wn * vp * gamma2(vp) * (vm - 1)

    wm_sol = fsolve(wm_solvable, x0=np.array([wm_guess]), full_output=True)
    wm: float = wm_sol[0][0]
    if wm_sol[2] != 1:
        logger.error(
            f"w_- solution was not found for alpha_n={alpha_n}, model={model.name}, wm_guess={wm_guess}. "
            f"Using w_-={wm}. "
            f"Reason: {wm_sol[3]}")

    # Compute vp with wp, wm & vm
    vm_cj = model.cs2(wm, Phase.BROKEN)
    ap_cj = model.alpha_plus(wn, wm)
    v_cj = v_plus(vm_cj, ap_cj, sol_type=SolutionType.DETON)
    if extra_output:
        return v_cj, vm_cj, ap_cj
    return v_cj


def v_chapman_jouguet_bag(alpha_plus: th.FloatOrArr) -> th.FloatOrArr:
    r"""Chapman-Jouguet speed for the bag model

    $\alpha_n$ can be given instead of $\alpha_+$, as
    "The two definitions of the transition strength coincide
    only in the case of detonations within the bag model."
    :notes:` \` p. 40

    $$v_{CJ}(\alpha_+) = \frac{1}{\sqrt{3}} \frac{1 + \sqrt{\alpha_+ + 3 \alpha_+^2}}{1 + \alpha_+}$$
    """
    return 1/np.sqrt(3) * (1 + np.sqrt(alpha_plus + 3*alpha_plus**2) / (1 + alpha_plus))
