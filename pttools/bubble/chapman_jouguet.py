"""
Chapman-Jouguet speed
"""

import typing as tp

import numpy as np
from scipy.optimize import fsolve

from pttools.bubble.boundary import Phase, v_plus
from pttools.bubble.relativity import gamma2
from pttools.models.model import Model


def v_chapman_jouguet(
        alpha_n: float,
        model: Model,
        wn_guess: float = 0.5,
        wm_guess: float = 1,
        extra_output: bool = False) -> tp.Union[float, tp.Tuple[float, float, float]]:
    """Chapman-Jouguet speed

    This is the minimum wall speed for detonations.
    TODO: Not tested yet

    (Based on the handwritten notes by Hindmarsh on 2022-06-15.)
    """

    # Get wn=wp from alpha_n
    def wn_solvable(wn_param):
        return model.theta(wn_param, Phase.SYMMETRIC) - model.theta(wn_param, Phase.BROKEN) - 3/4 * wn_param * alpha_n

    wn_sol = fsolve(wn_solvable, x0=np.ndarray([wn_guess]))
    wn = wn_sol[0]

    # Get wm
    def wm_solvable(wm_param):
        vm = model.cs2(wm_param, Phase.BROKEN)
        ap = model.alpha_plus(wp=wn, wm=wm_param)
        vp = v_plus(vm, ap)
        return wm_param**2 + wn * vp * gamma2(vp) * (vm - 1)

    wm_sol = fsolve(wm_solvable, x0=np.ndarray([wm_guess]))
    wm = wm_sol[0]

    # Compute vp with wp, wm & vm
    vm_cj = model.cs2(wm, Phase.BROKEN)
    ap_cj = model.alpha_plus(wn, wm)
    v_cj = v_plus(vm_cj, ap_cj)
    if extra_output:
        return v_cj, vm_cj, ap_cj
    return v_cj
