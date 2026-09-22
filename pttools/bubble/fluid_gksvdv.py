r"""Fluid shell solver based on :giese_2021:`\ `."""

import logging
import time
import typing as tp

import numpy as np

from pttools.bubble import chapman_jouguet, relativity
from pttools.bubble.fluid_base import GenericSolverOutput
from pttools.bubble.gksvdv.gksvdv21 import kappaNuMuModel
from pttools.bubble.phase import Phase
from pttools.bubble.solution_type import SolutionType
from pttools.speedup import NAN_ARR

if tp.TYPE_CHECKING:
    from pttools.models import Model

logger = logging.getLogger(__name__)


def sound_shell_gksvdv(
            model: "Model",
            v_wall: float,
            alpha_n: float,
            wn: float | None = None,
            wn_guess: float | None = None,
            wm_guess: float | None = None,
        ) -> GenericSolverOutput:
    r"""Fluid shell solver based on :giese_2021:`\ `."""
    start_time = time.perf_counter()

    if wn is None or np.isnan(wn):
        wn = model.wn(alpha_n, wn_guess=wn_guess)
    if wm_guess is None or np.isnan(wm_guess):
        wm_guess = 1.

    try:
        # The velocities returned by the solver are in the wall frame
        kappa_theta_bar_n, v, wow, xi, mode, vp_tilde, vm_tilde = kappaNuMuModel(
            cs2b=model.cs2(wm_guess, Phase.BROKEN),
            cs2s=model.cs2(wn, Phase.SYMMETRIC),
            al=model.alpha_theta_bar_n_from_alpha_n(alpha_n=alpha_n, wn=wn),
            vw=v_wall
           )
    except ValueError:
        return NAN_ARR, NAN_ARR, NAN_ARR, SolutionType.ERROR, \
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
            True, time.perf_counter() - start_time

    if mode == 0:
        sol_type = SolutionType.SUB_DEF
    elif mode == 1:
        sol_type = SolutionType.HYBRID
    elif mode == 2:
        sol_type = SolutionType.DETON
    else:
        raise ValueError("Got invalid mode from Giese solver:", mode)
    w = wow * wn

    # Velocities in the plasma frame
    vp: float = relativity.lorentz(xi=v_wall, v=vp_tilde)
    vm: float = relativity.lorentz(xi=v_wall, v=vm_tilde)

    # Shock
    v_sh: float = xi[-3]
    vm_sh: float = v[-3]
    vm_tilde_sh: float = relativity.lorentz(xi=v_sh, v=vm_sh)
    wm_sh: float = w[-3]

    # Enthalpies
    # The fluid velocity in the plasma frame is at its maximum at the wall.
    # In front of the wall, the fluid velocity decreases towards the shock, and behind the wall it decreases
    # towards the center of the bubble, so the maximum is ahead of the wall if the fluid there is moving.
    i_wall = np.argmax(v)
    # For detonations the fluid in front of the wall is at rest, and the maximum is just behind the wall.
    if sol_type == SolutionType.DETON:
        i_wall += 1
    wp: float = w[i_wall]
    wm: float = w[i_wall - 1]

    # Other
    v_cj = chapman_jouguet.v_chapman_jouguet(model, alpha_n=alpha_n, wn=wn, wn_guess=wn_guess)
    solution_found = True
    elapsed = time.perf_counter() - start_time

    return v, w, xi, sol_type, \
        vp, vm, vp_tilde, vm_tilde, \
        v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, v_cj, \
        not solution_found, elapsed
