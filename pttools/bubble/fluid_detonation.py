"""Fluid shell solver for detonations"""

import logging
import typing as tp

import numpy as np

from pttools.bubble.fluid_base import SolverOutput
from pttools.bubble import integrate
from pttools.bubble.junction import solve_junction, w2_junction
from pttools.bubble.junction_bag import fluid_speeds_at_wall_bag
from pttools.bubble.phase import Phase
from pttools.bubble import relativity
from pttools.bubble.solution_type import \
    SolutionType, cannot_be_detonation
from pttools.bubble import trim

if tp.TYPE_CHECKING:
    from pttools.models import Model

logger = logging.getLogger(__name__)


def sound_shell_detonation(
        model: "Model", v_wall: float, alpha_n: float, wn: float, v_cj: float,
        vm_tilde_guess: float, wm_guess: float, t_end: float, n_xi: int) -> SolverOutput:
    """Get the fluid shell profile of a detonation"""
    if cannot_be_detonation(v_wall, v_cj):
        logger.error("Too slow wall speed for a detonation: v_wall=%s, v_cj=%s", v_wall, v_cj)

    # Todo: use analytical ConstCSModel equations for both phases

    # Use bag model as the starting point. This may fail for points near the v_cj curve.
    vp_tilde_bag, vm_tilde_bag, vp_bag, vm_bag = fluid_speeds_at_wall_bag(
        v_wall, alpha_plus=alpha_n, sol_type=SolutionType.DETON)
    wm_bag = w2_junction(v1=vp_tilde_bag, w1=wn, v2=vm_tilde_bag)

    # The bag model works for more points than the pre-generated guesses, so let's use the bag model if we can.
    if not np.isnan(vm_tilde_bag):
        vm_tilde_guess = vm_tilde_bag
    if not np.isnan(wm_bag):
        wm_guess = wm_bag

    # Constant sound speed model vm_tilde
    # csb2_guess = model.cs2(w=wm_guess, phase=Phase.BROKEN)
    # atbn = model.alpha_theta_bar_n(wn)
    # a = v_wall / csb2_guess
    # b = 3*atbn - 1 - v_wall**2 * (1/csb2_guess + 3*atbn)
    # c = v_wall
    # vm_tilde_guess = -b + np.sqrt(b**2 - 4*a*c) / (2*a)

    # This does not work as well
    # if (wm_guess is None or np.isnan(wm_guess)) and not np.isnan(wm_bag):
    #     wm_guess = wm_bag
    # if (vm_tilde_guess is None or np.isnan(vm_tilde_guess)) and not np.isnan(vm_tilde_bag):
    #     vm_tilde_guess = vm_tilde_bag

    # If the guess is not a valid detonation, decrease vm
    v_mu_tilde_guess = np.sqrt(model.cs2(w=wm_guess, phase=Phase.BROKEN))
    v_mu_guess = relativity.lorentz(xi=v_wall, v=v_mu_tilde_guess)
    vm_guess = relativity.lorentz(xi=v_wall, v=vm_tilde_guess)
    if vm_guess > v_mu_guess:
        vm_guess = v_mu_guess
        vm_tilde_guess = relativity.lorentz(xi=v_wall, v=vm_guess)
        # vm_guess2 = relativity.lorentz(xi=v_wall, v=vm_tilde_guess)
        # if vm_guess2 > v_mu_guess or vm_tilde_guess < 0 or vm_tilde_guess > 1:
        #     raise RuntimeError("This should not happen. There is something wrong with the math.")

    # Solve junction conditions
    vm_tilde, wm = solve_junction(
        model,
        v1_tilde=v_wall, w1=wn,
        phase1=Phase.SYMMETRIC, phase2=Phase.BROKEN,
        v2_tilde_guess=vm_tilde_guess, w2_guess=wm_guess,
        w2_min=wn,
        allow_negative_entropy_flux_change=True,
    )
    # Convert to the plasma frame
    vm = relativity.lorentz(v_wall, vm_tilde)
    csb = np.sqrt(model.cs2(w=wm, phase=Phase.BROKEN))
    v_mu = relativity.lorentz(xi=v_wall, v=csb)
    solution_found = vm <= v_mu
    first_attempt_success = solution_found
    if not solution_found:
        vm_tilde2, wm2 = solve_junction(
            model,
            v1_tilde=v_wall, w1=wn,
            phase1=Phase.SYMMETRIC, phase2=Phase.BROKEN,
            v2_tilde_guess=0.7*csb, w2_guess=(wm_guess + wn) / 2,
            w2_min=wn,
            allow_negative_entropy_flux_change=True,
        )
        vm2 = relativity.lorentz(v_wall, vm_tilde2)
        solution_found = vm2 < v_mu
        if solution_found:
            vm_tilde = vm_tilde2
            vm = vm2
            wm = wm2
    if not solution_found:
        csb_lower = relativity.lorentz(xi=v_wall, v=0.5*v_mu)
        vm_tilde2, wm2 = solve_junction(
            model,
            v1_tilde=v_wall, w1=wn,
            phase1=Phase.SYMMETRIC, phase2=Phase.BROKEN,
            v2_tilde_guess=csb_lower, w2_guess=(wm_guess + wn) / 2,
            w2_min=wn,
            allow_negative_entropy_flux_change=True,
        )
        vm2 = relativity.lorentz(v_wall, vm_tilde2)
        solution_found = vm2 < v_mu
        if solution_found:
            vm_tilde = vm_tilde2
            vm = vm2
            wm = wm2
    if not first_attempt_success:
        if solution_found:
            logger.warning(
                "The detonation solver converged to a hybrid solution, but the attempt to fix it succeeded. "
                "v_wall=vp_tilde=%s, alpha_n=%s, "
                "vm=%s, vm_tilde=%s, v_mu=%s, "
                "vm_tilde_guess=%s",
                v_wall, alpha_n, vm, vm_tilde, v_mu, vm_tilde_guess
            )
        else:
            logger.error(
                "The detonation solver converged to a hybrid solution, and the attempt to fix it failed. "
                "v_wall=vp_tilde=%s, alpha_n=%s, "
                "vm=%s, vm_tilde=%s, v_mu=%s, "
                "vm_tilde_guess=%s",
                v_wall, alpha_n, vm, vm_tilde, v_mu, vm_tilde_guess
            )

    v, w, xi, t = integrate.fluid_integrate_param(
        v0=vm, w0=wm, xi0=v_wall,
        phase=Phase.BROKEN,
        t_end=-t_end,
        n_xi=n_xi,
        df_dtau_ptr=model.df_dtau_ptr()
    )
    v, w, xi, t = trim.trim_fluid_wall_to_cs(v, w, xi, t, v_wall, SolutionType.DETON, cs2_fun=model.cs2)

    # The fluid is still ahead of the wall
    vp = 0
    vp_tilde = v_wall

    # Shock quantities are those of the wall
    v_sh = v_wall
    vm_sh = vm
    vm_tilde_sh = vm_tilde

    # Revert the order of points in the arrays for concatenation
    return np.flip(v), np.flip(w), np.flip(xi), \
        vp, vm, vp_tilde, vm_tilde, \
        v_sh, vm_sh, vm_tilde_sh, \
        wn, wm, wm, solution_found
