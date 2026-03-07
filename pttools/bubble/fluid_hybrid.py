"""Fluid shell solver for hybrids"""

import logging
import time
import typing as tp

import numpy as np
from scipy.optimize import fsolve

from pttools.bubble.fluid_base import DeflagrationOutput, SolverOutput
from pttools.bubble.fluid_sub_def import sound_shell_deflagration_common
from pttools.bubble import integrate
from pttools.bubble.junction import v_plus_hybrid
from pttools.bubble.phase import Phase
from pttools.bubble import relativity
from pttools.bubble.shock import v_shock
from pttools.bubble.solution_type import SolutionType
from pttools.speedup.solvers import fsolve_vary

if tp.TYPE_CHECKING:
    from pttools.models import Model

logger = logging.getLogger(__name__)


def sound_shell_hybrid(
        model: "Model", v_wall: float, wn: float, wm: float, cs_n: float, v_cj: float,
        vp_tilde_guess: float, wp_guess: float, t_end: float, n_xi: int,
        thin_shell_limit: int,
        allow_failure: bool = False,
        allow_negative_entropy_flux_change: bool = False,
        warn_if_shock_barely_exists: bool = True) -> DeflagrationOutput:
    """Get the fluid shell profile of a hybrid"""
    # Exit velocity is at the sound speed
    vm_tilde = np.sqrt(model.cs2(wm, Phase.BROKEN))

    # Simple starting guesses
    if np.isnan(vp_tilde_guess):
        vp_tilde_guess = 0.75 * vm_tilde
    if np.isnan(wp_guess):
        wp_guess = 2*wm

    ret = sound_shell_deflagration_common(
        model,
        v_wall=v_wall,
        vm_tilde=vm_tilde,
        wn=wn, wm=wm,
        cs_n=cs_n, v_cj=v_cj,
        vp_tilde_guess=vp_tilde_guess, wp_guess=wp_guess,
        sol_type=SolutionType.HYBRID,
        t_end=t_end, n_xi=n_xi,
        thin_shell_limit=thin_shell_limit,
        allow_failure=allow_failure,
        allow_negative_entropy_flux_change=allow_negative_entropy_flux_change,
        warn_if_shock_barely_exists=warn_if_shock_barely_exists
    )
    if not np.isnan(ret[4]):
        return ret

    vm = relativity.lorentz(xi=v_wall, v=vm_tilde)
    # Shock velocity at xi_wall
    v_sh_estimate = v_shock(model, wn=wn, xi=v_wall, cs_n=cs_n)
    vp_guess = relativity.lorentz(xi=v_wall, v=vp_tilde_guess)

    # More complex starting guesses
    if np.isnan(vp_tilde_guess) or vp_guess < v_sh_estimate or vp_guess < vm or np.isnan(wp_guess) \
            or wp_guess < wn or wp_guess < wm:
        vp_guess = 1.05 * v_sh_estimate
        vp_tilde_guess = relativity.lorentz(xi=v_wall, v=vp_guess)
        wp_guess = wn + 1.3*np.abs(wm - wn)
        logger.warning(
            "vp_tilde_guess or wp_guess was not provided for the hybrid solver or was invalid. "
            "Using automatic starting guesses. vp_guess=%s, vp_tilde_guess=%s, wp_guess=%s",
            vp_guess, vp_tilde_guess, wp_guess
        )

    ret2 = sound_shell_deflagration_common(
        model,
        v_wall=v_wall,
        vm_tilde=vm_tilde,
        wn=wn, wm=wm,
        cs_n=cs_n, v_cj=v_cj,
        vp_tilde_guess=vp_tilde_guess, wp_guess=wp_guess,
        sol_type=SolutionType.HYBRID,
        t_end=t_end, n_xi=n_xi,
        thin_shell_limit=thin_shell_limit,
        allow_failure=allow_failure,
        allow_negative_entropy_flux_change=allow_negative_entropy_flux_change,
        warn_if_shock_barely_exists=warn_if_shock_barely_exists
    )
    if not np.isnan(ret2[4]):
        return ret2
    return ret


def sound_shell_solvable_hybrid(
        # params: th.FloatArr1D,
        wm: float, model: "Model", v_wall: float, wn: float, cs_n: float, v_cj: float,
        vp_tilde_guess: float, wp_guess: float, t_end: float, n_xi: int, thin_shell_limit: int) -> float:
    if isinstance(wm, np.ndarray):
        wm = wm[0]
    if np.isnan(wm) or wm < 0:
        return np.nan
    # pylint: disable=unused-variable
    v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wn_estimate, wm_sh = sound_shell_hybrid(
        model, v_wall=v_wall, wn=wn, wm=wm,
        cs_n=cs_n, v_cj=v_cj,
        vp_tilde_guess=vp_tilde_guess, wp_guess=wp_guess, t_end=t_end, n_xi=n_xi, thin_shell_limit=thin_shell_limit,
        allow_failure=True,
        allow_negative_entropy_flux_change=True,
        warn_if_shock_barely_exists=False
    )
    diff = wn_estimate - wn
    # logger.debug(
    #     "Hybrid solvable results: wn_target=%s, wn_computed=%s, diff=%s, wm=%s, vp=%s",
    #     wn, wn_estimate, diff, wm, vp
    # )
    return diff


def sound_shell_solver_hybrid(
        model: "Model",
        start_time: float,
        v_wall: float, alpha_n: float, wn: float, cs_n: float, v_cj: float, high_alpha_n: bool,
        vp_tilde_guess: float, wp_guess: float, wm_guess: float, wn_rtol: float, t_end: float, n_xi: int,
        thin_shell_limit: int,
        allow_failure: bool, log_high_alpha_n_failures: bool) -> SolverOutput:
    """Solve for the fluid shell profile of a hybrid"""
    if v_wall >= v_cj:
        raise RuntimeError(f"Invalid v_wall for a hybrid: v_wall={v_wall}, v_cj={v_cj}")

    # This may not work, as we don't know whether the solvable has a different sign at the endpoints.
    # sol = root_scalar(
    #     sound_shell_solvable_hybrid,
    #     x0=0.99*wm_guess,
    #     x1=1.01*wm_guess,
    #     args=(model, v_wall, wn, cs_n, v_cj, vp_tilde_guess, wp_guess, t_end, n_xi, thin_shell_limit)
    # )
    # wm = sol.root
    # solution_found = sol.converged
    # reason = sol.flag

    # if not high_alpha_n:
    #     logger.error("FALLBACK")
    sol = fsolve_vary(
        sound_shell_solvable_hybrid,
        np.array([wm_guess]),
        args=(model, v_wall, wn, cs_n, v_cj, vp_tilde_guess, wp_guess, t_end, n_xi, thin_shell_limit),
        log_status=log_high_alpha_n_failures or not high_alpha_n
    )
    solution_found = sol[2] == 1
    wm = sol[0][0]
    reason = sol[3]

    # If both solvers failed, then adjust the search range for wm
    if not solution_found:
        logger.debug("Entering backup hybrid solver")
        wms = np.linspace(0.3 * wm_guess, 3 * wm_guess, 20)
        vps = np.zeros_like(wms)
        v_sh = v_shock(model, wn=wn, xi=v_wall, cs_n=cs_n)
        for i, wm_i in enumerate(wms):
            vp = v_plus_hybrid(
                model,
                v_wall=v_wall, wm=wm_i,
                vp_tilde_guess=vp_tilde_guess, wp_guess=wp_guess,
                allow_failure=allow_failure,
                allow_negative_entropy_flux_change=allow_failure
            )
            # We must approach the shock curve from above
            if vp > v_sh:
                vps[i] = vp

        valid_wm_inds = np.argwhere(vps != 0)
        valid_wms = wms[valid_wm_inds][:, 0]
        # if valid_wms.size >= 2:
        #     wm_min = wms[valid_wms[0, 0]]
        #     wm_max = wms[valid_wms[-1, 0]]
        #
        #     sol = root_scalar(
        #         sound_shell_solvable_hybrid,
        #         x0=wm_min,
        #         x1=wm_max,
        #         args=(model, v_wall, wn, cs_n, v_cj, vp_tilde_guess, wp_guess, t_end, n_xi, thin_shell_limit)
        #     )
        #     if sol.converged:
        #         solution_found = True
        #         wm = sol.root
        #         reason = sol.flag

        logger.debug("Valid wms: %s, inds: %s", valid_wms, valid_wm_inds)
        for i in range(vps.size):
            if vps[i] == 0:
                continue
            vp_i = vps[i]
            wm_i = wms[i]
            logger.debug("wm=%s, vp=%s", wm_i, vp_i)
            # sol = fsolve(
            #     sound_shell_solvable_hybrid,
            #     np.array([wm_i]),
            #     args=(model, v_wall, wn, cs_n, v_cj, vp_tilde_guess, wp_guess, t_end, n_xi, thin_shell_limit),
            #     # log_status=log_high_alpha_n_failures or not high_alpha_n,
            #     full_output=True
            # )
            # if sol[2] == 1:
            #     solution_found = True
            #     wm = sol[0][0]
            #     reason = sol[3]
            #     break

            sol = fsolve(
                sound_shell_solvable_hybrid,
                x0=wm_i,
                args=(model, v_wall, wn, cs_n, v_cj, vp_tilde_guess, wp_guess, t_end, n_xi, thin_shell_limit),
                full_output=True
            )
            if sol[2] == 1:
                solution_found = True
                wm = sol[0][0]
                reason = sol[3]
                break

        logger.debug(
            "Backup hybrid solver results: solution_found=%s, valid_wms=%s, vps=%s, v_sh=%s",
            solution_found, valid_wms, vps, v_sh
        )

    v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wn_estimate, wm_sh = sound_shell_hybrid(
        model, v_wall, wn, wm,
        cs_n=cs_n, v_cj=v_cj,
        vp_tilde_guess=vp_tilde_guess,
        wp_guess=wp_guess,
        t_end=t_end, n_xi=n_xi,
        thin_shell_limit=thin_shell_limit,
        allow_failure=allow_failure,
        allow_negative_entropy_flux_change=True,
        warn_if_shock_barely_exists=False
    )
    # wp = w[0]
    if solution_found and not np.isclose(wn_estimate, wn, rtol=wn_rtol):
        solution_found = False
        reason = f"Result not within rtol={wn_rtol}."
    if not solution_found:
        msg = (
            f"Hybrid solution was not found for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. " +
            f"Got wn_estimate={wn_estimate} for wn={wn}. " +
            ("(as expected)" if high_alpha_n else "") +
            f"Reason: {reason} " +
            f"Elapsed: {time.perf_counter() - start_time} s."
        )
        if high_alpha_n:
            if log_high_alpha_n_failures:
                logger.warning(msg)
        else:
            logger.error(msg)

    vm = relativity.lorentz(v_wall, np.sqrt(model.cs2(wm, Phase.BROKEN)))
    v_tail, w_tail, xi_tail, t_tail = integrate.fluid_integrate_param(
        vm, wm, v_wall,
        phase=Phase.BROKEN,
        t_end=-t_end,
        n_xi=n_xi,
        df_dtau_ptr=model.df_dtau_ptr()
    )
    v = np.concatenate((np.flip(v_tail), v))
    w = np.concatenate((np.flip(w_tail), w))
    xi = np.concatenate((np.flip(xi_tail), xi))

    return v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found
