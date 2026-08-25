r"""Fluid shell solver for subsonic deflagrations

This solver uses a shooting method, as
"As a shooting method is always required if the speed of sound depends on the temperature."
:gw_pt_ssm:`\ ` p. 35
"""

import logging
import time
import typing as tp

import numpy as np
from scipy.optimize import fsolve, root_scalar

from pttools.bubble.phase import Phase
from pttools.bubble.const import DEFAULT_N_XI, DEFAULT_SOLVER_RTOL, DEFAULT_T_END, THIN_SHELL_T_POINTS_MIN
from pttools.bubble.fluid_base import DEFLAGRATION_NAN, DeflagrationOutput, SolverOutput
from pttools.bubble import integrate
from pttools.bubble.junction import solve_junction, w2_junction
from pttools.bubble import relativity
from pttools.bubble.shock import find_shock_index
from pttools.bubble.shock_bag import v_shock_bag, wm_shock_bag
from pttools.bubble.solution_type import SolutionType, is_surely_detonation
from pttools.bubble import v_minus
from pttools.speedup.solvers import fsolve_vary
import pttools.type_hints as th

if tp.TYPE_CHECKING:
    from pttools.models import Model

logger = logging.getLogger(__name__)


def sound_shell_deflagration(
        model: "Model",
        v_wall: float,
        wn: float,
        w_center: float,
        cs_n: float,
        v_cj: float,
        vp_guess: float | None = None,
        wp_guess: float | None = None,
        t_end: float = DEFAULT_T_END,
        n_xi: int = DEFAULT_N_XI,
        thin_shell_limit: int = THIN_SHELL_T_POINTS_MIN,
        allow_failure: bool = False,
        allow_negative_entropy_flux_change: bool = False,
        warn_if_shock_barely_exists: bool = True) -> DeflagrationOutput:
    """Get the fluid shell profile of a subsonic deflagration"""
    if vp_guess is None or np.isnan(vp_guess) or wp_guess is None or np.isnan(wp_guess):
        # Use bag model as the starting guess

        # alpha_plus_bag = alpha.find_alpha_plus(v_wall, alpha_n, n_xi=const.N_XI_DEFAULT)
        # vp_tilde_bag, vm_tilde_bag, vp_bag, vm_bag = boundary.fluid_speeds_at_wall(
        #     v_wall, alpha_p=alpha_plus_bag, sol_type=SolutionType.SUB_DEF)
        # wp_bag = boundary.w2_junction(vm_tilde_bag, w_center, vp_tilde_bag)
        # vp_tilde_bag, wp_bag = bag.junction_bag(v_wall, w_center, 0, 1, greater_branch=False)

        # The boundary conditions are symmetric with respect to the indices,
        # and can therefore be used with the opposite indices.
        Vp = 1
        Vm = 0
        alpha_minus = 4 * (Vm - Vp) / (3 * w_center)
        vp_tilde_guess = v_minus(vp=v_wall, ap=alpha_minus, sol_type=SolutionType.SUB_DEF)
        vp_guess = -relativity.lorentz(vp_tilde_guess, v_wall)
        wp_guess = w2_junction(v_wall, w_center, vp_tilde_guess)
    else:
        # if vp_guess > v_wall:
        #     logger.warning("Using invalid vp_guess=%s", vp_guess)
        #     vp_guess = 0.9 * v_wall
        vp_tilde_guess = -relativity.lorentz(vp_guess, v_wall)

    invalid_param = None
    if np.isnan(wn) or wn < 0:
        invalid_param = "wn"
    elif np.isnan(w_center) or w_center < 0:
        invalid_param = "w_center"
    elif np.isnan(vp_tilde_guess) or vp_tilde_guess < 0 or vp_tilde_guess > 1:
        invalid_param = "vp_tilde_guess"
    elif np.isnan(vp_guess) or vp_guess < 0:
        invalid_param = "vp_guess"
    elif np.isnan(wp_guess) or wp_guess < 0:
        invalid_param = "wp_guess"

    if invalid_param is not None:
        logger.error(
            f"Invalid parameter: {invalid_param}. Got: "
            f"model={model.label_unicode}, v_wall={v_wall}, wn={wn}, w_center={w_center}, "
            f"vp_guess={vp_guess}, vp_tilde_guess={vp_tilde_guess}, wp_guess={wp_guess}"
        )
        return DEFLAGRATION_NAN

    if wp_guess < wn:
        logger.warning("Using invalid wp_guess=%s", wp_guess)
        wp_guess = 1.1 * wn

    return sound_shell_deflagration_common(
        model,
        v_wall=v_wall,
        vm_tilde=v_wall,
        wn=wn, wm=w_center,
        cs_n=cs_n, v_cj=v_cj,
        vp_tilde_guess=vp_tilde_guess,
        wp_guess=wp_guess,
        sol_type=SolutionType.SUB_DEF,
        t_end=t_end, n_xi=n_xi,
        thin_shell_limit=thin_shell_limit,
        allow_failure=allow_failure,
        allow_negative_entropy_flux_change=allow_negative_entropy_flux_change,
        warn_if_shock_barely_exists=warn_if_shock_barely_exists
    )


def sound_shell_deflagration_common(
        model: "Model",
        v_wall: float,
        vm_tilde: float,
        wn: float, wm: float,
        cs_n: float, v_cj: float,
        vp_tilde_guess: float, wp_guess: float,
        sol_type: SolutionType,
        n_xi: int = DEFAULT_N_XI,
        t_end: float = DEFAULT_T_END,
        thin_shell_limit: int = THIN_SHELL_T_POINTS_MIN,
        allow_failure: bool = False,
        allow_negative_entropy_flux_change: bool = False,
        warn_if_shock_barely_exists: bool = True) -> DeflagrationOutput:
    """Common component of subsonic deflagration and supersonic deflagration (hybrid) fluid shell solvers"""
    if v_wall < 0 or v_wall > 1 or vm_tilde < 0 or vm_tilde > 1 or wn < 0 or wm < 0 or cs_n < 0 or cs_n > 1 \
            or vp_tilde_guess < 0 or vp_tilde_guess > 1 or wp_guess < 0 \
            or is_surely_detonation(v_wall, v_cj):
        logger.error(
            "Invalid starting values: "
            "v_wall=%s, vm_tilde=%s, wn=%s, wm=%s, cs_n=%s, vp_tilde_guess=%s, wp_guess=%s",
            v_wall, vm_tilde, wn, wm, cs_n, vp_tilde_guess, wp_guess
        )
        return DEFLAGRATION_NAN

    # Solve the boundary conditions at the wall
    vp_tilde, wp = solve_junction(
        model, vm_tilde, wm,
        Phase.BROKEN, Phase.SYMMETRIC,
        v2_tilde_guess=vp_tilde_guess,
        w2_guess=wp_guess,
        allow_failure=allow_failure,
        allow_negative_entropy_flux_change=allow_negative_entropy_flux_change
    )
    vp = -relativity.lorentz(vp_tilde, v_wall)

    # Ensure that the junction solver converges to the correct solution
    # print(v_wall, vp, vp_tilde, vp_tilde_guess, wp, wp_guess)
    # if vp < 0:
    #     vp_tilde2, wp2 = boundary.solve_junction(
    #         model, vm_tilde, wm,
    #         Phase.BROKEN, Phase.SYMMETRIC,
    #         v2_tilde_guess=0.1*vp_tilde_guess, w2_guess=5*wp_guess,
    #         allow_failure=allow_failure
    #     )
    #     vp2 = -relativity.lorentz(vp_tilde, v_wall)
    #     if vp2 > 0:
    #         print("SUCCESS")
    #         vp = vp2
    #         vp_tilde = vp_tilde2
    #         wp = wp2
    #     else:
    #         print("FAILURE")
    # print(v_wall, vp, vp_tilde, vp_tilde_guess, wp, wp_guess)

    if vp < 0 or wp < 0:
        logger.error(
            "Junction solver gave an invalid starting point: "
            "vp=%s, wp=%s, vp_tilde=%s for vp_tilde_guess=%s, wp_guess=%s",
            vp, wp, vp_tilde, vp_tilde_guess, wp_guess)
        return DEFLAGRATION_NAN

    # Manual correction for hybrids
    # if sol_type == SolutionType.HYBRID:
    #     # If we are already below the shock velocity, then add a manual correction
    #     vm_shock_tilde, w_shock = solve_shock(
    #         model,
    #         # The fluid before the shock is still
    #         v1_tilde=v_wall,
    #         w1=wn,
    #         csp=cs_n,
    #         backwards=True, warn_if_barely_exists=warn_if_shock_barely_exists
    #     )
    #     vm_shock = relativity.lorentz(v_wall, vm_shock_tilde)
    #     if vm_shock < 0 or vm_shock > 1:
    #         raise RuntimeError(f"Got invalid vm_shock={vm_shock} when attempting to correct a hybrid.")
    #     if vp < vm_shock:
    #         logger.warning(
    #             "vp < v_shock at the wall. Applying manual correction. Got: vp=%s, v_shock=%s",
    #             vp, vm_shock
    #         )
    #         vp = vm_shock + 1e-3
    #         wp = w_shock + 1e-3

    # logger.debug(f"vp_tilde={vp_tilde}, vp={vp}, wp={wp}")

    # Integrate from the wall to the shock
    # pylint: disable=unused-variable
    v, w, xi, t = integrate.fluid_integrate_param(
        v0=vp, w0=wp, xi0=v_wall,
        phase=Phase.SYMMETRIC,
        t_end=-t_end,
        n_xi=n_xi,
        df_dtau_ptr=model.df_dtau_ptr(),
        # method="RK45"
    )
    if np.argmax(xi) == 0:
        logger.error("Deflagration solver gave a detonation-like solution.")
        return DEFLAGRATION_NAN
    i_shock = find_shock_index(
        model,
        v=v, w=w, xi=xi,
        v_wall=v_wall, wn=wn,
        cs_n=cs_n, sol_type=sol_type,
        error_on_failure=False,
        zero_on_failure=True,
        warn_if_barely_exists=warn_if_shock_barely_exists
    )
    if i_shock == 0 or i_shock + 1 >= xi.size:
        logger.error("The shock was not found by the deflagration solver.")
        return DEFLAGRATION_NAN

    attempts = 5
    if i_shock < thin_shell_limit:
        i_shock_step = 20
        t_end2 = t[i_shock + i_shock_step]
        for i in range(attempts):
            # if i_shock >= thin_shell_limit:
            #     break
            # logger.warning(
            #     "The accuracy for locating the shock may not be sufficient, "
            #     "as it was encountered early at i=%s/%s. Adjusting t_end=%s to compensate. Attempt %s/%s",
            #     i_shock, xi.size, t_end2, i+1, attempts
            # )
            v2, w2, xi2, t2 = integrate.fluid_integrate_param(
                v0=vp, w0=wp, xi0=v_wall,
                phase=Phase.SYMMETRIC,
                t_end=t_end2,
                n_xi=n_xi,
                df_dtau_ptr=model.df_dtau_ptr(),
                # method="RK45"
            )
            if np.argmax(xi2) == 0:
                logger.error("Adjusting t_end gave a detonation-like solution. Using the previous solution.")
                break
            i_shock2 = find_shock_index(
                model,
                v=v2, w=w2, xi=xi2,
                v_wall=v_wall, wn=wn,
                cs_n=cs_n, sol_type=sol_type,
                error_on_failure=False,
                zero_on_failure=True,
                warn_if_barely_exists=warn_if_shock_barely_exists
            )
            if i_shock2 == 0 or i_shock2 + i_shock_step >= xi2.size:
                logger.error(
                    "The shock was not found after t_end adjustment at i=%s/%s. Using the previous solution.",
                    i+1, attempts
                )
                break
            i_shock = i_shock2
            v = v2
            w = w2
            xi = xi2
            t = t2

            if i_shock >= thin_shell_limit:
                break
            t_end2 = t[i_shock + i_shock_step]

    if i_shock <= 1:
        logger.error(
            "The shock was not found for v_wall=%s despite %s t_end adjustments.",
            v_wall, attempts
        )
        return DEFLAGRATION_NAN

    v = v[:i_shock]
    w = w[:i_shock]
    xi = xi[:i_shock]

    xi_sh = xi[-1]
    vm_sh = v[-1]
    wm_sh = w[-1]
    vm_tilde_sh = relativity.lorentz(xi_sh, vm_sh)
    wn_estimate = w2_junction(vm_tilde_sh, wm_sh, xi_sh)

    vm = relativity.lorentz(vm_tilde, v_wall)
    return v, w, xi, vp, vm, vp_tilde, vm_tilde, xi_sh, vm_sh, vm_tilde_sh, wp, wn_estimate, wm_sh


def sound_shell_deflagration_reverse(
        model: "Model", v_wall: float, wn: float, xi_sh: float,
        t_end: float = DEFAULT_T_END,
        n_xi: int = DEFAULT_N_XI,
        allow_failure: bool = False):
    logger.warning("UNTESTED, will probably produce invalid results")

    if np.isnan(v_wall) or v_wall < 0 or v_wall > 1 or np.isnan(xi_sh) or xi_sh < 0 or xi_sh > 1:
        logger.error(
            "Invalid parameters: v_wall=%s, xi_sh=%s",
            v_wall, xi_sh
        )
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr, np.nan, np.nan

    # Solve boundary conditions at the shock
    vm_sh = v_shock_bag(xi_sh)
    wm_sh = wm_shock_bag(xi_sh, wn)

    # Integrate from the shock to the wall
    logger.info(
        f"Integrating deflagration with v_wall=%s, wn=%s from vm_sh=%s, wm_sh=%s, xi_sh=%s",
        v_wall, wn, vm_sh, wm_sh, xi_sh
    )
    v, w, xi, t = integrate.fluid_integrate_param(
        v0=vm_sh, w0=wm_sh, xi0=xi_sh,
        phase=Phase.SYMMETRIC,
        t_end=t_end,
        n_xi=n_xi,
        df_dtau_ptr=model.df_dtau_ptr(),
        # method="RK45"
    )
    # Trim the integration to the wall
    v = np.flip(v)
    w = np.flip(w)
    xi = np.flip(xi)
    # print(np.array([v, w, xi]).T)
    i_min_xi = np.argmin(xi)
    i_wall = np.argmax(xi[i_min_xi:] >= v_wall) + i_min_xi
    # If the curve goes vertical before xi_wall is reached
    if i_wall == i_min_xi:
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr, np.nan, np.nan
    v = v[i_wall:]
    w = w[i_wall:]
    xi = xi[i_wall:]

    # Solve boundary conditions at the wall
    vp = v[0]
    wp = w[0]
    vp_tilde = -relativity.lorentz(vp, v_wall)
    if np.isnan(vp_tilde) or vp_tilde < 0:
        logger.warning("Got vp_tilde < 0")
        # nan_arr = np.array([np.nan])
        # return nan_arr, nan_arr, nan_arr, np.nan, np.nan

    vm_tilde, wm = solve_junction(
        model, vp_tilde, wp,
        Phase.SYMMETRIC, Phase.BROKEN,
        v2_tilde_guess=v_wall, w2_guess=wp,
        allow_failure=allow_failure
    )
    vm = relativity.lorentz(vm_tilde, v_wall)

    return v, w, xi, wp, wm, vm


def sound_shell_solvable_deflagration_reverse(
        params: th.FloatArr1D,
        model: "Model",
        v_wall: float,
        wn: float,
        t_end: float = DEFAULT_T_END,
        n_xi: int = DEFAULT_N_XI) -> float:
    xi_sh = params[0]
    # pylint: disable=unused-variable
    v, w, xi, vm, wm = sound_shell_deflagration_reverse(
        model, v_wall, wn, xi_sh, t_end=t_end, n_xi=n_xi, allow_failure=True)
    return vm


def sound_shell_solvable_deflagration(
        # params: th.FloatArr1D,
        w_center: float,
        model: "Model", v_wall: float, wn: float, cs_n: float, v_cj: float,
        vp_guess: float, wp_guess: float,
        t_end: float = DEFAULT_T_END,
        n_xi: int = DEFAULT_N_XI,
        thin_shell_limit: int = THIN_SHELL_T_POINTS_MIN) -> float:
    if isinstance(w_center, np.ndarray):
        w_center = w_center[0]
    if np.isnan(w_center) or w_center < 0:
        return np.nan
    # pylint: disable=unused-variable
    v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wn_estimate, wm_sh = sound_shell_deflagration(
        model, v_wall=v_wall, wn=wn, w_center=w_center, cs_n=cs_n, v_cj=v_cj,
        vp_guess=vp_guess, wp_guess=wp_guess, t_end=t_end, n_xi=n_xi, thin_shell_limit=thin_shell_limit,
        allow_failure=True,
        allow_negative_entropy_flux_change=True,
        warn_if_shock_barely_exists=False
    )
    return wn_estimate - wn


def sound_shell_solver_deflagration(
        model: "Model",
        start_time: float,
        v_wall: float, alpha_n: float, wn: float, cs_n: float, v_cj: float, high_alpha_n: bool,
        wm_guess: float, vp_guess: float, wp_guess: float, wn_rtol: float, t_end: float,
        n_xi: int = DEFAULT_N_XI,
        thin_shell_limit: int = THIN_SHELL_T_POINTS_MIN,
        rtol: float = DEFAULT_SOLVER_RTOL,
        allow_failure: bool = False,
        log_high_alpha_n_failures: bool = True) -> SolverOutput:
    """Solve for the fluid shell profile of a subsonic deflagration"""
    if vp_guess > v_wall:
        vp_guess_new = 0.95 * v_wall
        if log_high_alpha_n_failures or not high_alpha_n:
            logger.warning(
                "Invalid vp_guess=%s > v_wall=%s, replacing with vp_guess=%s. "
                "This can occur when v_wall < v_wall_min of the reference data.",
                vp_guess, v_wall, vp_guess_new
            )
        vp_guess = vp_guess_new

    sol = root_scalar(
        sound_shell_solvable_deflagration,
        x0=0.99*wm_guess,
        x1=1.01*wm_guess,
        args=(model, v_wall, wn, cs_n, v_cj, vp_guess, wp_guess, t_end, n_xi, thin_shell_limit),
        rtol=rtol
    )
    wm = sol.root
    solution_found = sol.converged
    reason = sol.flag

    if not solution_found:
        # if not high_alpha_n:
        #     logger.error("FALLBACK")
        sol = fsolve_vary(
            sound_shell_solvable_deflagration,
            np.array([wm_guess]),
            args=(model, v_wall, wn, cs_n, v_cj, vp_guess, wp_guess, t_end, n_xi, thin_shell_limit),
            log_status=log_high_alpha_n_failures or not high_alpha_n,
            xtol=rtol
        )
        wm = sol[0][0]
        solution_found = sol[2] == 1
        reason = sol[3]

    v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wn_estimate, wm_sh = sound_shell_deflagration(
        model, v_wall, wn, wm,
        cs_n=cs_n, v_cj=v_cj,
        vp_guess=vp_guess, wp_guess=wp_guess, t_end=t_end, n_xi=n_xi, thin_shell_limit=thin_shell_limit,
        allow_failure=allow_failure,
        allow_negative_entropy_flux_change=True,
        warn_if_shock_barely_exists=False
    )
    if solution_found and not np.isclose(wn_estimate, wn, rtol=wn_rtol):
        solution_found = False
        reason = f"Result not within rtol={wn_rtol}."
    if not solution_found:
        msg = (
            f"Deflagration solution was not found for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. " +
            ("(as expected) " if high_alpha_n else "") +
            f"Got wn_estimate={wn_estimate} for wn={wn}." +
            f"Reason: {reason} " +
            f"Elapsed: {time.perf_counter() - start_time} s."
        )
        if high_alpha_n:
            if log_high_alpha_n_failures:
                logger.warning(msg)
        else:
            logger.error(msg)
    # print(np.array([v, w, xi]).T)
    # print("wn, xi_sh", wn, xi_sh)

    return v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found


def sound_shell_solver_deflagration_reverse(
        model: "Model",
        start_time: float,
        v_wall: float, alpha_n: float, wn: float,
        t_end: float = DEFAULT_T_END,
        n_xi: int = DEFAULT_N_XI,
        rtol: float = DEFAULT_SOLVER_RTOL) -> SolverOutput:
    # This is arbitrary and should be replaced by a value from the bag model
    xi_sh_guess = 1.1 * np.sqrt(model.cs2_max(wn, Phase.BROKEN))
    sol = fsolve(
        sound_shell_solvable_deflagration_reverse,
        xi_sh_guess,
        args=(model, v_wall, wn, t_end, n_xi),
        full_output=True,
        xtol=rtol
    )
    xi_sh = sol[0][0]
    solution_found = True
    if sol[2] != 1:
        solution_found = False
        logger.error(
            "Deflagration solution was not found for model=%s, v_wall=%s, alpha_n=%s. "
            "Using xi_sh=%s. Reason: %s Elapsed: %s s.",
            model.name, v_wall, alpha_n, xi_sh, sol[3].replace("\n ", ""), time.perf_counter() - start_time
        )
    v, w, xi, wp, wm, vm = sound_shell_deflagration_reverse(model, v_wall, wn, xi_sh, t_end=t_end, n_xi=n_xi)

    return v, w, xi, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, wp, wm, np.nan, solution_found
