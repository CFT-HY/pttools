r"""Solver for the fluid velocity profile of a bubble"""

import logging
import time
import typing as tp

import numpy as np
from scipy.optimize import fsolve, root_scalar

from pttools.speedup.solvers import fsolve_vary
from . import alpha
from . import boundary
from .boundary import Phase, SolutionType
from . import chapman_jouguet
from . import const
from . import integrate
from . import fluid_bag
from . import fluid_reference
from . import props
from . import relativity
from . import shock
from . import transition
from . import trim
if tp.TYPE_CHECKING:
    from pttools.models import Model

logger = logging.getLogger(__name__)

# The output consists of:
# v, w, xi
# vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh
# solution_found
SolverOutput = tp.Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    float, float, float, float, float, float, float, float, float, float,
    bool
]
DeflagrationOutput = tp.Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    float, float, float, float, float, float, float, float, float, float
]
DEFLAGRATION_NAN: DeflagrationOutput = \
    const.nan_arr, const.nan_arr, const.nan_arr, \
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def fluid_shell_deflagration(
        model: "Model",
        v_wall: float, wn: float, w_center: float,
        cs_n: float,
        vp_guess: float = None, wp_guess: float = None,
        allow_failure: bool = False,
        warn_if_shock_barely_exists: bool = True) -> DeflagrationOutput:
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
        alpha_minus = 4*(Vm - Vp)/(3*w_center)
        vp_tilde_guess = boundary.v_minus(vp=v_wall, ap=alpha_minus, sol_type=SolutionType.SUB_DEF)
        vp_guess = -relativity.lorentz(vp_tilde_guess, v_wall)
        wp_guess = boundary.w2_junction(v_wall, w_center, vp_tilde_guess)
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

    return fluid_shell_deflagration_common(
        model,
        v_wall=v_wall,
        vm_tilde=v_wall,
        wn=wn, wm=w_center,
        cs_n=cs_n,
        vp_tilde_guess=vp_tilde_guess, wp_guess=wp_guess,
        sol_type=SolutionType.SUB_DEF,
        allow_failure=allow_failure,
        warn_if_shock_barely_exists=warn_if_shock_barely_exists
    )


def fluid_shell_deflagration_common(
        model: "Model",
        v_wall: float,
        vm_tilde: float,
        wn: float, wm: float,
        cs_n: float,
        vp_tilde_guess: float, wp_guess: float,
        sol_type: SolutionType,
        allow_failure: bool,
        warn_if_shock_barely_exists: bool) -> DeflagrationOutput:
    # Solve the boundary conditions at the wall
    vp_tilde, wp = boundary.solve_junction(
        model, vm_tilde, wm,
        Phase.BROKEN, Phase.SYMMETRIC,
        v2_tilde_guess=vp_tilde_guess, w2_guess=wp_guess,
        allow_failure=allow_failure
    )
    vp = -relativity.lorentz(vp_tilde, v_wall)

    # Manual correction for hybrids
    # if sol_type == SolutionType.HYBRID:
    #     # If we are already below the shock velocity, then add a manual correction
    #     vm_shock_tilde, w_shock = shock.solve_shock(
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
    #         logger.warning("vp < v_shock at the wall. Applying manual correction. Got: vp=%s, v_shock=%s", vp, vm_shock)
    #         vp = vm_shock + 1e-3
    #         wp = w_shock + 1e-3

    # logger.debug(f"vp_tilde={vp_tilde}, vp={vp}, wp={wp}")

    # Integrate from the wall to the shock
    # pylint: disable=unused-variable
    v, w, xi, t = integrate.fluid_integrate_param(
        v0=vp, w0=wp, xi0=v_wall,
        phase=Phase.SYMMETRIC,
        t_end=-const.T_END_DEFAULT,
        n_xi=const.N_XI_DEFAULT,
        df_dtau_ptr=model.df_dtau_ptr(),
        # method="RK45"
    )
    i_shock = shock.find_shock_index(
        model, v, xi, v_wall, wn,
        cs_n=cs_n, sol_type=sol_type,
        allow_failure=allow_failure, warn_if_barely_exists=warn_if_shock_barely_exists
    )
    if i_shock == 0:
        logger.error("The shock was not found by the deflagration solver")
        return DEFLAGRATION_NAN
    v = v[:i_shock]
    w = w[:i_shock]
    xi = xi[:i_shock]

    xi_sh = xi[-1]
    vm_sh = v[-1]
    wm_sh = w[-1]
    vm_tilde_sh = relativity.lorentz(xi_sh, vm_sh)
    wn_estimate = boundary.w2_junction(vm_tilde_sh, wm_sh, xi_sh)

    vm = relativity.lorentz(vm_tilde, v_wall)
    return v, w, xi, vp, vm, vp_tilde, vm_tilde, xi_sh, vm_sh, vm_tilde_sh, wp, wn_estimate, wm_sh


def fluid_shell_deflagration_reverse(model: "Model", v_wall: float, wn: float, xi_sh: float, allow_failure: bool = False):
    logger.warning("UNTESTED, will probably produce invalid results")

    if np.isnan(v_wall) or v_wall < 0 or v_wall > 1 or np.isnan(xi_sh) or xi_sh < 0 or xi_sh > 1:
        logger.error(f"Invalid parameters: v_wall={v_wall}, xi_sh={xi_sh}")
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr, np.nan, np.nan

    # Solve boundary conditions at the shock
    vm_sh = shock.v_shock_bag(xi_sh)
    wm_sh = shock.wm_shock_bag(xi_sh, wn)

    # Integrate from the shock to the wall
    logger.info(
        f"Integrating deflagration with v_wall={v_wall}, wn={wn} from vm_sh={vm_sh}, wm_sh={wm_sh}, xi_sh={xi_sh}")
    v, w, xi, t = integrate.fluid_integrate_param(
        v0=vm_sh, w0=wm_sh, xi0=xi_sh,
        phase=Phase.SYMMETRIC,
        t_end=const.T_END_DEFAULT,
        n_xi=const.N_XI_DEFAULT,
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

    vm_tilde, wm = boundary.solve_junction(
        model, vp_tilde, wp,
        Phase.SYMMETRIC, Phase.BROKEN,
        v2_tilde_guess=v_wall, w2_guess=wp,
        allow_failure=allow_failure
    )
    vm = relativity.lorentz(vm_tilde, v_wall)

    return v, w, xi, wp, wm, vm


def fluid_shell_detonation(
        model: "Model", v_wall: float, alpha_n: float, wn: float, v_cj: float,
        vm_tilde_guess: float, wm_guess: float) -> SolverOutput:
    if transition.cannot_be_detonation(v_wall, v_cj):
        raise ValueError(f"Too slow wall speed for a detonation: v_wall={v_wall}, v_cj={v_cj}")
    # Use bag model as the starting point. This may fail for points near the v_cj curve.
    vp_tilde_bag, vm_tilde_bag, vp_bag, vm_bag = boundary.fluid_speeds_at_wall(
        v_wall, alpha_p=alpha_n, sol_type=SolutionType.DETON)
    wm_bag = boundary.w2_junction(v1=vp_tilde_bag, w1=wn, v2=vm_tilde_bag)

    # The bag model works for more points than the pre-generated guesses, so let's use the bag model if we can.
    if not np.isnan(vm_tilde_bag):
        vm_tilde_guess = vm_tilde_bag
    if not np.isnan(wm_bag):
        wm_guess = wm_bag

    # Solve junction conditions
    vm_tilde, wm = boundary.solve_junction(
        model,
        v1_tilde=v_wall, w1=wn,
        phase1=Phase.SYMMETRIC, phase2=Phase.BROKEN,
        v2_tilde_guess=vm_tilde_guess, w2_guess=wm_guess)

    # Convert to the plasma frame
    vm = relativity.lorentz(v_wall, vm_tilde)

    v, w, xi, t = integrate.fluid_integrate_param(
        v0=vm, w0=wm, xi0=v_wall,
        phase=Phase.BROKEN,
        t_end=-const.T_END_DEFAULT,
        n_xi=const.N_XI_DEFAULT,
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

    solution_found = True
    # Revert the order of points in the arrays for concatenation
    return np.flip(v), np.flip(w), np.flip(xi), vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wn, wm, wm, solution_found


def fluid_shell_hybrid(
        model: "Model", v_wall: float, wn: float, wm: float, cs_n: float,
        vp_tilde_guess: float, wp_guess: float,
        allow_failure: bool = False, warn_if_shock_barely_exists: bool = True) -> DeflagrationOutput:
    # Exit velocity is at the sound speed
    vm_tilde = np.sqrt(model.cs2(wm, Phase.BROKEN))

    if np.isnan(vp_tilde_guess):
        vp_tilde_guess = 0.75*vm_tilde
    if np.isnan(wp_guess):
        wp_guess = 2*wm
    return fluid_shell_deflagration_common(
        model,
        v_wall=v_wall,
        vm_tilde=vm_tilde,
        wn=wn, wm=wm,
        cs_n=cs_n,
        vp_tilde_guess=vp_tilde_guess, wp_guess=wp_guess,
        sol_type=SolutionType.HYBRID,
        allow_failure=allow_failure,
        warn_if_shock_barely_exists=warn_if_shock_barely_exists
    )


# Solvables

def fluid_shell_solvable_deflagration_reverse(params: np.ndarray, model: "Model", v_wall: float, wn: float) -> float:
    xi_sh = params[0]
    # pylint: disable=unused-variable
    v, w, xi, vm, wm = fluid_shell_deflagration_reverse(model, v_wall, wn, xi_sh, allow_failure=True)
    return vm


def fluid_shell_solvable_deflagration(
        # params: np.ndarray,
        w_center: float, model: "Model", v_wall: float, wn: float, cs_n: float,
        vp_guess: float, wp_guess: float) -> float:
    if isinstance(w_center, np.ndarray):
        w_center = w_center[0]
    if np.isnan(w_center) or w_center < 0:
        return np.nan
    # pylint: disable=unused-variable
    v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wn_estimate, wm_sh = fluid_shell_deflagration(
        model, v_wall=v_wall, wn=wn, w_center=w_center, cs_n=cs_n,
        vp_guess=vp_guess, wp_guess=wp_guess,
        allow_failure=True, warn_if_shock_barely_exists=False)
    return wn_estimate - wn


def fluid_shell_solvable_hybrid(
        # params: np.ndarray,
        wm: float, model: "Model", v_wall: float, wn: float, cs_n: float,
        vp_tilde_guess: float, wp_guess: float) -> float:
    if isinstance(wm, np.ndarray):
        wm = wm[0]
    if np.isnan(wm) or wm < 0:
        return np.nan
    # pylint: disable=unused-variable
    v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wn_estimate, wm_sh = fluid_shell_hybrid(
        model, v_wall=v_wall, wn=wn, wm=wm,
        cs_n=cs_n,
        vp_tilde_guess=vp_tilde_guess, wp_guess=wp_guess,
        allow_failure=True, warn_if_shock_barely_exists=False)
    return wn_estimate - wn


# Solvers

def fluid_shell_solver_deflagration(
        model: "Model",
        start_time: float,
        v_wall: float, alpha_n: float, wn: float, cs_n: float, high_alpha_n: bool,
        wm_guess: float, vp_guess: float, wp_guess: float, wn_rtol: float,
        allow_failure: bool, log_high_alpha_n_failures: bool = True) -> SolverOutput:
    if vp_guess > v_wall:
        vp_guess_new = 0.95 * v_wall
        if log_high_alpha_n_failures or not high_alpha_n:
            logger.error("Invalid vp_guess=%s > v_wall=%s, replacing with vp_guess=%s", vp_guess, v_wall, vp_guess_new)
        vp_guess = vp_guess_new

    sol = root_scalar(
        fluid_shell_solvable_deflagration,
        x0=0.99*wm_guess,
        x1=1.01*wm_guess,
        args=(model, v_wall, wn, cs_n, vp_guess, wp_guess),
    )
    wm = sol.root
    solution_found = sol.converged
    reason = sol.flag

    if not solution_found:
        # if not high_alpha_n:
        #     logger.error("FALLBACK")
        sol = fsolve_vary(
            fluid_shell_solvable_deflagration,
            np.array([wm_guess]),
            args=(model, v_wall, wn, cs_n, vp_guess, wp_guess),
            log_status=log_high_alpha_n_failures or not high_alpha_n
        )
        wm = sol[0][0]
        solution_found = sol[2] == 1
        reason = sol[3]

    v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wn_estimate, wm_sh = fluid_shell_deflagration(
        model, v_wall, wn, wm,
        cs_n=cs_n,
        vp_guess=vp_guess, wp_guess=wp_guess,
        allow_failure=allow_failure, warn_if_shock_barely_exists=False
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


def fluid_shell_solver_deflagration_reverse(
        model: "Model",
        start_time: float,
        v_wall: float, alpha_n: float, wn: float) -> SolverOutput:
    # This is arbitrary and should be replaced by a value from the bag model
    xi_sh_guess = 1.1 * np.sqrt(model.cs2_max(wn, Phase.BROKEN))
    sol = fsolve(
        fluid_shell_solvable_deflagration_reverse,
        xi_sh_guess,
        args=(model, v_wall, wn),
        full_output=True
    )
    xi_sh = sol[0][0]
    solution_found = True
    if sol[2] != 1:
        solution_found = False
        logger.error(
            f"Deflagration solution was not found for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. "
            f"Using xi_sh={xi_sh}. Reason: {sol[3]} Elapsed: {time.perf_counter() - start_time} s."
        )
    v, w, xi, wp, wm, vm = fluid_shell_deflagration_reverse(model, v_wall, wn, xi_sh)

    return v, w, xi, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, wp, wm, np.nan, solution_found


def fluid_shell_solver_hybrid(
        model: "Model",
        start_time: float,
        v_wall: float, alpha_n: float, wn: float, cs_n: float, high_alpha_n: bool,
        vp_tilde_guess: float, wp_guess: float, wm_guess: float, wn_rtol: float,
        allow_failure: bool, log_high_alpha_n_failures: bool) -> SolverOutput:

    sol = root_scalar(
        fluid_shell_solvable_hybrid,
        x0=0.99*wm_guess,
        x1=1.01*wm_guess,
        args=(model, v_wall, wn, cs_n, vp_tilde_guess, wp_guess)
    )
    wm = sol.root
    solution_found = sol.converged
    reason = sol.flag

    if not solution_found:
        # if not high_alpha_n:
        #     logger.error("FALLBACK")
        sol = fsolve_vary(
            fluid_shell_solvable_hybrid,
            np.array([wm_guess]),
            args=(model, v_wall, wn, cs_n, vp_tilde_guess, wp_guess),
            log_status=log_high_alpha_n_failures or not high_alpha_n
        )
        wm = sol[0][0]
        solution_found = sol[2] == 1
        reason = sol[3]

    v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wn_estimate, wm_sh = fluid_shell_hybrid(
        model, v_wall, wn, wm,
        cs_n=cs_n,
        vp_tilde_guess=vp_tilde_guess,
        wp_guess=wp_guess,
        allow_failure=allow_failure,
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
        t_end=-const.T_END_DEFAULT,
        df_dtau_ptr=model.df_dtau_ptr())
    v = np.concatenate((np.flip(v_tail), v))
    w = np.concatenate((np.flip(w_tail), w))
    xi = np.concatenate((np.flip(xi_tail), xi))

    return v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found


# Main function

def fluid_shell_generic(
            model: "Model",
            v_wall: float,
            alpha_n: float,
            sol_type: tp.Optional[SolutionType] = None,
            wn: float = None,
            vp_guess: float = None,
            wn_guess: float = None,
            wp_guess: float = None,
            wm_guess: float = None,
            wn_rtol: float = 1e-4,
            alpha_n_max_bag: float = None,
            high_alpha_n: bool = None,
            n_xi: int = const.N_XI_DEFAULT,
            reverse: bool = False,
            allow_failure: bool = False,
            use_bag_solver: bool = False,
            log_success: bool = True,
            log_high_alpha_n_failures: bool = False
        ) -> tp.Tuple[
            np.ndarray, np.ndarray, np.ndarray, SolutionType,
            float, float, float, float, float, float, float, float, float, float, float, bool, float]:
    """Generic fluid shell solver

    In most cases you should not have to call this directly. Create a Bubble instead.
    """
    start_time = time.perf_counter()
    if alpha_n_max_bag is None:
        alpha_n_max_bag = alpha.alpha_n_max_deflagration_bag(v_wall)
    if high_alpha_n is None:
        high_alpha_n = alpha_n > alpha_n_max_bag

    if wn is None:
        wn = model.w_n(alpha_n, wn_guess=wn_guess)
    # The shock curve hits v=0 here
    cs_n = np.sqrt(model.cs2(wn, Phase.SYMMETRIC))

    if use_bag_solver and model.DEFAULT_NAME == "bag":
        if high_alpha_n:
            logger.info("Got model=%s, v_wall=%s, alpha_n=%s, for which there is no solution.", model.label_unicode, v_wall, alpha_n)
            return const.nan_arr, const.nan_arr, const.nan_arr, SolutionType.ERROR, \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, True, time.perf_counter() - start_time

        logger.info("Using bag solver for model=%s, v_wall=%s, alpha_n=%s", model.label_unicode, v_wall, alpha_n)
        sol_type2 = transition.identify_solution_type_bag(v_wall, alpha_n)
        if sol_type is not None and sol_type != sol_type2:
            raise ValueError(f"Bag model gave a different solution type ({sol_type2}) than what was given ({sol_type}).")

        v, w, xi = fluid_bag.fluid_shell(v_wall, alpha_n)
        # The results of the old solver are scaled to wn=1
        w *= wn
        if np.any(np.isnan(v)):
            return v, w, xi, sol_type2, \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, True, time.perf_counter() - start_time

        vp, vm, vp_tilde, vm_tilde, wp, wm, wn, wm_sh = props.v_and_w_from_solution(v, w, xi, v_wall, sol_type2)

        # The wm_guess is not needed for the bag model
        v_cj: float = chapman_jouguet.v_chapman_jouguet(model, alpha_n, wn=wn, wm_guess=wm)
        return v, w, xi, sol_type2, \
            vp, vm, vp_tilde, vm_tilde, np.nan, np.nan, np.nan, wp, wm, wm_sh, v_cj, False, time.perf_counter() - start_time

    sol_type = transition.validate_solution_type(
        model,
        v_wall=v_wall, alpha_n=alpha_n, sol_type=sol_type,
        wn=wn, wm_guess=wm_guess
    )

    # Load and scale reference data
    using_ref = False
    vp_ref, vm_ref, vp_tilde_ref, vm_tilde_ref, wp_ref, wm_ref = fluid_reference.ref().get(v_wall, alpha_n, sol_type)

    if vp_guess is None or np.isnan(vp_guess):
        using_ref = True
        vp_guess = vp_ref
        vp_tilde_guess = vp_tilde_ref
    else:
        vp_tilde_guess = relativity.lorentz(v_wall, vp_guess)

    # The reference data has wn=1 and therefore has to be scaled with wn.
    if wp_guess is None or np.isnan(wp_guess):
        using_ref = True
        wp_guess = wp_ref * wn
    if wm_guess is None or np.isnan(wp_guess):
        using_ref = True
        if np.isnan(wm_ref):
            logger.warning(
                "No reference data for v_wall=%s, alpha_n=%s. Using an arbitrary starting guess.",
                v_wall, alpha_n
            )
            wm_guess = 0.3 * wn
        else:
            wm_guess = wm_ref * wn
    # if wn_guess is None:
    #     wn_guess = min(wp_guess, wm_guess)

    if using_ref and np.any(np.isnan((vp_ref, vm_ref, vp_tilde_ref, vm_tilde_ref, wp_ref, wm_ref))):
        logger.warning(
            "Using arbitrary starting guesses at v_wall=%s, alpha_n=%s,"
            "as all starting guesses were not provided, and the reference has nan values."
        )

    if vp_guess < 0 or vp_guess > 1 or vp_tilde_guess < 0 or vp_tilde_guess > 1 or wm_guess < 0 or wp_guess < wn:
        raise ValueError(
            f"Got invalid guesses: vp_tilde={vp_tilde_guess}, wp={wp_guess}, wm={wm_guess}"
            f"for v_wall={v_wall}, alpha_n={alpha_n}, wn={wn_guess}"
        )

    v_cj = chapman_jouguet.v_chapman_jouguet(model, alpha_n, wn=wn, wm_guess=wm_guess)
    dxi = 1. / n_xi

    if log_success:
        logger.info(
            "Solving fluid shell for model=%s, v_wall=%s, alpha_n=%s " +
            (f"(alpha_n_max_bag={alpha_n_max_bag}) " if high_alpha_n and sol_type != SolutionType.DETON else "") +
            "with sol_type=%s, v_cj=%s, wn=%s "
            "and starting guesses vp=%s vp_tilde=%s, wp=%s, wm=%s, wn=%s",
            model.label_unicode, v_wall, alpha_n,
            sol_type, v_cj, wn,
            vp_guess, vp_tilde_guess, wp_guess, wm_guess, wn_guess
        )

    # Detonations are the simplest case
    if sol_type == SolutionType.DETON:
        v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found = \
            fluid_shell_detonation(model, v_wall, alpha_n, wn, v_cj, vm_tilde_guess=vm_tilde_ref, wm_guess=wm_ref)
    elif sol_type == SolutionType.SUB_DEF:
        if transition.cannot_be_sub_def(model, v_wall, wn):
            raise ValueError(
                f"Invalid parameters for a subsonic deflagration: model={model.name}, v_wall={v_wall}, wn={wn}. "
                "Decrease v_wall or increase csb2."
            )

        # In more advanced models,
        # the direction of the integration will probably have to be determined by trial and error.
        if reverse:
            logger.warning("Using reverse deflagration solver, which has not been properly tested.")
            v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found = \
                fluid_shell_solver_deflagration_reverse(model, start_time, v_wall, alpha_n, wn)
        else:
            v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found = \
                fluid_shell_solver_deflagration(
                    model, start_time,
                    v_wall, alpha_n, wn,
                    cs_n=cs_n,
                    high_alpha_n=high_alpha_n,
                    wm_guess=wm_guess, vp_guess=vp_guess, wp_guess=wp_guess, wn_rtol=wn_rtol,
                    allow_failure=allow_failure, log_high_alpha_n_failures=log_high_alpha_n_failures
                )
    elif sol_type == SolutionType.HYBRID:
        v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found = \
            fluid_shell_solver_hybrid(
                model, start_time,
                v_wall, alpha_n, wn,
                cs_n=cs_n,
                high_alpha_n=high_alpha_n,
                vp_tilde_guess=vp_tilde_guess,
                wp_guess=wp_guess,
                wm_guess=wm_guess,
                wn_rtol=wn_rtol,
                allow_failure=allow_failure,
                log_high_alpha_n_failures=log_high_alpha_n_failures
            )
    else:
        raise ValueError(f"Invalid solution type: {sol_type}")

    # Behind and ahead of the bubble the fluid is still
    xif = np.linspace(xi[-1] + dxi, 1, 2)
    xib = np.linspace(0, xi[0] - dxi, 2)
    vf = np.zeros_like(xif)
    vb = np.zeros_like(xib)
    wf = np.ones_like(xif) * wn
    w_center = min(wm, w[0])
    wb = np.ones_like(vb) * w_center

    v = np.concatenate((vb, v, vf))
    w = np.concatenate((wb, w, wf))
    xi = np.concatenate((xib, xi, xif))

    elapsed = time.perf_counter() - start_time
    if solution_found and log_success:
        logger.info(
            "Solved fluid shell for model=%s, v_wall=%s, alpha_n=%s, sol_type=%s. Elapsed: %s s",
            model.label_unicode, v_wall, alpha_n, sol_type, elapsed
        )
    return v, w, xi, sol_type, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, v_cj, not solution_found, elapsed
