r"""Solver for the fluid velocity profile of a bubble"""

import logging
import typing as tp

import numpy as np
from scipy.optimize import fsolve

from . import boundary
from .boundary import Phase, SolutionType
from . import chapman_jouguet
from . import const
from . import integrate
from . import fluid_reference
from . import relativity
from . import shock
from . import transition
from . import trim
if tp.TYPE_CHECKING:
    from pttools.models import Model

logger = logging.getLogger(__name__)

SolverOutput = tp.Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]


def fluid_shell_deflagration(
        model: "Model",
        v_wall: float, wn: float, w_center: float,
        vp_guess: float = None, wp_guess: float = None,
        allow_failure: bool = False,
        warn_if_shock_barely_exists: bool = True) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    # using_bag = False
    if vp_guess is None or wp_guess is None:
        # using_bag = True
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
        vp_tilde_guess = relativity.lorentz(vp_guess, v_wall)

    invalid_param = None
    if np.isnan(vp_tilde_guess) < 0 or vp_tilde_guess > 1:
        invalid_param = "vp_tilde_guess"
    elif vp_guess < 0 or vp_guess > v_wall:
        invalid_param = "vp_guess"
    elif np.isnan(wp_guess):
        invalid_param = "wp_guess"
    elif wp_guess < wn:
        logger.warning(f"Using invalid wp_guess={wp_guess}")
    if invalid_param is not None:
        logger.error(
            f"Invalid parameter: {invalid_param}. Got: "
            f"model={model.label_unicode}, v_wall={v_wall}, wn={wn}, w_center={w_center}, "
            f"vp_guess={vp_guess}, vp_tilde_guess={vp_tilde_guess}, wp_guess={wp_guess}"
        )
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr, np.nan, np.nan

    return fluid_shell_deflagration_common(
        model,
        v_wall, wn,
        v_wall, w_center,
        vp_tilde_guess, wp_guess,
        SolutionType.SUB_DEF,
        allow_failure=allow_failure,
        warn_if_shock_barely_exists=warn_if_shock_barely_exists)


def fluid_shell_deflagration_common(
        model: "Model",
        v_wall: float, wn: float, vm_tilde: float, wm: float, vp_tilde_guess: float, wp_guess: float,
        sol_type: SolutionType,
        allow_failure: bool,
        warn_if_shock_barely_exists: bool) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    # Solve the boundary conditions at the wall
    vp_tilde, wp = boundary.solve_junction(
        model, vm_tilde, wm,
        Phase.BROKEN, Phase.SYMMETRIC,
        v2_guess=vp_tilde_guess, w2_guess=wp_guess,
        allow_failure=allow_failure
    )
    vp = -relativity.lorentz(vp_tilde, v_wall)
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
        model, v, xi, v_wall, wn, sol_type,
        allow_failure=allow_failure, warn_if_barely_exists=warn_if_shock_barely_exists
    )
    if i_shock == 0:
        logger.error("The shock was not found by the deflagration solver")
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr, wp, np.nan
    v = v[:i_shock]
    w = w[:i_shock]
    xi = xi[:i_shock]

    xi_sh = xi[-1]
    vm_tilde_sh = relativity.lorentz(xi_sh, v[-1])
    wn_estimate = boundary.w2_junction(vm_tilde_sh, w[-1], xi_sh)
    return v, w, xi, wp, wn_estimate


def fluid_shell_deflagration_reverse(model: "Model", v_wall: float, wn: float, xi_sh: float, allow_failure: bool = False):
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
        v2_guess=v_wall, w2_guess=wp,
        allow_failure=allow_failure
    )
    vm = relativity.lorentz(vm_tilde, v_wall)

    return v, w, xi, wp, wm, vm


def fluid_shell_detonation(model: "Model", v_wall: float, alpha_n: float, wn: float, v_cj: float) -> SolverOutput:
    if transition.cannot_be_detonation(v_wall, v_cj):
        raise ValueError(f"Too slow wall speed for a detonation: v_wall={v_wall}, v_cj={v_cj}")
    # Use bag model as the starting point
    vp_tilde_bag, vm_tilde_bag, vp_bag, vm_bag = boundary.fluid_speeds_at_wall(
        v_wall, alpha_p=alpha_n, sol_type=SolutionType.DETON)
    wm_bag = boundary.w2_junction(v1=vp_tilde_bag, w1=wn, v2=vm_tilde_bag)
    # Solve junction conditions
    vm_tilde, wm = boundary.solve_junction(
        model,
        v1=v_wall, w1=wn,
        phase1=Phase.SYMMETRIC, phase2=Phase.BROKEN,
        v2_guess=vm_tilde_bag, w2_guess=wm_bag)

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

    # Revert the order of points in the arrays for concatenation
    return np.flip(v), np.flip(w), np.flip(xi), wn, wm, True


def fluid_shell_hybrid(
        model: "Model", v_wall: float, wn: float, wm: float,
        vp_tilde_guess: float, wp_guess: float,
        allow_failure: bool = False, warn_if_shock_barely_exists: bool = True):
    # Exit velocity is at the sound speed
    vm_tilde = np.sqrt(model.cs2(wm, Phase.BROKEN))
    return fluid_shell_deflagration_common(
        model, v_wall, wn,
        vm_tilde, wm,
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


def fluid_shell_solvable_deflagration(params: np.ndarray, model: "Model", v_wall: float, wn: float) -> float:
    w_center = params[0]
    # pylint: disable=unused-variable
    v, w, xi, wp, wn_estimate = fluid_shell_deflagration(
        model, v_wall, wn, w_center,
        allow_failure=True, warn_if_shock_barely_exists=False)
    return wn_estimate - wn


def fluid_shell_solvable_hybrid(
        params: np.ndarray, model: "Model", v_wall: float, wn: float,
        vp_tilde_guess: float, wp_guess: float) -> float:
    wm = params[0]
    # pylint: disable=unused-variable
    v, w, xi, wp, wn_estimate = fluid_shell_hybrid(
        model, v_wall, wn, wm,
        vp_tilde_guess=vp_tilde_guess, wp_guess=wp_guess,
        allow_failure=True, warn_if_shock_barely_exists=False)
    return wn_estimate - wn


# Solvers

def fluid_shell_solver_deflagration(
        model: "Model", v_wall: float, alpha_n: float, wn: float, wm_guess: float, allow_failure: bool) -> SolverOutput:
    sol = fsolve(
        fluid_shell_solvable_deflagration,
        np.array([wm_guess]),
        args=(model, v_wall, wn),
        full_output=True
    )
    wm = sol[0][0]
    solution_found = True
    if sol[2] != 1:
        solution_found = False
        logger.error(
            f"Deflagration solution was not found for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. "
            f"Using wm={wm}. Reason: {sol[3]}"
        )
    v, w, xi, wp, wn_estimate = fluid_shell_deflagration(
        model, v_wall, wn, wm, allow_failure=allow_failure)
    if not np.isclose(wn_estimate, wn):
        solution_found = False
        logger.error(
            f"Deflagration solution was not found for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. "
            f"Got wn_estimate={wn_estimate}, which differs from wn={wn}."
        )
    # print(np.array([v, w, xi]).T)
    # print("wn, xi_sh", wn, xi_sh)

    return v, w, xi, wp, wm, solution_found


def fluid_shell_solver_deflagration_reverse(model: "Model", v_wall: float, alpha_n: float, wn: float) -> SolverOutput:
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
            f"Using xi_sh={xi_sh}. Reason: {sol[3]}"
        )
    v, w, xi, wp, wm, vm = fluid_shell_deflagration_reverse(model, v_wall, wn, xi_sh)

    return v, w, xi, wp, wm, solution_found


def fluid_shell_solver_hybrid(
        model: "Model", v_wall: float, alpha_n: float, wn: float,
        vp_tilde_guess: float, wp_guess: float, wm_guess: float, allow_failure: bool) -> SolverOutput:
    sol = fsolve(
        fluid_shell_solvable_hybrid,
        np.array([wm_guess]),
        args=(model, v_wall, wn, vp_tilde_guess, wp_guess),
        full_output=True
    )
    wm = sol[0][0]
    solution_found = True
    if sol[2] != 1:
        solution_found = False
        logger.error(
            f"Hybrid solution was not found for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. "
            f"Using wm={wm}. Reason: {sol[3]}"
        )
    v, w, xi, wp, wn_estimate = fluid_shell_hybrid(
        model, v_wall, wn, wm,
        vp_tilde_guess=vp_tilde_guess,
        wp_guess=wp_guess,
        allow_failure=allow_failure
    )
    wp = w[0]
    if not np.isclose(wn_estimate, wn):
        solution_found = False
        logger.error(
            f"Hybrid solution was not found for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. "
            f"Got wn_estimate={wn_estimate}, which differs from wn={wn}."
        )
    vm = relativity.lorentz(v_wall, np.sqrt(model.cs2(wm, Phase.BROKEN)))
    v_tail, w_tail, xi_tail, t_tail = integrate.fluid_integrate_param(
        vm, wm, v_wall,
        phase=Phase.BROKEN,
        t_end=-const.T_END_DEFAULT,
        df_dtau_ptr=model.df_dtau_ptr())
    v = np.concatenate((np.flip(v_tail), v))
    w = np.concatenate((np.flip(w_tail), w))
    xi = np.concatenate((np.flip(xi_tail), xi))

    return v, w, xi, wp, wm, solution_found


# Main function

def fluid_shell_generic(
            model: "Model",
            v_wall: float,
            alpha_n: float,
            sol_type: tp.Optional[SolutionType] = None,
            vp_tilde_guess: float = None,
            wn_guess: float = 1,
            wp_guess: float = None,
            wm_guess: float = None,
            n_xi: int = const.N_XI_DEFAULT,
            reverse: bool = False,
            allow_failure: bool = False
        ):
    logger.info(
        "Solving fluid shell for model=%s, v_wall=%s, alpha_n=%s, sol_type=%s",
        model.label_unicode, v_wall, alpha_n, sol_type,
    )

    wn = model.w_n(alpha_n, wn_guess=wn_guess)

    # Load and scale reference data
    vp_ref, vm_ref, vp_tilde_ref, vm_tilde_ref, wp_ref, wm_ref = fluid_reference.ref().get(v_wall, alpha_n)
    if vp_tilde_guess is None:
        vp_tilde_guess = vp_tilde_ref
    # The reference data has wn=1 and therefore has to be scaled with wn.
    if wp_guess is None:
        wp_guess = wp_ref * wn
    if wm_guess is None:
        wm_guess = wm_ref * wn
    # if wn_guess is None:
    #     wn_guess = min(wp_guess, wm_guess)

    if vp_tilde_guess < 0 or wm_guess < 0 or wn_guess < 0 or wp_guess < wn_guess or wp_guess < wm_guess:
        raise ValueError(f"Got invalid guesses: vp_tilde={vp_tilde_guess}, wp={wp_guess}, wm={wm_guess}, wn={wn_guess}")

    logger.debug(
        "with starting guesses vp_tilde=%s, wp=%s, wm=%s, wn=%s",
        vp_tilde_guess, wp_guess, wm_guess, wn_guess
    )

    sol_type = transition.validate_solution_type(
        model,
        v_wall=v_wall, alpha_n=alpha_n, sol_type=sol_type,
        wn_guess=wn, wm_guess=wm_guess
    )

    v_cj = chapman_jouguet.v_chapman_jouguet(model, alpha_n, wn, wm_guess)
    dxi = 1. / n_xi
    logger.info(
        "Solved model parameters: v_cj=%s, wn=%s for bubble with model=%s, v_wall=%s, sol_type=%s, alpha_n=%s",
        v_cj, wn, model.label_unicode, v_wall, sol_type, alpha_n
    )

    # Detonations are the simplest case
    if sol_type == SolutionType.DETON:
        v, w, xi, wp, wm, solution_found = fluid_shell_detonation(model, v_wall, alpha_n, wn, v_cj)
    elif sol_type == SolutionType.SUB_DEF:
        if transition.cannot_be_sub_def(model, v_wall, wn):
            raise ValueError(
                f"Invalid parameters for a subsonic deflagration: model={model.name}, v_wall={v_wall}, wn={wn}. "
                "Decrease v_wall or increase csb2."
            )

        # In more advanced models,
        # the direction of the integration will probably have to be determined by trial and error.
        if reverse:
            v, w, xi, wp, wm, solution_found = fluid_shell_solver_deflagration_reverse(model, v_wall, alpha_n, wn)
        else:
            v, w, xi, wp, wm, solution_found = fluid_shell_solver_deflagration(
                model, v_wall, alpha_n, wn, wm_guess, allow_failure)
    elif sol_type == SolutionType.HYBRID:
        v, w, xi, wp, wm, solution_found = fluid_shell_solver_hybrid(
            model, v_wall, alpha_n, wn,
            vp_tilde_guess=vp_tilde_guess,
            wp_guess=wp_guess,
            wm_guess=wm_guess,
            allow_failure=allow_failure
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

    if solution_found:
        logger.info(
            "Solved fluid shell for model=%s, v_wall=%s, alpha_n=%s, sol_type=%s",
            model.label_unicode, v_wall, alpha_n, sol_type
        )
    else:
        logger.error(
            "Failed to find a solution. Returning approximate results for model=%s, v_wall=%s, alpha_n=%s, sol_type=%s",
            model.label_unicode, v_wall, alpha_n, sol_type
        )
    return v, w, xi, sol_type, wp, wm, not solution_found
