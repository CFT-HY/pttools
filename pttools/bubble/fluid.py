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
from . import relativity
from . import shock
from . import transition
from . import trim
if tp.TYPE_CHECKING:
    from pttools.models import Model

logger = logging.getLogger(__name__)


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

    return v, w, xi, vm, wm


def fluid_shell_deflagration(
        model: "Model",
        v_wall: float, wn: float, w_center: float,
        vp_guess: float = None, wp_guess: float = None,
        allow_failure: bool = False,
        warn_if_shock_barely_exists: bool = True) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
            f"v_wall={v_wall}, w_center={w_center}, "
            f"vp_tilde_guess={vp_tilde_guess}, vp_guess={vp_guess}, wp_guess={wp_guess}, wn={wn}"
        )
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr, np.nan

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
        warn_if_shock_barely_exists: bool) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
    v = v[:i_shock]
    w = w[:i_shock]
    xi = xi[:i_shock]

    xi_sh = xi[-1]
    vm_tilde_sh = relativity.lorentz(xi_sh, v[-1])
    wn_estimate = boundary.w2_junction(vm_tilde_sh, w[-1], xi_sh)
    return v, w, xi, wn_estimate


def fluid_shell_deflagration_reverse_solvable(params: np.ndarray, model: "Model", v_wall: float, wn: float) -> float:
    xi_sh = params[0]
    # pylint: disable=unused-variable
    v, w, xi, vm, wm = fluid_shell_deflagration_reverse(model, v_wall, wn, xi_sh, allow_failure=True)
    return vm


def fluid_shell_deflagration_solvable(params: np.ndarray, model: "Model", v_wall: float, wn: float) -> float:
    w_center = params[0]
    # pylint: disable=unused-variable
    v, w, xi, wn_estimate = fluid_shell_deflagration(
        model, v_wall, wn, w_center,
        allow_failure=True, warn_if_shock_barely_exists=False)
    v, w, xi, wn_estimate = fluid_shell_deflagration(model, v_wall, wn, w_center, allow_failure=True)
    return wn_estimate - wn


def fluid_shell_hybrid(
        model: "Model", v_wall: float, wn: float, wm: float,
        allow_failure: bool = False, warn_if_shock_barely_exists: bool = True):
    # Exit velocity is at the sound speed
    vm_tilde = np.sqrt(model.cs2(wm, Phase.BROKEN))
    return fluid_shell_deflagration_common(
        model, v_wall, wn,
        vm_tilde, wm,
        vp_tilde_guess=0.75*vm_tilde, wp_guess=2*wm,
        sol_type=SolutionType.HYBRID,
        allow_failure=allow_failure,
        warn_if_shock_barely_exists=warn_if_shock_barely_exists
    )


def fluid_shell_hybrid_solvable(params: np.ndarray, model: "Model", v_wall: float, wn: float) -> float:
    wm = params[0]
    # pylint: disable=unused-variable
    v, w, xi, wn_estimate = fluid_shell_hybrid(
        model, v_wall, wn, wm,
        allow_failure=True, warn_if_shock_barely_exists=False)
    return wn_estimate - wn


def fluid_shell_generic(
            model: "Model",
            v_wall: float,
            alpha_n: float,
            sol_type: tp.Optional[SolutionType] = None,
            wn_guess: float = 1,
            wm_guess: float = 2,
            n_xi: int = const.N_XI_DEFAULT,
            reverse: bool = False,
            allow_failure: bool = False
        ):
    logger.info(
        "Solving fluid shell for model=%s, v_wall=%s, alpha_n=%s, sol_type=%s",
        model.label_unicode, v_wall, alpha_n, sol_type
    )
    sol_type = transition.validate_solution_type(
        model,
        v_wall=v_wall, alpha_n=alpha_n, sol_type=sol_type,
        wn_guess=wn_guess, wm_guess=wm_guess)

    failed = False
    wn = model.w_n(alpha_n, wn_guess=wn_guess)
    v_cj = chapman_jouguet.v_chapman_jouguet(model, alpha_n, wn, wm_guess)
    dxi = 1. / n_xi
    logger.info(
        "Solved model parameters: v_cj=%s, wn=%s for bubble with model=%s, v_wall=%s, sol_type=%s, alpha_n=%s",
        v_cj, wn, model.label_unicode, v_wall, sol_type, alpha_n
    )

    # Detonations are the simplest case
    if sol_type == SolutionType.DETON:
        if transition.cannot_be_detonation(v_wall, v_cj):
            raise ValueError(f"Too slow wall speed for a detonation: v_wall={v_wall}, v_cj={v_cj}")
        wp = wn
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
        v, w, xi, t = trim.trim_fluid_wall_to_cs(v, w, xi, t, v_wall, sol_type, cs2_fun=model.cs2)
        w_center = w[-1]

        # Revert the order of points in the arrays for concatenation
        v = np.flip(v)
        w = np.flip(w)
        xi = np.flip(xi)

    elif sol_type == SolutionType.SUB_DEF:
        if transition.cannot_be_sub_def(model, v_wall, wn):
            raise ValueError(
                f"Invalid parameters for a subsonic deflagration: model={model.name}, v_wall={v_wall}, wn={wn}. "
                "Decrease v_wall or increase csb2."
            )

        # Todo: In more advanced models,
        #  the direction of the integration will probably have to be determined by trial and error.

        if reverse:
            xi_sh_guess = 1.1*np.sqrt(model.cs2_max(wn, Phase.BROKEN))
            sol = fsolve(
                fluid_shell_deflagration_reverse_solvable,
                xi_sh_guess,
                args=(model, v_wall, wn),
                full_output=True
            )
            xi_sh = sol[0][0]
            if sol[2] != 1:
                failed = True
                logger.error(
                    f"Deflagration solution was not found for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. "
                    f"Using xi_sh={xi_sh}. Reason: {sol[3]}"
                )
            v, w, xi, _, w_center = fluid_shell_deflagration_reverse(model, v_wall, wn, xi_sh)
        else:
            wm_guess = 0.3*wn
            sol = fsolve(
                fluid_shell_deflagration_solvable,
                wm_guess,
                args=(model, v_wall, wn),
                full_output=True
            )
            w_center = sol[0][0]
            if sol[2] != 1:
                failed = True
                logger.error(
                    f"Deflagration solution was not found for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. "
                    f"Using w_center={w_center}. Reason: {sol[3]}"
                )
            v, w, xi, wn_estimate = fluid_shell_deflagration(
                model, v_wall, wn, w_center, allow_failure=allow_failure)
            if not np.isclose(wn_estimate, wn):
                logger.error(
                    f"Deflagration solution was not found for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. "
                    f"Got wn_estimate={wn_estimate}, which differs from wn={wn}."
                )
            # print(np.array([v, w, xi]).T)
            # print("wn, xi_sh", wn, xi_sh)
        wm = w_center
        wp = w[0]
    elif sol_type == SolutionType.HYBRID:
        wm_guess = 2*wn
        sol = fsolve(
            fluid_shell_hybrid_solvable,
            wm_guess,
            args=(model, v_wall, wn),
            full_output=True
        )
        wm = sol[0][0]
        if sol[2] != 1:
            failed = True
            logger.error(
                f"Hybrid solution was not found for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. "
                f"Using wm={wm}. Reason: {sol[3]}"
            )
        v, w, xi, wn_estimate = fluid_shell_hybrid(model, v_wall, wn, wm, allow_failure=allow_failure)
        wp = w[0]
        if not np.isclose(wn_estimate, wn):
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
        w_center = w[0]
    else:
        raise ValueError(f"Invalid solution type: {sol_type}")

    # Behind and ahead of the bubble the fluid is still
    xif = np.linspace(xi[-1] + dxi, 1, 2)
    xib = np.linspace(0, xi[0] - dxi, 2)
    vf = np.zeros_like(xif)
    vb = np.zeros_like(xib)
    wf = np.ones_like(xif) * wn
    wb = np.ones_like(vb) * w_center

    v = np.concatenate((vb, v, vf))
    w = np.concatenate((wb, w, wf))
    xi = np.concatenate((xib, xi, xif))

    # params = {
    #     "vm": vm,
    #     "wm": wm,
    #     "wn": wn,
    #     "dxi": dxi,
    # }

    if failed:
        logger.error(
            "Failed to find a solution. Returning approximate results for model=%s, v_wall=%s, alpha_n=%s, sol_type=%s",
            model.label_unicode, v_wall, alpha_n, sol_type
        )
    else:
        logger.info(
            "Solved fluid shell for model=%s, v_wall=%s, alpha_n=%s, sol_type=%s",
            model.label_unicode, v_wall, alpha_n, sol_type
        )
    return v, w, xi, sol_type, wp, wm, failed
