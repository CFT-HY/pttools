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

try:
    from giese.lisa import kappaNuMuModel
except ImportError:
    kappaNuMuModel = None

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


def sound_shell_deflagration(
        model: "Model",
        v_wall: float, wn: float, w_center: float,
        cs_n: float, v_cj: float,
        vp_guess: float = None, wp_guess: float = None,
        t_end: float = const.T_END_DEFAULT, n_xi: int = const.N_XI_DEFAULT,
        thin_shell_limit: int = const.THIN_SHELL_T_POINTS_MIN,
        allow_failure: bool = False,
        allow_negative_entropy_flux_change: bool = False,
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

    return sound_shell_deflagration_common(
        model,
        v_wall=v_wall,
        vm_tilde=v_wall,
        wn=wn, wm=w_center,
        cs_n=cs_n, v_cj=v_cj,
        vp_tilde_guess=vp_tilde_guess, wp_guess=wp_guess,
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
        n_xi: int, t_end: float,
        thin_shell_limit: int,
        allow_failure: bool,
        allow_negative_entropy_flux_change: bool,
        warn_if_shock_barely_exists: bool) -> DeflagrationOutput:
    if v_wall < 0 or v_wall > 1 or vm_tilde < 0 or vm_tilde > 1 or wn < 0 or wm < 0 or cs_n < 0 or cs_n > 1 \
            or vp_tilde_guess < 0 or vp_tilde_guess > 1 or wp_guess < 0 or transition.is_surely_detonation(v_wall, v_cj):
        logger.error(
            "Invalid starting values: v_wall=%s, vm_tilde=%s, wn=%s, wm=%s, cs_n=%s, vp_tilde_guess=%s, wp_guess=%s",
            v_wall, vm_tilde, wn, wm, cs_n, vp_tilde_guess, wp_guess
        )
        return DEFLAGRATION_NAN

    # Solve the boundary conditions at the wall
    vp_tilde, wp = boundary.solve_junction(
        model, vm_tilde, wm,
        Phase.BROKEN, Phase.SYMMETRIC,
        v2_tilde_guess=vp_tilde_guess, w2_guess=wp_guess,
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
        t_end=-t_end,
        n_xi=n_xi,
        df_dtau_ptr=model.df_dtau_ptr(),
        # method="RK45"
    )
    if np.argmax(xi) == 0:
        logger.error("Deflagration solver gave a detonation-like solution.")
        return DEFLAGRATION_NAN
    i_shock = shock.find_shock_index(
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
            i_shock2 = shock.find_shock_index(
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
        logger.error("The shock was not found for v_wall=%s despite %s t_end adjustments.", v_wall, attempts)
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


def sound_shell_deflagration_reverse(
        model: "Model", v_wall: float, wn: float, xi_sh: float, t_end: float, n_xi: int,
        allow_failure: bool = False):
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

    vm_tilde, wm = boundary.solve_junction(
        model, vp_tilde, wp,
        Phase.SYMMETRIC, Phase.BROKEN,
        v2_tilde_guess=v_wall, w2_guess=wp,
        allow_failure=allow_failure
    )
    vm = relativity.lorentz(vm_tilde, v_wall)

    return v, w, xi, wp, wm, vm


def sound_shell_detonation(
        model: "Model", v_wall: float, alpha_n: float, wn: float, v_cj: float,
        vm_tilde_guess: float, wm_guess: float, t_end: float, n_xi: int) -> SolverOutput:
    if transition.cannot_be_detonation(v_wall, v_cj):
        logger.error(f"Too slow wall speed for a detonation: v_wall={v_wall}, v_cj={v_cj}")

    # Todo: use analytical ConstCSModel equations for both phases

    # Use bag model as the starting point. This may fail for points near the v_cj curve.
    vp_tilde_bag, vm_tilde_bag, vp_bag, vm_bag = boundary.fluid_speeds_at_wall(
        v_wall, alpha_p=alpha_n, sol_type=SolutionType.DETON)
    wm_bag = boundary.w2_junction(v1=vp_tilde_bag, w1=wn, v2=vm_tilde_bag)

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
    vm_tilde, wm = boundary.solve_junction(
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
        vm_tilde2, wm2 = boundary.solve_junction(
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
        vm_tilde2, wm2 = boundary.solve_junction(
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
    return np.flip(v), np.flip(w), np.flip(xi), vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wn, wm, wm, solution_found


def sound_shell_hybrid(
        model: "Model", v_wall: float, wn: float, wm: float, cs_n: float, v_cj: float,
        vp_tilde_guess: float, wp_guess: float, t_end: float, n_xi: int,
        thin_shell_limit: int,
        allow_failure: bool = False,
        allow_negative_entropy_flux_change: bool = False,
        warn_if_shock_barely_exists: bool = True) -> DeflagrationOutput:
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
    v_sh_estimate = shock.v_shock(model, wn=wn, xi=v_wall, cs_n=cs_n)
    vp_guess = relativity.lorentz(xi=v_wall, v=vp_tilde_guess)

    # More complex starting guesses
    if np.isnan(vp_tilde_guess) or vp_guess < v_sh_estimate or vp_guess < vm or np.isnan(wp_guess) or wp_guess < wn or wp_guess < wm:
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


# Solvables

def sound_shell_solvable_deflagration_reverse(
        params: np.ndarray, model: "Model", v_wall: float, wn: float, t_end: float, n_xi: int) -> float:
    xi_sh = params[0]
    # pylint: disable=unused-variable
    v, w, xi, vm, wm = sound_shell_deflagration_reverse(
        model, v_wall, wn, xi_sh, t_end=t_end, n_xi=n_xi, allow_failure=True)
    return vm


def sound_shell_solvable_deflagration(
        # params: np.ndarray,
        w_center: float, model: "Model", v_wall: float, wn: float, cs_n: float, v_cj: float,
        vp_guess: float, wp_guess: float, t_end: float, n_xi: int, thin_shell_limit: int) -> float:
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


def sound_shell_solvable_hybrid(
        # params: np.ndarray,
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
    logger.debug(
        "Hybrid solvable results: wn_target=%s, wn_computed=%s, diff=%s, wm=%s, vp=%s",
        wn, wn_estimate, diff, wm, vp
    )
    return diff


# Solvers

def sound_shell_solver_deflagration(
        model: "Model",
        start_time: float,
        v_wall: float, alpha_n: float, wn: float, cs_n: float, v_cj: float, high_alpha_n: bool,
        wm_guess: float, vp_guess: float, wp_guess: float, wn_rtol: float, t_end: float, n_xi: int,
        thin_shell_limit: int,
        allow_failure: bool, log_high_alpha_n_failures: bool = True) -> SolverOutput:
    if vp_guess > v_wall:
        vp_guess_new = 0.95 * v_wall
        if log_high_alpha_n_failures or not high_alpha_n:
            logger.error("Invalid vp_guess=%s > v_wall=%s, replacing with vp_guess=%s", vp_guess, v_wall, vp_guess_new)
        vp_guess = vp_guess_new

    sol = root_scalar(
        sound_shell_solvable_deflagration,
        x0=0.99*wm_guess,
        x1=1.01*wm_guess,
        args=(model, v_wall, wn, cs_n, v_cj, vp_guess, wp_guess, t_end, n_xi, thin_shell_limit),
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
            log_status=log_high_alpha_n_failures or not high_alpha_n
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
        v_wall: float, alpha_n: float, wn: float, t_end: float, n_xi: int) -> SolverOutput:
    # This is arbitrary and should be replaced by a value from the bag model
    xi_sh_guess = 1.1 * np.sqrt(model.cs2_max(wn, Phase.BROKEN))
    sol = fsolve(
        sound_shell_solvable_deflagration_reverse,
        xi_sh_guess,
        args=(model, v_wall, wn, t_end, n_xi),
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
    v, w, xi, wp, wm, vm = sound_shell_deflagration_reverse(model, v_wall, wn, xi_sh, t_end=t_end, n_xi=n_xi)

    return v, w, xi, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, wp, wm, np.nan, solution_found


def sound_shell_solver_hybrid(
        model: "Model",
        start_time: float,
        v_wall: float, alpha_n: float, wn: float, cs_n: float, v_cj: float, high_alpha_n: bool,
        vp_tilde_guess: float, wp_guess: float, wm_guess: float, wn_rtol: float, t_end: float, n_xi: int,
        thin_shell_limit: int,
        allow_failure: bool, log_high_alpha_n_failures: bool) -> SolverOutput:
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
        v_sh = shock.v_shock(model, wn=wn, xi=v_wall, cs_n=cs_n)
        for i, wm_i in enumerate(wms):
            vp = boundary.v_plus_hybrid(
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

        logger.debug(f"Valid wms: {valid_wms}, inds: {valid_wm_inds}")
        for i in range(vps.size):
            if vps[i] == 0:
                continue
            vp_i = vps[i]
            wm_i = wms[i]
            logger.debug(f"wm={wm_i}, vp={vp_i}")
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


# Main function

def sound_shell_generic(
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
            t_end: float = const.T_END_DEFAULT,
            n_xi: int = const.N_XI_DEFAULT,
            thin_shell_limit: int = const.THIN_SHELL_T_POINTS_MIN,
            reverse: bool = False,
            allow_failure: bool = False,
            use_bag_solver: bool = False,
            use_giese_solver: bool = False,
            log_success: bool = True,
            log_high_alpha_n_failures: bool = False
        ) -> tp.Tuple[
            np.ndarray, np.ndarray, np.ndarray, SolutionType,
            float, float, float, float, float, float, float, float, float, float, float, bool, float]:
    """Generic fluid shell solver

    In most cases you should not have to call this directly. Create a Bubble instead.
    """
    if use_giese_solver:
        return sound_shell_giese(
            model=model, v_wall=v_wall, alpha_n=alpha_n, wn=wn, wn_guess=wn_guess, wm_guess=wm_guess)

    start_time = time.perf_counter()
    if alpha_n_max_bag is None:
        alpha_n_max_bag = alpha.alpha_n_max_deflagration_bag(v_wall)
    if high_alpha_n is None:
        high_alpha_n = alpha_n > alpha_n_max_bag

    if wn is None or np.isnan(wn):
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

        v, w, xi = fluid_bag.sound_shell_bag(v_wall, alpha_n)
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
        # Deflagrations have their own method for guessing wp, so this can be nan.
        wp_guess = wp_ref * wn
    if wm_guess is None or np.isnan(wp_guess):
        using_ref = True
        if np.isnan(wm_ref):
            logger.warning(
                "No reference data for v_wall=%s, alpha_n=%s. Using an arbitrary starting guess.",
                v_wall, alpha_n
            )
            # This is arbitrary, but seems to work OK.
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
            sound_shell_detonation(
                model, v_wall, alpha_n, wn, v_cj,
                vm_tilde_guess=vm_tilde_ref, wm_guess=wm_ref, t_end=t_end, n_xi=n_xi,
            )
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
                sound_shell_solver_deflagration_reverse(
                    model, start_time, v_wall, alpha_n, wn,
                    t_end=t_end, n_xi=n_xi
                )
        else:
            v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found = \
                sound_shell_solver_deflagration(
                    model, start_time,
                    v_wall, alpha_n, wn,
                    cs_n=cs_n, v_cj=v_cj,
                    high_alpha_n=high_alpha_n,
                    wm_guess=wm_guess, vp_guess=vp_guess, wp_guess=wp_guess, wn_rtol=wn_rtol, t_end=t_end, n_xi=n_xi,
                    thin_shell_limit=thin_shell_limit,
                    allow_failure=allow_failure, log_high_alpha_n_failures=log_high_alpha_n_failures
                )
    elif sol_type == SolutionType.HYBRID:
        v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found = \
            sound_shell_solver_hybrid(
                model, start_time,
                v_wall, alpha_n, wn,
                cs_n=cs_n,
                v_cj=v_cj,
                high_alpha_n=high_alpha_n,
                vp_tilde_guess=vp_tilde_guess,
                wp_guess=wp_guess,
                wm_guess=wm_guess,
                wn_rtol=wn_rtol,
                t_end=t_end, n_xi=n_xi,
                thin_shell_limit=thin_shell_limit,
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


fluid_shell_generic = sound_shell_generic


def sound_shell_giese(
            model: "Model",
            v_wall: float,
            alpha_n: float,
            wn: float = None,
            wn_guess: float = None,
            wm_guess: float = None,
        ) -> tp.Tuple[
            np.ndarray, np.ndarray, np.ndarray, SolutionType,
            float, float, float, float, float, float, float, float, float, float, float, bool, float]:
    if kappaNuMuModel is None:
        raise ImportError("The Giese code has to be installed to use this solver.")

    start_time = time.perf_counter()

    if wn is None or np.isnan(wn):
        wn = model.w_n(alpha_n, wn_guess=wn_guess)
    if wm_guess is None or np.isnan(wm_guess):
        wm_guess = 1.

    try:
        kappa_theta_bar_n, v, wow, xi, mode, vp, vm = kappaNuMuModel(
            cs2b=model.cs2(wm_guess, Phase.BROKEN),
            cs2s=model.cs2(wn, Phase.SYMMETRIC),
            al=model.alpha_theta_bar_n_from_alpha_n(alpha_n=alpha_n, wn=wn),
            vw=v_wall
           )
    except ValueError:
        return const.nan_arr, const.nan_arr, const.nan_arr, SolutionType.ERROR, \
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, True, time.perf_counter() - start_time

    if mode == 0:
        sol_type = SolutionType.SUB_DEF
    elif mode == 1:
        sol_type = SolutionType.HYBRID
    elif mode == 2:
        sol_type = SolutionType.DETON
    else:
        raise ValueError("Got invalid mode from Giese solver:", mode)
    w: np.ndarray = wow * wn

    # Velocities in the wall frame
    vp_tilde: float = relativity.lorentz(xi=v_wall, v=vp)
    vm_tilde: float = relativity.lorentz(xi=v_wall, v=vm)

    # Shock
    v_sh: float = xi[-3]
    vm_sh: float = v[-3]
    vm_tilde_sh: float = relativity.lorentz(xi=v_sh, v=vm_sh)
    wm_sh: float = w[-3]

    # Enthalpies
    i_wall = np.argmax(v)
    wp: float = w[i_wall]
    wm: float = w[i_wall - 1]

    # Other
    v_cj = chapman_jouguet.v_chapman_jouguet(model, alpha_n=alpha_n, wn=wn, wn_guess=wn_guess)
    solution_found = True
    elapsed = time.perf_counter() - start_time

    return v, w, xi, sol_type, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, v_cj, not solution_found, elapsed
