r"""Functions for fluid differential equations

Now in parametric form (Jacky Lindsay and Mike Soughton MPhys project 2017-18).
RHS is Eq (33) in Espinosa et al (plus $\frac{dw}{dt}$ not written there)
"""

import logging
import typing as tp

import numba
import numpy as np
from scipy.optimize import fsolve

import pttools.type_hints as th
from pttools import speedup
if tp.TYPE_CHECKING:
    from pttools.models.model import Model
from . import alpha
from . import approx
from . import bag
from . import boundary
from .boundary import Phase, SolutionType
from . import check
from . import chapman_jouguet
from . import const
from . import integrate
from . import props
from . import quantities
from . import relativity
from . import shock
from . import transition
from . import trim

logger = logging.getLogger(__name__)


# Main function for integrating fluid equations and deriving v, w
# for complete range 0 < xi < 1

@numba.njit
def fluid_shell(
        v_wall: float,
        alpha_n: float,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun: th.CS2Fun = bag.cs2_bag_scalar,
        cs2_fun_ptr: th.CS2FunScalarPtr = bag.CS2_BAG_SCALAR_PTR,
        df_dtau_ptr: speedup.DifferentialPointer = integrate.DF_DTAU_BAG_PTR) \
        -> tp.Union[tp.Tuple[float, float, float], tp.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""
    Finds fluid shell $(v, w, \xi)$ from a given $v_\text{wall}, \alpha_n$, which must be scalars.

    Computes $\alpha_+$ from $\alpha_n$ and then calls :py:func:`fluid_shell_alpha_plus`.

    Bag model only!

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param n_xi: number of $\xi$ points
    :return: $v, w, \xi$
    """
    # check_physical_params([v_wall,alpha_n])
    sol_type = transition.identify_solution_type_bag(v_wall, alpha_n)
    if sol_type == SolutionType.ERROR:
        with numba.objmode:
            logger.error("Giving up because of identify_solution_type error")
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr
    al_p = alpha.find_alpha_plus_bag(v_wall, alpha_n, n_xi, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr)
    if np.isnan(al_p):
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr
    # SolutionType has to be passed by its value when jitting
    return fluid_shell_alpha_plus(v_wall, al_p, sol_type.value, n_xi, cs2_fun=cs2_fun, df_dtau_ptr=df_dtau_ptr)


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
        allow_failure: bool = False) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    using_bag = False
    if vp_guess is None or wp_guess is None:
        using_bag = True
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
        SolutionType.SUB_DEF, allow_failure)


def fluid_shell_deflagration_common(
        model: "Model",
        v_wall: float, wn: float, vm_tilde: float, wm: float, vp_tilde_guess: float, wp_guess: float,
        sol_type: SolutionType,
        allow_failure: bool) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
    v, w, xi, t = integrate.fluid_integrate_param(
        v0=vp, w0=wp, xi0=v_wall,
        phase=Phase.SYMMETRIC,
        t_end=-const.T_END_DEFAULT,
        n_xi=const.N_XI_DEFAULT,
        df_dtau_ptr=model.df_dtau_ptr(),
        # method="RK45"
    )
    i_shock = shock.find_shock_index(model, v, xi, v_wall, wn, sol_type, allow_failure=allow_failure)
    v = v[:i_shock]
    w = w[:i_shock]
    xi = xi[:i_shock]

    xi_sh = xi[-1]
    vm_tilde_sh = relativity.lorentz(xi_sh, v[-1])
    wn_estimate = boundary.w2_junction(vm_tilde_sh, w[-1], xi_sh)
    return v, w, xi, wn_estimate


def fluid_shell_deflagration_reverse_solvable(params: np.ndarray, model: "Model", v_wall: float, wn: float) -> float:
    xi_sh = params[0]
    v, w, xi, vm, wm = fluid_shell_deflagration_reverse(model, v_wall, wn, xi_sh, allow_failure=True)
    return vm


def fluid_shell_deflagration_solvable(params: np.ndarray, model: "Model", v_wall: float, wn: float) -> float:
    w_center = params[0]
    v, w, xi, wn_estimate = fluid_shell_deflagration(model, v_wall, wn, w_center, allow_failure=True)
    return wn_estimate - wn


def fluid_shell_hybrid(model: "Model", v_wall: float, wn: float, wm: float, allow_failure: bool = False):
    # Exit velocity is at the sound speed
    vm_tilde = np.sqrt(model.cs2(wm, Phase.BROKEN))
    return fluid_shell_deflagration_common(
        model, v_wall, wn,
        vm_tilde, wm,
        vp_tilde_guess=0.75*vm_tilde, wp_guess=2*wm,
        sol_type=SolutionType.HYBRID,
        allow_failure=allow_failure
    )


def fluid_shell_hybrid_solvable(params: np.ndarray, model: "Model", v_wall: float, wn: float) -> float:
    wm = params[0]
    v, w, xi, wn_estimate = fluid_shell_hybrid(model, v_wall, wn, wm, allow_failure=True)
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
        f"Solving fluid shell for model={model}, v_wall={v_wall}, sol_type={sol_type}, alpha_n={alpha_n}"
    )
    sol_type = transition.validate_solution_type(
        model,
        v_wall=v_wall, alpha_n=alpha_n, sol_type=sol_type,
        wn_guess=wn_guess, wm_guess=wm_guess)

    failed = False
    wn = model.w_n(alpha_n, wn_guess=wn_guess)
    v_cj = chapman_jouguet.v_chapman_jouguet(model, alpha_n, wn, wm_guess)
    dxi = 1. / n_xi
    logger.info(f"Solved model parameters: v_cj={v_cj}, wn={wn}")

    # Detonations are the simplest case
    if sol_type == SolutionType.DETON:
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
        v, w, xi, t = trim.trim_fluid_wall_to_cs(v, w, xi, t, v_wall, sol_type, cs2_fun=model.cs2)
        w_center = w[-1]

        # Revert the order of points in the arrays for concatenation
        v = np.flip(v)
        w = np.flip(w)
        xi = np.flip(xi)

    elif sol_type == SolutionType.SUB_DEF:
        if transition.cannot_be_sub_def(model, v_wall, wn):
            raise ValueError(
                f"Invalid parameters for a subsonic deflagration: model={model}, v_wall={v_wall}, wn={wn}. "
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
                    f"Deflagration solution was not found for model={model}, v_wall={v_wall}, alpha_n={alpha_n}. "
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
                    f"Deflagration solution was not found for model={model}, v_wall={v_wall}, alpha_n={alpha_n}. "
                    f"Using w_center={w_center}. Reason: {sol[3]}"
                )
            v, w, xi, wn_estimate = fluid_shell_deflagration(
                model, v_wall, wn, w_center, allow_failure=allow_failure)
            if not np.isclose(wn_estimate, wn):
                logger.error(
                    f"Deflagration solution was not found for model={model}, v_wall={v_wall}, alpha_n={alpha_n}. "
                    f"Got wn_estimate={wn_estimate}, which differs from wn={wn}."
                )
            # print(np.array([v, w, xi]).T)
            # print("wn, xi_sh", wn, xi_sh)
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
                f"Hybrid solution was not found for model={model}, v_wall={v_wall}, alpha_n={alpha_n}. "
                f"Using wm={wm}. Reason: {sol[3]}"
            )
        v, w, xi, wn_estimate = fluid_shell_hybrid(model, v_wall, wn, wm, allow_failure=allow_failure)
        if not np.isclose(wn_estimate, wn):
            logger.error(
                f"Hybrid solution was not found for model={model}, v_wall={v_wall}, alpha_n={alpha_n}. "
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
        logger.error("Failed to find a solution. Returning approximate results.")
    else:
        logger.info("Solved fluid shell.")
    return v, w, xi


@numba.njit
def fluid_shell_alpha_plus(
        v_wall: float,
        alpha_plus: float,
        sol_type: SolutionType = SolutionType.UNKNOWN,
        n_xi: int = const.N_XI_DEFAULT,
        w_n: float = 1.,
        cs2_fun: th.CS2Fun = bag.cs2_bag,
        df_dtau_ptr: speedup.DifferentialPointer = integrate.DF_DTAU_BAG_PTR,
        sol_type_fun: callable = None) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Finds the fluid shell profile (v, w, xi) from a given $v_\text{wall}, \alpha_+$ (at-wall strength parameter).
    When $v=0$ (behind and ahead of shell), this uses only two points.

    Bag model only!

    :param v_wall: $v_\text{wall}$
    :param alpha_plus: $\alpha_+$
    :param sol_type: specify wall type if more than one permitted.
    :param n_xi: increase resolution
    :param w_n: specify enthalpy outside fluid shell
    :param cs2_fun: sound speed squared as a function of enthalpy, default
    :param df_dtau_ptr: pointer to the differential equation function
    :return: $v, w, \xi$
    """
    # These didn't work, and therefore this function gets cs2_fun as a function instead of a pointer
    # cs2_fun = bag.CS2ScalarCType(cs2_fun_ptr)
    # cs2_fun = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(cs2_fun)

    check.check_wall_speed(v_wall)

    if sol_type == SolutionType.UNKNOWN.value:
        sol_type = transition.identify_solution_type_alpha_plus(v_wall, alpha_plus).value
    # The identification above may set sol_type to error
    if sol_type == SolutionType.ERROR.value:
        with numba.objmode:
            logger.error("Giving up because of identify_solution_type error")
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr

    # Solve boundary conditions at wall
    # See the function docstring for the abbreviations
    vfp_w, vfm_w, vfp_p, vfm_p = boundary.fluid_speeds_at_wall(v_wall, alpha_plus, sol_type)
    wp = 1.0  # Nominal value - will be rescaled later
    wm = wp / boundary.enthalpy_ratio(vfm_w, vfp_w)  # enthalpy just behind wall

    dxi = 1. / n_xi
    # dxi = 10*eps

    # Set up parts outside shell where v=0. Need 2 points only.
    # Forwards integration, from v_wall to xi=1
    xif = np.linspace(v_wall + dxi, 1.0, 2)
    vf = np.zeros_like(xif)
    wf = np.ones_like(xif) * wp

    # Backwards integration, from cs or 0 to v_wall
    # TODO: set a value for the phase
    xib = np.linspace(min(cs2_fun(w_n, 0) ** 0.5, v_wall) - dxi, 0.0, 2)
    vb = np.zeros_like(xib)
    wb = np.ones_like(xib) * wm

    # TODO: remove the dependence on sol_type
    # Instead:
    # - check if the value of alpha_plus allows only one type of solution
    # - If yes, compute using that
    # - Otherwise compute both and then see which takes to the correct direction

    # Integrate forward and find shock.
    if not sol_type == SolutionType.DETON.value:
        # First go
        v, w, xi, t = integrate.fluid_integrate_param(
            v0=vfp_p, w0=wp, xi0=v_wall,
            phase=Phase.SYMMETRIC.value, t_end=-const.T_END_DEFAULT, n_xi=const.N_XI_DEFAULT, df_dtau_ptr=df_dtau_ptr)
        v, w, xi, t = trim.trim_fluid_wall_to_shock(v, w, xi, t, sol_type)
        # Now refine so that there are ~N points between wall and shock.  A bit excessive for thin
        # shocks perhaps, but better safe than sorry. Then improve final point with shock_zoom...
        t_end_refine = t[-1]
        v, w, xi, t = integrate.fluid_integrate_param(
            v0=vfp_p, w0=wp, xi0=v_wall,
            phase=Phase.SYMMETRIC.value, t_end=t_end_refine, n_xi=n_xi, df_dtau_ptr=df_dtau_ptr)
        v, w, xi, t = trim.trim_fluid_wall_to_shock(v, w, xi, t, sol_type)
        v, w, xi = shock.shock_zoom_last_element(v, w, xi)
        # Now complete to xi = 1
        vf = np.concatenate((v, vf))
        # enthalpy
        vfp_s = xi[-1]  # Fluid velocity just ahead of shock in shock frame = shock speed
        vfm_s = 1 / (3 * vfp_s)  # Fluid velocity just behind shock in shock frame
        wf = np.ones_like(xif) * w[-1] * boundary.enthalpy_ratio(vfm_s, vfp_s)
        wf = np.concatenate((w, wf))
        # xi
        xif[0] = xi[-1]
        xif = np.concatenate((xi, xif))

    # Integrate backward to sound speed.
    if not sol_type == SolutionType.SUB_DEF.value:
        # First go
        v, w, xi, t = integrate.fluid_integrate_param(
            v0=vfm_p, w0=wm, xi0=v_wall,
            phase=Phase.BROKEN.value, t_end=-const.T_END_DEFAULT, n_xi=const.N_XI_DEFAULT, df_dtau_ptr=df_dtau_ptr)
        v, w, xi, t = trim.trim_fluid_wall_to_cs(v, w, xi, t, v_wall, sol_type)
        #    # Now refine so that there are ~N points between wall and point closest to cs
        #    # For walls just faster than sound, will give very (too?) fine a resolution.
        #        t_end_refine = t[-1]
        #        v,w,xi,t = fluid_integrate_param(vfm_p, wm, v_wall, t_end_refine, n_xi, cs2_fun)
        #        v, w, xi, t = trim_fluid_wall_to_cs(v, w, xi, t, v_wall, sol_type)

        # Now complete to xi = 0
        vb = np.concatenate((v, vb))
        wb = np.ones_like(xib) * w[-1]
        wb = np.concatenate((w, wb))
        # Can afford to bring this point all the way to cs2.
        # TODO: set a value for the phase
        xib[0] = cs2_fun(w[-1], 0) ** 0.5
        xib = np.concatenate((xi, xib))

    # Now put halves together in right order
    # Need to fix this according to python version
    #    v  = np.concatenate((np.flip(vb,0),vf))
    #    w  = np.concatenate((np.flip(wb,0),wf))
    #    w  = w*(w_n/w[-1])
    #    xi = np.concatenate((np.flip(xib,0),xif))
    v = np.concatenate((np.flipud(vb), vf))
    w = np.concatenate((np.flipud(wb), wf))
    # This fixes the scaling of the results.
    # The original scaling does not matter for computing the problem, but the endpoint w[-1] has to match w_n.
    w = w * (w_n / w[-1])
    # The memory layout of the resulting xi array may cause problems with old Numba versions.
    xi = np.concatenate((np.flipud(xib), xif))
    # Using .copy() results in a contiguous memory layout, alleviating the issue above.
    return v, w, xi.copy()


def fluid_shell_params(
        v_wall: float,
        alpha_n: float,
        Np: int = const.N_XI_DEFAULT,
        low_v_approx: bool = False,
        high_v_approx: bool = False):
    if low_v_approx and high_v_approx:
        raise ValueError("Both low and high v approximations can't be enabled at the same time.")

    # TODO: use greek symbols for kappa and omega
    check.check_physical_params((v_wall, alpha_n))

    sol_type = transition.identify_solution_type_bag(v_wall, alpha_n)

    if sol_type is SolutionType.ERROR:
        raise RuntimeError(f"No solution for v_wall = {v_wall}, alpha_n = {alpha_n}")

    v, w, xi = fluid_shell(v_wall, alpha_n, Np)

    # vmax = max(v)

    xi_even = np.linspace(1 / Np, 1 - 1 / Np, Np)
    v_sh = shock.v_shock_bag(xi_even)
    w_sh = shock.wm_shock_bag(xi_even)

    n_wall = props.find_v_index(xi, v_wall)
    n_cs = int(np.floor(const.CS0 * Np))
    n_sh = xi.size - 2

    r = w[n_wall] / w[n_wall - 1]
    alpha_plus = alpha_n * w[-1] / w[n_wall]

    ubarf2 = quantities.ubarf_squared(v, w, xi, v_wall)
    # Kinetic energy fraction of total (Bag equation of state)
    ke_frac = ubarf2 / (0.75 * (1 + alpha_n))
    # Efficiency of turning Higgs potential into kinetic energy
    kappa = ubarf2 / (0.75 * alpha_n)
    # and efficiency of turning Higgs potential into thermal energy
    dw = 0.75 * quantities.mean_enthalpy_change(v, w, xi, v_wall) / (0.75 * alpha_n * w[-1])

    if high_v_approx:
        v_approx = approx.v_approx_high_alpha(xi[n_wall:n_sh], v_wall, v[n_wall])
        w_approx = approx.w_approx_high_alpha(xi[n_wall:n_sh], v_wall, v[n_wall], w[n_wall])
    elif low_v_approx:
        v_approx = approx.v_approx_low_alpha(xi, v_wall, alpha_plus)
        w_approx = approx.w_approx_low_alpha(xi, v_wall, alpha_plus)
    else:
        v_approx = None
        w_approx = None

    return {
        # Arrays
        "v": v,
        "w": w,
        "xi": xi,
        "xi_even": xi_even,
        "v_sh": v_sh,
        "w_sh": w_sh,
        "v_approx": v_approx,
        "w_approx": w_approx,
        # Scalars
        "n_wall": n_wall,
        "n_cs": n_cs,
        "n_sh": n_sh,
        "r": r,
        "alpha_plus": alpha_plus,
        "ubarf2": ubarf2,
        "ke_frac": ke_frac,
        "kappa": kappa,
        "dw": dw,
        "sol_type": sol_type
    }
