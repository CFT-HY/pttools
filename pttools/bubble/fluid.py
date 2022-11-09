r"""Functions for fluid differential equations

Now in parametric form (Jacky Lindsay and Mike Soughton MPhys project 2017-18).
RHS is Eq (33) in Espinosa et al (plus $\frac{dw}{dt}$ not written there)
"""
# import ctypes
import logging
# import threading
import typing as tp

import numba
import numpy as np
import scipy.integrate as spi
import scipy.optimize

import pttools.type_hints as th
from pttools import speedup
from pttools.speedup.numba_wrapper import numbalsoda
if tp.TYPE_CHECKING:
    from pttools.models.model import Model
from . import alpha
from . import approx
from . import bag
from . import boundary
from . import check
from . import chapman_jouguet
from . import const
from . import props
from . import quantities
from . import relativity
from . import transition

logger = logging.getLogger(__name__)

DEFAULT_DF_DTAU: str = "bag"
# ODEINT_LOCK = threading.Lock()

#: Cache for the differential equations.
#: New differential equations have to be added here before usage so that they can be found by
#: :func:`scipy.integrate.odeint` and :func:`scipy.integrate.solve_ivp`.
differentials = speedup.DifferentialCache()

#
# class FluidShellParams:
#     def __init__(self):
#
#
# arrs = {
#     "v": v,
#     "w": w,
#     "xi": xi,
#     "v_sh": v_sh,
#     "w_sh": w_sh
# }
# scalars = {
#     "n_wall": n_wall,
#     "n_cs": n_cs,
#     "n_sh": n_sh,
#     "r": r,
#     "alpha_plus": alpha_plus,
#     "ubarf2": ubarf2,
#     "ke_frac": ke_frac,
#     "kappa": kappa,
#     "dw": dw
# }


# Todo: Think whether the differentials should be stored in the models.

def add_df_dtau(name: str, cs2_fun: bag.CS2Fun) -> speedup.DifferentialPointer:
    """Add a new differential equation to the cache based on the given sound speed function.

    :param name: the name of the function
    :param cs2_fun: function, which gives the speed of sound squared $c_s^2$.
    :return:
    """
    func = gen_df_dtau(cs2_fun)
    return differentials.add(name, func)


def gen_df_dtau(cs2_fun: bag.CS2Fun) -> speedup.Differential:
    r"""Generate a function for the differentials of fluid variables $(v, w, \xi)$ in parametric form.
    The parametrised differential equation is as in :gw_pt_ssm:`\ ` eq. B.14-16:

    - $\frac{dv}{dt} = 2v c_s^2 (1-v^2) (1 - \xi v)$
    - $\frac{dw}{dt} = \frac{w}{1-v^2} \frac{\xi - v}{1 - \xi v} (\frac{1}{c_s^2}+1) \frac{dv}{dt}$
    - $\frac{d\xi}{dt} = \xi \left( (\xi - v)^2 - c_s^2 (1 - \xi v)^2 \right)$

    :param cs2_fun: function, which gives the speed of sound squared $c_s^2$.
    :return: function for the differential equation
    """
    cs2_fun_numba = cs2_fun \
        if isinstance(cs2_fun, (speedup.CFunc, speedup.Dispatcher)) \
        else numba.cfunc("float64(float64, float64)")(cs2_fun)

    def df_dtau(t: float, u: np.ndarray, du: np.ndarray, args: np.ndarray = None) -> None:
        r"""Computes the differentials of the variables $(v, w, \xi)$ for a given $c_s^2$ function

        :param t: "time"
        :param u: point
        :param du: derivatives
        :param args: extra arguments: [phase]
        :return: $\frac{dv}{d\tau}, \frac{dw}{d\tau}, \frac{d\xi}{d\tau}$
        """
        v = u[0]
        w = u[1]
        xi = u[2]
        phase = args[0]
        cs2 = cs2_fun_numba(w, phase)
        xiXv = xi * v
        xi_v = xi - v
        v2 = v * v

        du[0] = 2 * v * cs2 * (1 - v2) * (1 - xiXv)  # dv/dt
        du[1] = (w / (1 - v2)) * (xi_v / (1 - xiXv)) * (1 / cs2 + 1) * du[0]  # dw_dt
        du[2] = xi * (xi_v ** 2 - cs2 * (1 - xiXv) ** 2)  # dxi/dt
    return df_dtau


#: Pointer to the differential equation of the bag model
DF_DTAU_BAG_PTR = add_df_dtau("bag", bag.cs2_bag_scalar_cfunc if speedup.NUMBA_DISABLE_JIT else bag.cs2_bag)


@numba.njit
def fluid_integrate_param(
        v0: float,
        w0: float,
        xi0: float,
        phase: float = -1.,
        t_end: float = const.T_END_DEFAULT,
        n_xi: int = const.N_XI_DEFAULT,
        df_dtau_ptr: speedup.DifferentialPointer = DF_DTAU_BAG_PTR,
        method: str = "odeint") -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Integrates parametric fluid equations in df_dtau from an initial condition.
    Positive t_end integrates along curves from $(v,w) = (0,c_{s,0})$ to $(1,1)$.
    Negative t_end integrates towards $(0,c_s{s,0})$.

    :param v0: $v_0$
    :param w0: $w_0$
    :param xi0: $\xi_0$
    :param phase: phase $\phi$
    :param t_end: $t_\text{end}$
    :param n_xi: number of $\xi$ points
    :param df_dtau_ptr: pointer to the differential equation function
    :param method: differential equation solver to be used
    :return: $v, w, \xi, t$
    """
    if phase < 0.:
        print("The phase has not been set! Assuming symmetric phase.")
        phase = 0.

    t = np.linspace(0., t_end, n_xi)
    y0 = np.array([v0, w0, xi0])
    # The second value ensures that the Numba typing is correct.
    data = np.array([phase, 0.])
    if method == "numba_lsoda" or speedup.NUMBA_INTEGRATE:
        if numbalsoda is None:
            raise ImportError("NumbaLSODA is not loaded")
        v, w, xi, success = fluid_integrate_param_numba(t=t, y0=y0, data=data, df_dtau_ptr=df_dtau_ptr)

    # This lock prevents a SystemError when running multiple threads
    # with ODEINT_LOCK:

    # SciPy differential equation solvers are not supported by Numba.
    # Putting these within numba.objmode can also be challenging, as function-type arguments are not supported.
    # For better performance, the "df_dtau" should be already fully Numba-compiled at this point instead
    # of taking functions as its arguments.
    else:
        with numba.objmode(v="float64[:]", w="float64[:]", xi="float64[:]", success="boolean"):
            if method == "odeint":
                v, w, xi, success = fluid_integrate_param_odeint(t=t, y0=y0, data=data, df_dtau_ptr=df_dtau_ptr)
            else:
                v, w, xi, success = fluid_integrate_param_solve_ivp(
                    t=t, y0=y0, data=data, df_dtau_ptr=df_dtau_ptr, method=method)
    if not success:
        raise RuntimeError("integration failed")
    return v, w, xi, t


@numba.njit
def fluid_integrate_param_numba(t: np.ndarray, y0: np.ndarray, data: np.ndarray, df_dtau_ptr: speedup.DifferentialPointer):
    r"""Integrate a differential equation using NumbaLSODA.

    :param t: time
    :param y0: starting point
    :param data: constants
    :param df_dtau_ptr: pointer to the differential equation function
    :return: $v, w, \xi$, success status
    """
    if speedup.NUMBA_DISABLE_JIT:
        raise NotImplementedError("NumbaLSODA is supported only when jitting is enabled")

    backwards = t[-1] < 0
    t_numba = -t if backwards else t
    data_numba = np.zeros((data.size + 1))
    data_numba[:-1] = data
    # Numba does not support float(bool)
    data_numba[-1] = int(backwards)
    usol, success = numbalsoda.lsoda(df_dtau_ptr, u0=y0, t_eval=t_numba, data=data_numba)
    if not success:
        with numba.objmode:
            logger.error(f"NumbaLSODA failed for %s integration", "backwards" if backwards else "forwards")
    v = usol[:, 0]
    w = usol[:, 1]
    xi = usol[:, 2]
    return v, w, xi, success


def fluid_integrate_param_odeint(t: np.ndarray, y0: np.ndarray, data: np.ndarray, df_dtau_ptr: speedup.DifferentialPointer):
    r"""Integrate a differential equation using :func:`scipy.integrate.odeint`.

    :param t: time
    :param y0: starting point
    :param df_dtau_ptr: pointer to the differential equation function, which is already in the cache
    :return: $v, w, \xi$, success status
    """
    try:
        func = differentials.get_odeint(df_dtau_ptr)
        soln: np.ndarray = spi.odeint(func, y0=y0, t=t, args=(data,))
        v = soln[:, 0]
        w = soln[:, 1]
        xi = soln[:, 2]
        success = True
    except Exception as e:
        logger.exception("odeint failed", exc_info=e)
        v = w = xi = np.zeros_like(t)
        success = False
    return v, w, xi, success


def fluid_integrate_param_solve_ivp(
        t: np.ndarray, y0: np.ndarray, data: np.ndarray, df_dtau_ptr: speedup.DifferentialPointer, method: str):
    """Integrate a differential equation using :func:`scipy.integrate.solve_ivp`.

    :param t: time
    :param y0: starting point
    :param df_dtau_ptr: pointer to the differential equation function, which is already in the cache
    :param method: name of the integrator to be used. See the :func:`scipy.integrate.solve_ivp` documentation.
    """
    try:
        func = differentials.get_solve_ivp(df_dtau_ptr)
        soln: spi._ivp.ivp.OdeResult = spi.solve_ivp(
            func, t_span=(t[0], t[-1]), y0=y0, method=method, t_eval=t, args=(data,))
        v = soln.y[0, :]
        w = soln.y[1, :]
        xi = soln.y[2, :]
        success = True
    except Exception as e:
        logger.exception("solve_ivp failed", exc_info=e)
        v = w = xi = np.zeros_like(t)
        success = False
    return v, w, xi, success


# Main function for integrating fluid equations and deriving v, w
# for complete range 0 < xi < 1

@numba.njit
def fluid_shell(
        v_wall: float,
        alpha_n: float,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun: bag.CS2Fun = bag.cs2_bag_scalar,
        cs2_fun_ptr: bag.CS2FunScalarPtr = bag.CS2_BAG_SCALAR_PTR,
        df_dtau_ptr: speedup.DifferentialPointer = DF_DTAU_BAG_PTR) \
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
    sol_type = transition.identify_solution_type(v_wall, alpha_n)
    if sol_type == boundary.SolutionType.ERROR:
        with numba.objmode:
            logger.error("Giving up because of identify_solution_type error")
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr
    al_p = alpha.find_alpha_plus(v_wall, alpha_n, n_xi, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr)
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
    vm_sh = props.v_shock(xi_sh)
    wm_sh = props.wm_shock(xi_sh, wn)

    # Integrate from the shock to the wall
    logger.info(
        f"Integrating deflagration with v_wall={v_wall}, wn={wn} from vm_sh={vm_sh}, wm_sh={wm_sh}, xi_sh={xi_sh}")
    v, w, xi, t = fluid_integrate_param(
        v0=vm_sh, w0=wm_sh, xi0=xi_sh,
        phase=boundary.Phase.SYMMETRIC,
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
        boundary.Phase.SYMMETRIC, boundary.Phase.BROKEN,
        v2_guess=v_wall, w2_guess=wp,
        allow_failure=allow_failure
    )
    vm = relativity.lorentz(vm_tilde, v_wall)

    return v, w, xi, vm, wm


def fluid_shell_deflagration(
        model: "Model",
        v_wall: float, alpha_n: float, wn: float, w_center: float,
        vp_guess: float = None, wp_guess: float = None,
        allow_failure: bool = False) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if vp_guess is None or wp_guess is None:
        # Use bag model as the starting guess

        # alpha_plus_bag = alpha.find_alpha_plus(v_wall, alpha_n, n_xi=const.N_XI_DEFAULT)
        # vp_tilde_bag, vm_tilde_bag, vp_bag, vm_bag = boundary.fluid_speeds_at_wall(
        #     v_wall, alpha_p=alpha_plus_bag, sol_type=boundary.SolutionType.SUB_DEF)
        # wp_bag = boundary.w2_junction(vm_tilde_bag, w_center, vp_tilde_bag)
        # vp_tilde_bag, wp_bag = bag.junction_bag(v_wall, w_center, 0, 1, greater_branch=False)

        # The boundary conditions are symmetric with respect to the indices,
        # and can therefore be used with the opposite indices.
        Vp = 1
        Vm = 0
        alpha_minus = 4*(Vm - Vp)/(3*w_center)
        vp_tilde_guess = boundary.v_minus(vp=v_wall, ap=alpha_minus, sol_type=boundary.SolutionType.SUB_DEF)
        vp_guess = -relativity.lorentz(vp_tilde_guess, v_wall)
        wp_guess = boundary.w2_junction(v_wall, w_center, vp_tilde_guess)
    else:
        vp_tilde_guess = relativity.lorentz(vp_guess, v_wall)

    invalid_param = None
    if np.isnan(vp_tilde_guess) < 0 or vp_tilde_guess > 1:
        invalid_param = "vp_tilde_guess"
    elif vp_guess < 0 or vp_guess > v_wall:
        invalid_param = "vp_guess"
    elif np.isnan(wp_guess) or wp_guess < wn:
        invalid_param = "wp_guess"
    if invalid_param is not None:
        logger.error(
            f"Invalid parameter: {invalid_param}. Got: "
            f"v_wall={v_wall}, w_center={w_center}, "
            f"vp_tilde_guess={vp_tilde_guess}, vp_guess={vp_guess}, wp_guess={wp_guess}"
        )
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr, np.nan, np.nan

    # Solve the boundary conditions at the wall
    vp_tilde, wp = boundary.solve_junction(
        model, v_wall, w_center,
        boundary.Phase.BROKEN, boundary.Phase.SYMMETRIC,
        v2_guess=vp_tilde_guess, w2_guess=wp_guess,
        allow_failure=allow_failure
    )
    vp = -relativity.lorentz(vp_tilde, v_wall)
    # logger.debug(f"vp_tilde={vp_tilde}, vp={vp}, wp={wp}")

    # Integrate from the wall to the shock
    v, w, xi, t = fluid_integrate_param(
        v0=vp, w0=wp, xi0=v_wall,
        phase=boundary.Phase.SYMMETRIC,
        t_end=-const.T_END_DEFAULT,
        n_xi=const.N_XI_DEFAULT,
        df_dtau_ptr=model.df_dtau_ptr(),
        # method="RK45"
    )
    # Trim the integration to the shock
    v_shock = props.v_shock(xi)
    i_shock = np.argmax(v <= v_shock)
    if i_shock == 0:
        if np.max(xi) < const.CS0:
            msg = \
                "The curve turns backwards before reaching the shock. " + \
                "Probably the model does not allow deflagrations with these parameters."
            logger.error(msg)
            if not allow_failure:
                raise RuntimeError(msg)
        i_shock = -1
    v = v[:i_shock]
    w = w[:i_shock]
    xi = xi[:i_shock]

    xi_sh = xi[-1]
    wm_sh = w[-1]
    wn = props.wp_shock(xi_sh, wm_sh)

    return v, w, xi, wn, xi_sh


def fluid_shell_deflagration_reverse_solvable(params: np.ndarray, model: "Model", v_wall: float, wn: float):
    xi_sh = params[0]
    v, w, xi, vm, wm = fluid_shell_deflagration_reverse(model, v_wall, wn, xi_sh, allow_failure=True)
    return vm


def fluid_shell_deflagration_solvable(params: np.ndarray, model: "Model", v_wall: float, alpha_n: float, wn: float):
    w_center = params[0]
    v, w, xi, wn_estimate, xi_sh = fluid_shell_deflagration(model, v_wall, alpha_n, wn, w_center, allow_failure=True)
    return wn_estimate - wn


def fluid_shell_generic(
            model: "Model",
            v_wall: float,
            alpha_n: float,
            sol_type: tp.Optional[boundary.SolutionType] = None,
            wn_guess: float = 1,
            wm_guess: float = 2,
            n_xi: int = const.N_XI_DEFAULT,
            reverse: bool = False,
            allow_failure: bool = False
        ):
    logger.info(
        f"Solving fluid shell for model={model}, v_wall={v_wall}, sol_type={sol_type}, alpha_n={alpha_n}"
    )
    if sol_type is None or sol_type is boundary.SolutionType.UNKNOWN:
        sol_type = transition.identify_solution_type_beyond_bag(model, v_wall, alpha_n, wn_guess, wm_guess)
    if sol_type is boundary.SolutionType.UNKNOWN:
        msg = \
            f"Could not determine solution type automatically for model={model}, v_wall={v_wall}, alpha_n={alpha_n}. " \
            "Please choose it manually."
        logger.error(msg)
        raise ValueError(msg)

    failed = False
    v_cj = chapman_jouguet.v_chapman_jouguet(model, alpha_n, wn_guess, wm_guess)
    wn = model.w_n(alpha_n, wn_guess=wn_guess)
    dxi = 1. / n_xi
    logger.info(f"Solved model parameters: v_cj={v_cj}, wn={wn}")

    # Detonations are the simplest case
    if sol_type == boundary.SolutionType.DETON:
        if transition.cannot_be_detonation(v_wall, v_cj):
            raise ValueError(f"Too slow wall speed for a detonation: v_wall={v_wall}, v_cj={v_cj}")
        # Use bag model as the starting point
        vp_tilde_bag, vm_tilde_bag, vp_bag, vm_bag = boundary.fluid_speeds_at_wall(
            v_wall, alpha_p=alpha_n, sol_type=boundary.SolutionType.DETON)
        wm_bag = boundary.w2_junction(v1=vp_tilde_bag, w1=wn, v2=vm_tilde_bag)
        # Solve junction conditions
        vm_tilde, wm = boundary.solve_junction(
            model,
            v1=v_wall, w1=wn,
            phase1=boundary.Phase.SYMMETRIC, phase2=boundary.Phase.BROKEN,
            v2_guess=vm_tilde_bag, w2_guess=wm_bag)

        # Convert to the plasma frame
        vm = relativity.lorentz(v_wall, vm_tilde)

        v, w, xi, t = fluid_integrate_param(
            v0=vm, w0=wm, xi0=v_wall,
            phase=boundary.Phase.BROKEN,
            t_end=-const.T_END_DEFAULT,
            n_xi=const.N_XI_DEFAULT,
            df_dtau_ptr=model.df_dtau_ptr()
        )
        v, w, xi, t = trim_fluid_wall_to_cs(v, w, xi, t, v_wall, sol_type, cs2_fun=model.cs2)
        w_center = w[-1]

        # Revert the order of points in the arrays for concatenation
        v = np.flip(v)
        w = np.flip(w)
        xi = np.flip(xi)

    elif sol_type == boundary.SolutionType.SUB_DEF:
        if transition.cannot_be_sub_def(v_wall, model, wn):
            raise ValueError("Invalid parameters for a subsonic deflagration")

        # Todo: In more advanced models,
        #  the direction of the integration will probably have to be determined by trial and error.

        if reverse:
            xi_sh_guess = 1.05*np.sqrt(transition.max_cs2_inside_def(model, wn))
            sol = scipy.optimize.fsolve(
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
            sol = scipy.optimize.fsolve(
                fluid_shell_deflagration_solvable,
                wm_guess,
                args=(model, v_wall, alpha_n, wn),
                full_output=True
            )
            w_center = sol[0][0]
            if sol[2] != 1:
                failed = True
                logger.error(
                    f"Deflagration solution was not found for model={model}, v_wall={v_wall}, alpha_n={alpha_n}. "
                    f"Using w_center={w_center}. Reason: {sol[3]}"
                )
            v, w, xi, wn, xi_sh = fluid_shell_deflagration(
                model, v_wall, alpha_n, wn, w_center, allow_failure=allow_failure)
            logger.info(f"Deflagration: w_center={w_center}, wn={wn}")
            # print(np.array([v, w, xi]).T)
            # print("wn, xi_sh", wn, xi_sh)
    elif sol_type == boundary.SolutionType.HYBRID:
        raise NotImplementedError
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
        sol_type: boundary.SolutionType = boundary.SolutionType.UNKNOWN,
        n_xi: int = const.N_XI_DEFAULT,
        w_n: float = 1.,
        cs2_fun: bag.CS2Fun = bag.cs2_bag,
        df_dtau_ptr: speedup.DifferentialPointer = DF_DTAU_BAG_PTR,
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

    if sol_type == boundary.SolutionType.UNKNOWN.value:
        sol_type = transition.identify_solution_type_alpha_plus(v_wall, alpha_plus).value
    # The identification above may set sol_type to error
    if sol_type == boundary.SolutionType.ERROR.value:
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
    if not sol_type == boundary.SolutionType.DETON.value:
        # First go
        v, w, xi, t = fluid_integrate_param(
            v0=vfp_p, w0=wp, xi0=v_wall, t_end=-const.T_END_DEFAULT, n_xi=const.N_XI_DEFAULT, df_dtau_ptr=df_dtau_ptr)
        v, w, xi, t = trim_fluid_wall_to_shock(v, w, xi, t, sol_type)
        # Now refine so that there are ~N points between wall and shock.  A bit excessive for thin
        # shocks perhaps, but better safe than sorry. Then improve final point with shock_zoom...
        t_end_refine = t[-1]
        v, w, xi, t = fluid_integrate_param(
            v0=vfp_p, w0=wp, xi0=v_wall, t_end=t_end_refine, n_xi=n_xi, df_dtau_ptr=df_dtau_ptr)
        v, w, xi, t = trim_fluid_wall_to_shock(v, w, xi, t, sol_type)
        v, w, xi = props.shock_zoom_last_element(v, w, xi)
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
    if not sol_type == boundary.SolutionType.SUB_DEF.value:
        # First go
        v, w, xi, t = fluid_integrate_param(
            v0=vfm_p, w0=wm, xi0=v_wall, t_end=-const.T_END_DEFAULT, n_xi=const.N_XI_DEFAULT, df_dtau_ptr=df_dtau_ptr)
        v, w, xi, t = trim_fluid_wall_to_cs(v, w, xi, t, v_wall, sol_type)
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

    sol_type = transition.identify_solution_type(v_wall, alpha_n)

    if sol_type is boundary.SolutionType.ERROR:
        raise RuntimeError(f"No solution for v_wall = {v_wall}, alpha_n = {alpha_n}")

    v, w, xi = fluid_shell(v_wall, alpha_n, Np)

    # vmax = max(v)

    xi_even = np.linspace(1 / Np, 1 - 1 / Np, Np)
    v_sh = props.v_shock(xi_even)
    w_sh = props.wm_shock(xi_even)

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


@numba.njit
def trim_fluid_wall_to_cs(
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray,
        t: np.ndarray,
        v_wall: th.FloatOrArr,
        sol_type: boundary.SolutionType,
        dxi_lim: float = const.DXI_SMALL,
        cs2_fun: bag.CS2Fun = bag.cs2_bag) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Picks out fluid variable arrays $(v, w, \xi, t)$ which are definitely behind
    the wall for detonation and hybrid.
    Also removes negative fluid speeds and $\xi \leq c_s$, which might be left by
    an inaccurate integration.
    If the wall is within about 1e-16 of cs, rounding errors are flagged.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param t: $t$
    :param v_wall: $v_\text{wall}$
    :param sol_type: solution type
    :param dxi_lim: not used
    :param cs2_fun: function, which gives $c_s^2$
    :return: trimmed $v, w, \xi, t$
    """
    check.check_wall_speed(v_wall)
    n_start = 0

    # TODO: should this be 0 to match with the error handling below?
    n_stop_index = -2
    # n_stop = 0
    if sol_type != boundary.SolutionType.SUB_DEF.value:
        for i in range(v.size):
            if v[i] <= 0 or xi[i] ** 2 <= cs2_fun(w[i], boundary.Phase.BROKEN.value):
                n_stop_index = i
                break

    if n_stop_index == 0:
        with numba.objmode:
            logger.warning((
                "Integation gave v < 0 or xi <= cs. "
                "sol_type: {}, v_wall: {}, xi[0] = {}, v[0] = {}. "
                "Fluid profile has only one element between vw and cs. "
                "Fix implemented by adding one extra point.").format(sol_type, v_wall, xi[0], v[0]))
        n_stop = 1
    else:
        n_stop = n_stop_index

    if (xi[0] == v_wall) and not (sol_type == boundary.SolutionType.DETON.value):
        n_start = 1
        n_stop += 1

    return v[n_start:n_stop], w[n_start:n_stop], xi[n_start:n_stop], t[n_start:n_stop]


@numba.njit
def trim_fluid_wall_to_shock(
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray,
        t: np.ndarray,
        sol_type: boundary.SolutionType) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Trims fluid variable arrays $(v, w, \xi)$ so last element is just ahead of shock.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param t: $t$
    :param sol_type: solution type
    :return: trimmed $v, w, \xi, t$
    """
    # TODO: should this be 0 to match with the error handling below?
    n_shock_index = -2
    # n_shock = 0
    if sol_type != boundary.SolutionType.DETON.value:
        for i in range(v.size):
            if v[i] <= props.v_shock(xi[i]):
                n_shock_index = i
                break

    if n_shock_index == 0:
        with numba.objmode:
            # F-strings are not yet supported by Numba, even in object mode.
            # https://github.com/numba/numba/issues/3250
            logger.warning((
                "v[0] < v_shock(xi[0]). "
                "sol_type: {}, xi[0] = {}, v[0] = {}, v_sh(xi[0]) = {}. "
                "Shock profile has only one element. Fix implemented by adding one extra point.").format(
                sol_type, xi[0], v[0], props.v_shock(xi[0])
            ))
        n_shock = 1
    else:
        n_shock = n_shock_index

    return v[:n_shock + 1], w[:n_shock + 1], xi[:n_shock + 1], t[:n_shock + 1]
