"""Functions for fluid differential equations

Now in parametric form (Jacky Lindsay and Mike Soughton MPhys project 2017-18)
# RHS is Eq (33) in Espinosa et al (plus $\frac{dw}{dt}$ not written there)
"""

import logging
# import threading
import typing as tp

import numba
try:
    import NumbaLSODA
except ImportError:
    NumbaLSODA = None
import numpy as np
import scipy.integrate as spi

import pttools.type_hints as th
from pttools import speedup
from . import alpha
from . import bag
from . import boundary
from . import check
from . import const
from . import props
from . import transition

logger = logging.getLogger(__name__)
if NumbaLSODA is None:
    logger.warning("Could not import NumbaLSODA")

DEFAULT_DF_DTAU: str = "bag"
# ODEINT_LOCK = threading.Lock()

differentials = speedup.DifferentialCache()


def add_df_dtau(name: str, cs2_fun: bag.CS2_FUN_TYPE) -> speedup.DifferentialPointer:
    func = gen_df_dtau(cs2_fun)
    return differentials.add(name, func)


def gen_df_dtau(cs2_fun: bag.CS2_FUN_TYPE) -> speedup.Differential:
    r"""Generate a function for the differentials of fluid variables $(v, w, \xi)$ in parametric form."""
    cs2_fun_numba = cs2_fun \
        if isinstance(cs2_fun, (speedup.CFunc, speedup.Dispatcher)) \
        else numba.cfunc("float64(float64)")(cs2_fun)

    def df_dtau(t: float, u: np.ndarray, du: np.ndarray, p: np.ndarray = None) -> None:
        r"""Computes the differentials of the variables $(v, w, \xi)$ for a given $c_s^2$ function

        :return: $\frac{dv}{d\tau}, \frac{dw}{d\tau}, \frac{d\xi}{d\tau}$
        """
        v = u[0]
        w = u[1]
        xi = u[2]
        cs2 = cs2_fun_numba(w)
        xiXv = xi * v
        xi_v = xi - v
        v2 = v * v

        du[0] = 2 * v * cs2 * (1 - v2) * (1 - xiXv)  # dv/dt
        du[1] = (w / (1 - v2)) * (xi_v / (1 - xiXv)) * (1 / cs2 + 1) * du[0]  # dw_dt
        du[2] = xi * (xi_v ** 2 - cs2 * (1 - xiXv) ** 2)  # dxi/dt
    return df_dtau


DF_DTAU_BAG_PTR = add_df_dtau("bag", bag.cs2_bag_scalar)


@numba.njit
def fluid_integrate_param(
        v0: float,
        w0: float,
        xi0: float,
        t_end: float = const.T_END_DEFAULT,
        n_xi: int = const.N_XI_DEFAULT,
        df_dtau_ptr: speedup.DifferentialPointer = DF_DTAU_BAG_PTR,
        method: str = "odeint") -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Integrates parametric fluid equations in df_dtau from an initial condition.
    Positive t_end integrates along curves from (v,w) = (0,cs0) to (1,1).
    Negative t_end integrates towards (0,cs0).
    :return: $v, w, \xi, t$
    """

    t = np.linspace(0., t_end, n_xi)
    y0 = np.array([v0, w0, xi0])
    if method == "numba_lsoda":
        if NumbaLSODA is None:
            raise ImportError("NumbaLSODA is not loaded")
        v, w, xi, success = fluid_integrate_param_numba(t, y0, df_dtau_ptr)

    # This lock prevents a SystemError when running multiple threads
    # with ODEINT_LOCK:

    # SciPy differential equation solvers are not supported by Numba.
    # Putting these within numba.objmode can also be challenging, as function-type arguments are not supported.
    # For better performance, the "df_dtau" should be already fully Numba-compiled at this point instead
    # of taking functions as its arguments.
    else:
        with numba.objmode(v="float64[:]", w="float64[:]", xi="float64[:]", success="boolean"):
            if method == "odeint":
                v, w, xi, success = fluid_integrate_param_odeint(t, y0, df_dtau_ptr)
            else:
                v, w, xi, success = fluid_integrate_param_solve_ivp(t, y0, df_dtau_ptr, method)
    if not success:
        raise RuntimeError("integration failed")
    return v, w, xi, t


@numba.njit
def fluid_integrate_param_numba(t: np.ndarray, y0: np.ndarray, df_dtau_ptr: speedup.DifferentialPointer):
    backwards = t[-1] < 0
    t_numba = -t if backwards else t
    data = np.array([1 if backwards else 0])
    usol, success = NumbaLSODA.lsoda(df_dtau_ptr, u0=y0, t_eval=t_numba, data=data)
    if not success:
        with numba.objmode:
            logger.error(f"NumbaLSODA failed for %s integration", "backwards" if backwards else "forwards")
    v = usol[:, 0]
    w = usol[:, 1]
    xi = usol[:, 2]
    return v, w, xi, success


def fluid_integrate_param_odeint(t: np.ndarray, y0: np.ndarray, df_dtau_ptr: speedup.DifferentialPointer):
    try:
        func = differentials.get_odeint(df_dtau_ptr)
        soln: np.ndarray = spi.odeint(func, y0, t)
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
        t: np.ndarray, y0: np.ndarray, df_dtau_ptr: speedup.DifferentialPointer, method: str):
    try:
        func = differentials.get_solve_ivp(df_dtau_ptr)
        soln: spi._ivp.ivp.OdeResult = spi.solve_ivp(
            func, t_span=(t[0], t[-1]), y0=y0, method=method, t_eval=t)
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

@speedup.njit_if_numba_integrate
def fluid_shell(
        v_wall: float,
        alpha_n: float,
        n_xi: int = const.N_XI_DEFAULT) \
        -> tp.Union[tp.Tuple[float, float, float], tp.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""
    Finds fluid shell $(v, w, \xi)$ from a given $v_\text{wall}, \alpha_n$, which must be scalars.
    Option to change xi resolution n_xi
    """
    # check_physical_params([v_wall,alpha_n])
    sol_type = transition.identify_solution_type(v_wall, alpha_n)
    if sol_type == boundary.SolutionType.ERROR:
        with numba.objmode:
            logger.error("Giving up because of identify_solution_type error")
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr
    al_p = alpha.find_alpha_plus(v_wall, alpha_n, n_xi)
    if not np.isnan(al_p):
        # SolutionType has to be passed by its value when jitting
        return fluid_shell_alpha_plus(v_wall, al_p, sol_type.value, n_xi)
    nan_arr = np.array([np.nan])
    return nan_arr, nan_arr, nan_arr


@speedup.njit_if_numba_integrate
def fluid_shell_alpha_plus(
        v_wall: float,
        alpha_plus: float,
        sol_type: boundary.SolutionType = boundary.SolutionType.UNKNOWN,
        n_xi: int = const.N_XI_DEFAULT,
        w_n: float = 1.,
        cs2_fun: bag.CS2_FUN_TYPE = bag.cs2_bag,
        df_dtau_ptr: speedup.DifferentialPointer = DF_DTAU_BAG_PTR) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Finds fluid shell (v, w, xi) from a given $v_\text{wall}, \alpha_+$ (at-wall strength parameter).
    Where $v=0$ (behind and ahead of shell) uses only two points.
    v_wall and alpha_plus must be scalars, and are converted from 1-element arrays if needed.

    :param v_wall: $v_\text{wall}$
    :param alpha_plus: $\alpha_+$
    :param sol_type: specify wall type if more than one permitted.
    :param n_xi: increase resolution
    :param w_n: specify enthalpy outside fluid shell
    :param cs2_fun: sound speed squared as a function of enthalpy, default
    :return: $v, w, \xi$
    """
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
    vfp_w, vfm_w, vfp_p, vfm_p = boundary.fluid_speeds_at_wall(v_wall, alpha_plus, sol_type)
    wp = 1.0  # Nominal value - will be rescaled later
    wm = wp / boundary.enthalpy_ratio(vfm_w, vfp_w)  # enthalpy just behind wall

    dxi = 1. / n_xi
    # dxi = 10*eps

    # Set up parts outside shell where v=0. Need 2 points only.
    xif = np.linspace(v_wall + dxi, 1.0, 2)
    vf = np.zeros_like(xif)
    wf = np.ones_like(xif) * wp

    xib = np.linspace(min(cs2_fun(w_n) ** 0.5, v_wall) - dxi, 0.0, 2)
    vb = np.zeros_like(xib)
    wb = np.ones_like(xib) * wm

    # Integrate forward and find shock.
    if not sol_type == boundary.SolutionType.DETON.value:
        # First go
        v, w, xi, t = fluid_integrate_param(vfp_p, wp, v_wall, -const.T_END_DEFAULT, const.N_XI_DEFAULT, df_dtau_ptr)
        v, w, xi, t = trim_fluid_wall_to_shock(v, w, xi, t, sol_type)
        # Now refine so that there are ~N points between wall and shock.  A bit excessive for thin
        # shocks perhaps, but better safe than sorry. Then improve final point with shock_zoom...
        t_end_refine = t[-1]
        v, w, xi, t = fluid_integrate_param(vfp_p, wp, v_wall, t_end_refine, n_xi, df_dtau_ptr)
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
        v, w, xi, t = fluid_integrate_param(vfm_p, wm, v_wall, -const.T_END_DEFAULT, const.N_XI_DEFAULT, df_dtau_ptr)
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
        xib[0] = cs2_fun(w[-1]) ** 0.5
        xib = np.concatenate((xi, xib))

    # Now put halves together in right order
    # Need to fix this according to python version
    #    v  = np.concatenate((np.flip(vb,0),vf))
    #    w  = np.concatenate((np.flip(wb,0),wf))
    #    w  = w*(w_n/w[-1])
    #    xi = np.concatenate((np.flip(xib,0),xif))
    v = np.concatenate((np.flipud(vb), vf))
    w = np.concatenate((np.flipud(wb), wf))
    w = w * (w_n / w[-1])
    xi = np.concatenate((np.flipud(xib), xif))

    return v, w, xi


@numba.njit
def trim_fluid_wall_to_cs(
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray,
        t: np.ndarray,
        v_wall: th.FLOAT_OR_ARR,
        sol_type: boundary.SolutionType,
        dxi_lim: float = const.DXI_SMALL,
        cs2_fun: bag.CS2_FUN_TYPE = bag.cs2_bag) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Picks out fluid variable arrays $(v, w, \xi, t)$ which are definitely behind
    the wall for deflagration and hybrid.
    Also removes negative fluid speeds and xi <= sound_speed, which might be left by
    an inaccurate integration.
    If the wall is within about 1e-16 of cs, rounding errors are flagged.
    """
    check.check_wall_speed(v_wall)
    n_start = 0

    # TODO: should this be 0 to match with the error handling below?
    n_stop_index = -2
    # n_stop = 0
    if not sol_type == boundary.SolutionType.SUB_DEF.value:
        for i in range(v.size):
            if v[i] <= 0 or xi[i] ** 2 <= cs2_fun(w[i]):
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
    Trims fluid variable arrays $(v, w, \xi)$ so last element is just ahead of shock

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
