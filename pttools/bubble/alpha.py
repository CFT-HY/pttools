r"""
Functions for computing $\alpha_n$, the strength parameter at nucleation temperature,
and $\alpha_+$, the strength parameter just in front of the wall.
"""

import ctypes
import threading
import typing as tp

import numba
import numpy as np
import scipy.optimize

import pttools.type_hints as th
from pttools import speedup
from pttools.bubble import bag
from pttools.bubble import boundary
from pttools.bubble import const
from pttools.bubble import fluid
from pttools.bubble import check
from pttools.bubble import props
from pttools.bubble import transition


CS2CFunc = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
CS2CACHE: tp.Dict[bag.CS2FunScalarPtr, CS2CFunc] = {}
find_alpha_plus_scalar_lock = threading.Lock()


@numba.njit
def alpha_n_max(v_wall: th.FloatOrArr, n_xi: int = const.N_XI_DEFAULT) -> th.FloatOrArr:
    r"""
    Calculates the relative trace anomaly outside the bubble, $\alpha_{n,\max}$,
    for given $v_\text{wall}$, which is max $\alpha_n$ for (supersonic) deflagration.

    :param v_wall: $v_\text{wall}$
    :param n_xi: number of $\xi$ points
    :return: $\alpha_{n,\max}$, the relative trace anomaly outside the bubble
    """
    return alpha_n_max_deflagration(v_wall, n_xi)


@numba.njit
def _alpha_n_max_deflagration_scalar(v_wall: float, n_xi: int) -> float:
    check.check_wall_speed(v_wall)
    # TODO: This may not be correct, as it makes an explicit reference to const.CS0
    # At least there is circular logic due to the call to fluid_shell_alpha_plus
    # TODO: This line is for the bag model only, as cs should depend on enthalpy and phase
    sol_type = boundary.SolutionType.HYBRID.value if v_wall > const.CS0 else boundary.SolutionType.SUB_DEF.value
    ap = 1. / 3 - 1.0e-10  # Warning - this is not safe.  Causes warnings for v low vw
    _, w, xi = fluid.fluid_shell_alpha_plus(v_wall, ap, sol_type, n_xi)
    n_wall = props.find_v_index(xi, v_wall)
    return w[n_wall + 1] * (1. / 3)


@numba.njit(parallel=True)
def _alpha_n_max_deflagration_arr(v_wall: np.ndarray, n_xi: int) -> np.ndarray:
    ret = np.zeros_like(v_wall)
    for i in numba.prange(v_wall.size):
        ret[i] = _alpha_n_max_deflagration_scalar(v_wall[i], n_xi)
    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalised to 1 at large xi
    # Need n_wall+1, as w is an integral of v, and lags by 1 step
    return ret


@numba.generated_jit(nopython=True)
def alpha_n_max_deflagration(v_wall: th.FloatOrArr, n_xi: int = const.N_XI_DEFAULT) -> th.FloatOrArrNumba:
    r"""
    Calculates the relative trace anomaly outside the bubble, $\alpha_{n,\max}$,
    for given $v_\text{wall}$, for deflagration.
    Works also for hybrids, as they are supersonic deflagrations.

    :param v_wall: $v_\text{wall}$
    :param n_xi: number of $\xi$ points
    :return: $\alpha_{n,\max}$
    """
    if isinstance(v_wall, numba.types.Float):
        return _alpha_n_max_deflagration_scalar
    if isinstance(v_wall, numba.types.Array):
        if not v_wall.ndim:
            return _alpha_n_max_deflagration_scalar
        return _alpha_n_max_deflagration_arr
    if isinstance(v_wall, float):
        return _alpha_n_max_deflagration_scalar(v_wall, n_xi)
    if isinstance(v_wall, np.ndarray):
        if not v_wall.ndim:
            return _alpha_n_max_deflagration_scalar(v_wall.item(), n_xi)
        return _alpha_n_max_deflagration_arr(v_wall, n_xi)
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


@numba.njit
def alpha_n_max_detonation(v_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Maximum allowed value of $\alpha_n$ for a detonation with wall speed $v_\text{wall}$.
    Same as :func:`alpha_plus_max_detonation`, since for a detonation $\alpha_n = \alpha_+$.

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{n,\max,\text{detonation}}$
    """
    return alpha_plus_max_detonation(v_wall)


def alpha_n_max_hybrid(v_wall: float, n_xi: int = const.N_XI_DEFAULT) -> float:
    r"""
    Calculates the relative trace anomaly outside the bubble, $\alpha_{n,\max}$,
    for given $v_\text{wall}$, assuming hybrid fluid shell

    :param v_wall: $v_\text{wall}$
    :param n_xi: number of $\xi$ points
    :return: $\alpha_{n,\max}$
    """
    sol_type = transition.identify_solution_type_alpha_plus(v_wall, 1. / 3)
    if sol_type == boundary.SolutionType.SUB_DEF:
        raise ValueError("Alpha_n_max_hybrid was called with v_wall < cs. Use alpha_n_max_deflagration instead.")

    # Might have been returned as "Detonation, which takes precedence over Hybrid
    sol_type = boundary.SolutionType.HYBRID
    ap = 1. / 3 - 1e-8
    _, w, xi = fluid.fluid_shell_alpha_plus(v_wall, ap, sol_type, n_xi)
    n_wall = props.find_v_index(xi, v_wall)

    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalised to 1 at large xi
    return w[n_wall] * (1. / 3)


@numba.njit
def alpha_n_min_deflagration(v_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Minimum $\alpha_n$ for a deflagration. Equal to maximum $\alpha_n$ for a detonation.
    Same as :func:`alpha_n_min_hybrid`, as a hybrid is a supersonic deflagration.

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{n,\min,\text{deflagration}} = \alpha_{n,\min,\text{hybrid}} = \alpha_{n,\max,\text{detonation}}$
    """
    # This check is implemented in the inner functions
    # check.check_wall_speed(v_wall)
    return alpha_n_max_detonation(v_wall)


@numba.njit
def alpha_n_min_hybrid(v_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Minimum $\alpha_n$ for a hybrid. Equal to maximum $\alpha_n$ for a detonation.
    Same as :func:`alpha_n_min_deflagration`, as a hybrid is a supersonic deflagration.

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{n,\min,\text{hybrid}} = \alpha_{n,\min,\text{deflagration}} = \alpha_{n,\max,\text{detonation}}$
    """
    # This check is implemented in the inner functions
    # check.check_wall_speed(v_wall)
    return alpha_n_max_detonation(v_wall)


@numba.njit
def alpha_plus_initial_guess(v_wall: th.FloatOrArr, alpha_n_given: float) -> th.FloatOrArr:
    r"""
    Initial guess for root-finding of $\alpha_+$ from $\alpha_n$.
    Linear approx between $\alpha_{n,\min}$ and $\alpha_{n,\max}$.
    Doesn't do obvious checks like Detonation - needs improving?

    :param v_wall: $v_\text{wall}$, wall speed
    :param alpha_n_given: $\alpha_{n, \text{given}}$
    :return: initial guess for $\alpha_+$
    """
    if alpha_n_given < 0.05:
        return alpha_n_given

    alpha_plus_min = alpha_plus_min_hybrid(v_wall)
    alpha_plus_max = 1. / 3

    alpha_n_min = alpha_n_min_hybrid(v_wall)
    alpha_n_max = alpha_n_max_deflagration(v_wall)

    slope = (alpha_plus_max - alpha_plus_min) / (alpha_n_max - alpha_n_min)
    return alpha_plus_min + slope * (alpha_n_given - alpha_n_min)


@speedup.vectorize(nopython=True)
def alpha_plus_max_detonation(v_wall: th.FloatOrArr) -> th.FloatOrArrNumba:
    r"""
    Maximum allowed value of $\alpha_+$ for a detonation with wall speed $v_\text{wall}$.
    Comes from inverting $v_w$ > $v_\text{Jouguet}$.

    $\alpha_{+,\max,\text{detonation}} = \frac{ (1 - \sqrt{3} v_\text{wall})^2 }{ 3(1 - v_\text{wall}^2 }$
    """
    check.check_wall_speed(v_wall)
    if v_wall < const.CS0:
        return 0
    a = 3 * (1 - v_wall ** 2)
    b = (1 - np.sqrt(3) * v_wall) ** 2
    return b / a


@speedup.vectorize(nopython=True)
def alpha_plus_min_hybrid(v_wall: th.FloatOrArr) -> th.FloatOrArrNumba:
    r"""
    Minimum allowed value of $\alpha_+$ for a hybrid with wall speed $v_\text{wall}$.
    Condition from coincidence of wall and shock.

    $\alpha_{+, \min, \text{hybrid}} = \frac{ (1 - \sqrt{3} v_\text{wall})^2 }{ 9 v_\text{wall}^2 - 1}$

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{+, \min, \text{hybrid}}$
    """
    check.check_wall_speed(v_wall)
    if v_wall < const.CS0:
        return 0
    b = (1 - np.sqrt(3) * v_wall) ** 2
    c = 9 * v_wall ** 2 - 1
    return b / c


@numba.njit
def find_alpha_n(
        v_wall: th.FloatOrArr,
        alpha_p: float,
        sol_type: boundary.SolutionType = boundary.SolutionType.UNKNOWN,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun: bag.CS2Fun = bag.cs2_bag,
        df_dtau_ptr: speedup.DifferentialPointer = fluid.DF_DTAU_BAG_PTR) -> float:
    r"""
    Calculates the transition strength parameter at the nucleation temperature,
    $\alpha_n$, from $\alpha_+$, for given $v_\text{wall}$.

    $$\alpha_n = \frac{4 \Delta \theta (T_n)}{3 w(T_n)} = \frac{4}{3} \frac{ \theta_s(T_n) - \theta_b(T_n) }{w(T_n)}$$

    :param v_wall: $v_\text{wall}$, wall speed
    :param alpha_p: $\alpha_+$, the at-wall strength parameter.
    :param sol_type: type of the bubble (detonation, deflagration etc.)
    :param n_xi: number of $\xi$ values to investigate
    :param cs2_fun: $c_s^2$ function
    :param df_dtau_ptr: pointer to the differential equations
    :return: $\alpha_n$, global strength parameter
    """
    check.check_wall_speed(v_wall)
    if sol_type == boundary.SolutionType.UNKNOWN.value:
        sol_type = transition.identify_solution_type_alpha_plus(v_wall, alpha_p).value
    _, w, xi = fluid.fluid_shell_alpha_plus(v_wall, alpha_p, sol_type, n_xi, cs2_fun=cs2_fun, df_dtau_ptr=df_dtau_ptr)
    n_wall = props.find_v_index(xi, v_wall)
    return alpha_p * w[n_wall] / w[-1]


def find_alpha_n_from_w_xi(w: np.ndarray, xi: np.ndarray, v_wall: float, alpha_p: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Calculates the transition strength parameter
    $\alpha_n = \frac{4}{3} \frac{\theta_s(T_n) - \theta_b(T_n)}{w(T_n)}$
    from $\alpha_+$.

    :return: $\alpha_n$
    """
    n_wall = props.find_v_index(xi, v_wall)
    return alpha_p * w[n_wall] / w[-1]


@numba.njit
def _find_alpha_plus_optimizer(
        alpha: np.ndarray,
        v_wall: float,
        sol_type: boundary.SolutionType,
        n_xi: int,
        alpha_n_given: float,
        cs2_fun: bag.CS2Fun,
        df_dtau_ptr: speedup.DifferentialPointer) -> float:
    """find_alpha_plus() is looking for the zeroes of this function: $\alpha_n = \alpha_{n,\text{given}}$."""
    return find_alpha_n(v_wall, alpha.item(), sol_type, n_xi, cs2_fun=cs2_fun, df_dtau_ptr=df_dtau_ptr) - alpha_n_given


def _find_alpha_plus_scalar_cs2_converter(cs2_fun_ptr: bag.CS2FunScalarPtr) -> CS2CFunc:
    r"""Converter for getting a $c_s^2$ ctypes function from a pointer

    This is a rather ugly hack. There should be a better way to call a function by a pointer!
    """
    with find_alpha_plus_scalar_lock:
        if cs2_fun_ptr in CS2CACHE:
            return CS2CACHE[cs2_fun_ptr]
        # https://numba.pydata.org/numba-doc/0.15.1/interface_c.html
        cs2_fun = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(cs2_fun_ptr)
        CS2CACHE[cs2_fun_ptr] = cs2_fun
        return cs2_fun


@numba.njit
def _find_alpha_plus_scalar(
        v_wall: float,
        alpha_n_given: float,
        n_xi: int,
        cs2_fun_ptr: bag.CS2FunScalarPtr,
        df_dtau_ptr: speedup.DifferentialPointer,
        xtol: float) -> float:
    """
    TODO: this might not generalize directly to models other than the bag model.
    It's possibly that the equations don't require any modifications, but instead the optimizer will simply
    fail in some cases.
    At least the sol_type dependence in fluid_shell_alpha_plus should be removed.
    """
    if alpha_n_given < alpha_n_max_detonation(v_wall):
        # Must be detonation
        # sol_type = boundary.SolutionType.DETON
        return alpha_n_given
    if alpha_n_given >= alpha_n_max_deflagration(v_wall):
        # Greater than the maximum possible -> fail
        return np.nan
    sol_type = boundary.SolutionType.SUB_DEF if v_wall <= const.CS0 else boundary.SolutionType.HYBRID
    ap_initial_guess = alpha_plus_initial_guess(v_wall, alpha_n_given)
    with numba.objmode(ret="float64"):
        cs2_fun = _find_alpha_plus_scalar_cs2_converter(cs2_fun_ptr)

        # This returns np.float64
        ret: float = scipy.optimize.fsolve(
            _find_alpha_plus_optimizer,
            ap_initial_guess,
            args=(v_wall, sol_type, n_xi, alpha_n_given, cs2_fun, df_dtau_ptr),
            xtol=xtol,
            factor=0.1)[0]
    return ret


@numba.njit(parallel=True)
def _find_alpha_plus_arr(
        v_wall: np.ndarray,
        alpha_n_given: float,
        n_xi: int,
        cs2_fun_ptr: bag.CS2FunScalarPtr,
        df_dtau_ptr: speedup.DifferentialPointer,
        xtol: float) -> np.ndarray:
    ap = np.zeros_like(v_wall)
    for i in numba.prange(v_wall.size):
        ap[i] = _find_alpha_plus_scalar(v_wall[i], alpha_n_given, n_xi, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr)
    return ap


@numba.generated_jit(nopython=True)
def find_alpha_plus(
        v_wall: th.FloatOrArr,
        alpha_n_given: float,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun_ptr: bag.CS2FunScalarPtr = bag.CS2_BAG_SCALAR_PTR,
        df_dtau_ptr: speedup.DifferentialPointer = fluid.DF_DTAU_BAG_PTR,
        xtol: float = const.FIND_ALPHA_PLUS_TOL) -> th.FloatOrArrNumba:
    r"""
    Calculate the at-wall strength parameter $\alpha_+$ from given $\alpha_n$ and $v_\text{wall}$.

    $$\alpha_+ = \frac{4 \Delta \theta (T_+)}{3 w_+} = \frac{4}{3} \frac{ \theta_s(T_+) - \theta_b(T_+) }{w(T_+)}$$
    (:gw_pt_ssm:`\ `, eq. 2.11)

    Uses :func:`scipy.optimize.fsolve` and therefore spends time in the Python interpreter even when jitted.
    This should be taken into account when running parallel simulations.

    :param v_wall: $v_\text{wall}$, the wall speed
    :param alpha_n_given: $\alpha_n$, the global strength parameter
    :param n_xi: number of $\xi$ points
    :return: $\alpha_+$, the the at-wall strength parameter
    """
    if isinstance(v_wall, numba.types.Float):
        return _find_alpha_plus_scalar
    if isinstance(v_wall, numba.types.Array):
        if not v_wall.ndim:
            return _find_alpha_plus_scalar
        return _find_alpha_plus_arr
    if isinstance(v_wall, float):
        return _find_alpha_plus_scalar(v_wall, alpha_n_given, n_xi, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr, xtol=xtol)
    if isinstance(v_wall, np.ndarray):
        if not v_wall.ndim:
            return _find_alpha_plus_scalar(v_wall.item(), alpha_n_given, n_xi, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr, xtol=xtol)
        return _find_alpha_plus_arr(v_wall, alpha_n_given, n_xi, cs2_fun=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr, xtol=xtol)
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")
