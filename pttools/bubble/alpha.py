r"""
Functions for $\alpha_n$ (strength parameter at nucleation temperature) and
$alpha_\text{plus}$ (strength parameter just in front of the wall)
"""

import numba
import numpy as np
import scipy.optimize as opt

import pttools.type_hints as th
from pttools import speedup
from . import boundary
from . import const
from . import fluid
from . import check
from . import props
from . import transition


@numba.njit
def find_alpha_n(
        v_wall: th.FLOAT_OR_ARR,
        alpha_p: float,
        sol_type: boundary.SolutionType = boundary.SolutionType.UNKNOWN,
        n_xi: int = const.N_XI_DEFAULT) -> float:
    r"""
    Calculates $\alpha_n$ from $\alpha_+$, for given $v_\text{wall}$.

    $\alpha = \frac{ \frac{3}{4} \text{difference in trace anomaly} }{\text{enthalpy}}$

    :param v_wall: $v_\text{wall}$, wall speed
    :param alpha_p: $\alpha_+$, the at-wall strength parameter.
    :param sol_type: type of the bubble (detonation, deflagration etc.)
    :param n_xi: number of $\xi$ values to investigate
    :return: $\alpha_n$, global strength parameter
    """
    check.check_wall_speed(v_wall)
    if sol_type == boundary.SolutionType.UNKNOWN.value:
        sol_type = transition.identify_solution_type_alpha_plus(v_wall, alpha_p).value
    _, w, xi = fluid.fluid_shell_alpha_plus(v_wall, alpha_p, sol_type, n_xi)
    n_wall = props.find_v_index(xi, v_wall)
    return alpha_p * w[n_wall] / w[-1]


@numba.njit
def _find_alpha_plus_optimizer(
        x: np.ndarray,
        v_wall: float,
        sol_type: boundary.SolutionType,
        n_xi: int,
        alpha_n_given: float) -> float:
    """find_alpha_plus() is looking for the zeroes of this function."""
    return find_alpha_n(v_wall, x.item(), sol_type, n_xi) - alpha_n_given


@numba.njit
def _find_alpha_plus_scalar(v_wall: float, alpha_n_given: float, n_xi: int) -> float:
    if alpha_n_given < alpha_n_max_detonation(v_wall):
        # Must be detonation
        # sol_type = boundary.SolutionType.DETON
        return alpha_n_given
    if alpha_n_given < alpha_n_max_deflagration(v_wall):
        sol_type = boundary.SolutionType.SUB_DEF if v_wall <= const.CS0 else boundary.SolutionType.HYBRID

        a_initial_guess = alpha_plus_initial_guess(v_wall, alpha_n_given)
        with numba.objmode(ret="float64"):
            # This returns np.float64
            ret: float = opt.fsolve(
                _find_alpha_plus_optimizer,
                a_initial_guess,
                args=(v_wall, sol_type, n_xi, alpha_n_given),
                xtol=const.FIND_ALPHA_PLUS_TOL,
                factor=0.1)[0]
        return ret
    return np.nan


@numba.njit(parallel=True)
def _find_alpha_plus_arr(v_wall: np.ndarray, alpha_n_given: float, n_xi: int) -> np.ndarray:
    ap = np.zeros_like(v_wall)
    for i in numba.prange(v_wall.size):
        ap[i] = _find_alpha_plus_scalar(v_wall[i], alpha_n_given, n_xi)
    return ap


@numba.generated_jit(nopython=True)
def find_alpha_plus(
        v_wall: th.FLOAT_OR_ARR,
        alpha_n_given: float,
        n_xi: int = const.N_XI_DEFAULT) -> th.FLOAT_OR_ARR_NUMBA:
    r"""
    Calculate $\alpha_+$ from a given $\alpha_n$ and $v_\text{wall}$.

    $\alpha = \frac{ \frac{3}{4} \text{difference in trace anomaly} }{\text{enthalpy}}$

    :param v_wall: $v_\text{wall}$, the wall speed
    :param alpha_n_given: $\alpha_n$, the global strength parameter
    :param n_xi:
    :return: $\alpha_+$, the the at-wall strength parameter
    """
    if isinstance(v_wall, numba.types.Float):
        return _find_alpha_plus_scalar
    if isinstance(v_wall, numba.types.Array):
        if not v_wall.ndim:
            return _find_alpha_plus_scalar
        return _find_alpha_plus_arr
    if isinstance(v_wall, float):
        return _find_alpha_plus_scalar(v_wall, alpha_n_given, n_xi)
    if isinstance(v_wall, np.ndarray):
        if not v_wall.ndim:
            return _find_alpha_plus_scalar(v_wall.item(), alpha_n_given, n_xi)
        return _find_alpha_plus_arr(v_wall, alpha_n_given, n_xi)
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


@numba.njit
def alpha_plus_initial_guess(v_wall: th.FLOAT_OR_ARR, alpha_n_given: float) -> th.FLOAT_OR_ARR:
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


def find_alpha_n_from_w_xi(w: np.ndarray, xi: np.ndarray, v_wall: float, alpha_p: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Calculates the transition strength parameter
    $\alpha_n = \frac{4}{3} \frac{\theta_s(T_n) - \theta_b(T_n)}{w(T_n)}$,
    from $\alpha_+$.

    :return: $\alpha_n$
    """
    n_wall = props.find_v_index(xi, v_wall)
    return alpha_p * w[n_wall] / w[-1]


# TODO: this cannot be jitted yet due to the use of fluid_shell_alpha_plus()
def alpha_n_max_hybrid(v_wall: float, n_xi: int = const.N_XI_DEFAULT) -> float:
    r"""
    Calculates the relative trace anomaly outside the bubble, $\alpha_{n,\max}$,
     for given $v_\text{wall}$, assuming hybrid fluid shell

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
def alpha_n_max(v_wall: th.FLOAT_OR_ARR, n_xi: int = const.N_XI_DEFAULT) -> th.FLOAT_OR_ARR:
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
def alpha_n_max_deflagration(v_wall: th.FLOAT_OR_ARR, n_xi: int = const.N_XI_DEFAULT) -> th.FLOAT_OR_ARR_NUMBA:
    r"""
    Calculates the relative trace anomaly outside the bubble, $\alpha_{n,\max}$,
    for given $v_\text{wall}$, for deflagration.
    Works also for hybrids, as they are supersonic deflagrations.

    :param v_wall: $v_\text{wall}$
    :param n_xi: number of $\xi$ points
    :return: maximum $\alpha_n$
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


@speedup.vectorize(nopython=True)
def alpha_plus_max_detonation(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR_NUMBA:
    r"""
    Maximum allowed value of $\alpha_+$ for a detonation with wall speed $v_\text{wall}$.
    Comes from inverting $v_w$ > $v_\text{Jouguet}$.
    """
    check.check_wall_speed(v_wall)
    if v_wall < const.CS0:
        return 0
    a = 3 * (1 - v_wall ** 2)
    b = (1 - np.sqrt(3) * v_wall) ** 2
    return b / a


@numba.njit
def alpha_n_max_detonation(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Maximum allowed value of $\alpha_n$ for a detonation with wall speed $v_\text{wall}$.
    Same as alpha_plus_max_detonation, because $\alpha_n = \alpha_+$ for detonation.
    """
    return alpha_plus_max_detonation(v_wall)


@speedup.vectorize(nopython=True)
def alpha_plus_min_hybrid(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR_NUMBA:
    r"""
    Minimum allowed value of $\alpha_+$ for a hybrid with wall speed $v_\text{wall}$.
    Condition from coincidence of wall and shock.
    """
    check.check_wall_speed(v_wall)
    if v_wall < const.CS0:
        return 0
    b = (1 - np.sqrt(3) * v_wall) ** 2
    c = 9 * v_wall ** 2 - 1
    return b / c


@numba.njit
def alpha_n_min_hybrid(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Minimum $\alpha_n$ for a hybrid. Equal to maximum $\alpha_n$ for a detonation.
    Same as alpha_n_min_deflagration, as a hybrid is a supersonic deflagration.
    """
    # This check is implemented in the inner functions
    # check.check_wall_speed(v_wall)
    return alpha_n_max_detonation(v_wall)


@numba.njit
def alpha_n_min_deflagration(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Minimum $\alpha_n$ for a deflagration. Equal to maximum $\alpha_n$ for a detonation.
    Same as alpha_n_min_hybrid, as a hybrid is a supersonic deflagration.
    """
    # This check is implemented in the inner functions
    # check.check_wall_speed(v_wall)
    return alpha_n_max_detonation(v_wall)
