"""Functions for alpha_n (strength parameter at nucleation temp) and
alpha_p(lus) (strength parameter just in front of wall)
"""

import numba
import numpy as np
import scipy.optimize as opt

import pttools.type_hints as th
from . import boundary
from . import const
from . import fluid
from . import check
from . import props
from . import transition


def find_alpha_n(
        v_wall: th.FLOAT_OR_ARR,
        alpha_p: float,
        sol_type: boundary.SolutionType = boundary.SolutionType.UNKNOWN,
        n_xi: int = const.N_XI_DEFAULT) -> float:
    r"""
    Calculates $\alpha_n$ from $\alpha_+$, for given v_wall.

    $\alpha = \frac{ \frac{3}{4} \text{difference in trace anomaly} }{\text{enthalpy}}$

    :param v_wall: $v_\text{wall}$, wall speed
    :param alpha_p: $\alpha_+$, the at-wall strength parameter.
    :param sol_type:
    :param n_xi:
    :return: $\alpha_n$, global strength parameter
    """
    check.check_wall_speed(v_wall)
    if sol_type == boundary.SolutionType.UNKNOWN:
        sol_type = transition.identify_solution_type_alpha_plus(v_wall, alpha_p)
    _, w, xi = fluid.fluid_shell_alpha_plus(v_wall, alpha_p, sol_type, n_xi)
    n_wall = props.find_v_index(xi, v_wall)
    return alpha_p * w[n_wall] / w[-1]


def find_alpha_plus(v_wall: th.FLOAT_OR_ARR, alpha_n_given: float, n_xi: int = const.N_XI_DEFAULT) -> th.FLOAT_OR_ARR:
    r"""
    Calculate $\alpha_+$ from a given $\alpha_n$ and $v_\text{wall}$.

    $\alpha = \frac{ \frac{3}{4} \text{difference in trace anomaly} }{\text{enthalpy}}$

    :param v_wall: $v_\text{wall}$, the wall speed
    :param alpha_n_given: $\alpha_n$, the global strength parameter
    :param n_xi:
    :return: $\alpha_+$, the the at-wall strength parameter
    """
    it = np.nditer([None, v_wall], [], [['writeonly', 'allocate'], ['readonly']])
    for ap, vw in it:
        if alpha_n_given < alpha_n_max_detonation(vw):
            # Must be detonation
            sol_type = boundary.SolutionType.DETON
            ap[...] = alpha_n_given
        else:
            if alpha_n_given < alpha_n_max_deflagration(vw):
                if vw <= const.CS0:
                    sol_type = boundary.SolutionType.SUB_DEF
                else:
                    sol_type = boundary.SolutionType.HYBRID

                def func(x):
                    return find_alpha_n(vw, x, sol_type, n_xi) - alpha_n_given

                a_initial_guess = alpha_plus_initial_guess(vw, alpha_n_given)
                al_p = opt.fsolve(func, a_initial_guess, xtol=const.FIND_ALPHA_PLUS_TOL, factor=0.1)[0]
                ap[...] = al_p
            else:
                ap[...] = np.nan

    if isinstance(v_wall, np.ndarray):
        return it.operands[0]
    return type(v_wall)(it.operands[0])


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
        a_guess = alpha_n_given
    else:
        alpha_plus_min = alpha_plus_min_hybrid(v_wall)
        alpha_plus_max = 1. / 3

        alpha_n_min = alpha_n_min_hybrid(v_wall)
        alpha_n_max = alpha_n_max_deflagration(v_wall)

        slope = (alpha_plus_max - alpha_plus_min) / (alpha_n_max - alpha_n_min)

        a_guess = alpha_plus_min + slope * (alpha_n_given - alpha_n_min)

    return a_guess


def find_alpha_n_from_w_xi(w: np.ndarray, xi: np.ndarray, v_wall: float, alpha_p: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Calculates $\alpha_n$,
    $\alpha = \frac{ \frac{3}{4} \text{difference in trace anomaly} }{\text{enthalpy}}$
    from $\alpha_+$

    :return: $\alpha_n$
    """
    n_wall = props.find_v_index(xi, v_wall)
    return alpha_p * w[n_wall] / w[-1]


# TODO: this cannot be jitted yet due to the use of fluid_shell_alpha_plus()
def alpha_n_max_hybrid(v_wall: float, n_xi: int = const.N_XI_DEFAULT) -> float:
    r"""
    Calculates $\alpha_{n,\max}$ for given $v_\text{wall\$, assuming hybrid fluid shell

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


# @numba.njit
def alpha_n_max(v_wall: th.FLOAT_OR_ARR, Np=const.N_XI_DEFAULT) -> th.FLOAT_OR_ARR:
    r"""
    Calculates $\alpha_{n,\max}$ (relative trace anomaly outside bubble)
    for given $v_\text{wall}$, which is max $\alpha_n$ for (supersonic) deflagration.

    :return: $\alpha_{n,\max}$, the relative trace anomaly outside the bubble
    """
    return alpha_n_max_deflagration(v_wall, Np)


# @numba.njit
def _alpha_n_max_deflagration_scalar(v_wall: float, Np: int) -> float:
    check.check_wall_speed(v_wall)
    if v_wall > const.CS0:
        sol_type = boundary.SolutionType.HYBRID
    else:
        sol_type = boundary.SolutionType.SUB_DEF

    ap = 1. / 3 - 1.0e-10  # Warning - this is not safe.  Causes warnings for v low vw
    _, w, xi = fluid.fluid_shell_alpha_plus(v_wall, ap, sol_type, Np)
    n_wall = props.find_v_index(xi, v_wall)
    return w[n_wall + 1] * (1. / 3)


# @numba.njit
def _alpha_n_max_deflagration_arr(v_wall: np.ndarray, Np: int) -> np.ndarray:
    ret = np.zeros_like(v_wall)
    for i in range(v_wall.size):
        ret[i] = _alpha_n_max_deflagration_scalar(v_wall[i], Np)
    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalised to 1 at large xi
    # Need n_wall+1, as w is an integral of v, and lags by 1 step
    return ret


# TODO: this cannot be jitted yet due to the use of fluid_shell_alpha_plus()
# @numba.generated_jit(nopython=True)
def alpha_n_max_deflagration(v_wall: th.FLOAT_OR_ARR, Np: int = const.N_XI_DEFAULT) -> th.FLOAT_OR_ARR_NUMBA:
    r"""
    Calculates maximum $\alpha_n$ (relative trace anomaly outside bubble)
    for given $v_\text{wall}$, for deflagration.
    Works also for hybrids, as they are supersonic deflagrations

    :return: maximum $\alpha_n$
    """
    if isinstance(v_wall, numba.types.Float):
        return _alpha_n_max_deflagration_scalar
    if isinstance(v_wall, numba.types.Array):
        return _alpha_n_max_deflagration_arr
    if isinstance(v_wall, float):
        return _alpha_n_max_deflagration_scalar(v_wall, Np)
    if isinstance(v_wall, np.ndarray):
        if not v_wall.ndim:
            return _alpha_n_max_deflagration_scalar(v_wall.item(), Np)
        return _alpha_n_max_deflagration_arr(v_wall, Np)
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


@numba.njit
def _alpha_plus_max_detonation_scalar(v_wall: float):
    check.check_wall_speed(v_wall)
    a = 3 * (1 - v_wall ** 2)
    if v_wall < const.CS0:
        b = 0.0
    else:
        b = (1 - np.sqrt(3) * v_wall) ** 2
    return b / a


@numba.njit
def _alpha_plus_max_detonation_arr(v_wall: np.ndarray):
    bb = np.zeros(v_wall.shape)
    for i in range(v_wall.size):
        bb[i] = _alpha_plus_max_detonation_scalar(v_wall[i])
    return bb


@numba.generated_jit(nopython=True)
def alpha_plus_max_detonation(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR_NUMBA:
    r"""
    Maximum allowed value of $\alpha_+$ for a detonation with wall speed $v_\text{wall}$.
    Comes from inverting $v_w$ > $v_\text{Jouguet}$.
    """
    if isinstance(v_wall, numba.types.Float):
        return _alpha_plus_max_detonation_scalar
    if isinstance(v_wall, numba.types.Array):
        if not v_wall.ndim:
            return _alpha_plus_max_detonation_scalar
        return _alpha_plus_max_detonation_arr
    if isinstance(v_wall, float):
        return _alpha_plus_max_detonation_scalar(v_wall)
    if isinstance(v_wall, np.ndarray):
        if not v_wall.ndim:
            return _alpha_plus_max_detonation_scalar(v_wall.item())
        return _alpha_plus_max_detonation_arr(v_wall)
    else:
        raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


@numba.njit
def alpha_n_max_detonation(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Maximum allowed value of $\alpha_n$ for a detonation with wall speed $v_\text{wall}$.
    Same as alpha_plus_max_detonation, because $\alpha_n = \alpha_+$ for detonation.
    """
    return alpha_plus_max_detonation(v_wall)


@numba.njit
def _alpha_plus_min_hybrid_scalar(v_wall: float) -> float:
    check.check_wall_speed(v_wall)
    if v_wall < const.CS0:
        return 0
    b = (1 - np.sqrt(3) * v_wall) ** 2
    c = 9 * v_wall ** 2 - 1
    return b / c


@numba.njit
def _alpha_plus_min_hybrid_arr(v_wall: np.ndarray) -> np.ndarray:
    # TODO: implement this in a way that is performant also without Numba
    ret = np.zeros_like(v_wall)
    for i in range(v_wall.size):
        ret[i] = _alpha_plus_min_hybrid_scalar(v_wall[i])
    return ret


@numba.generated_jit(nopython=True)
def alpha_plus_min_hybrid(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR_NUMBA:
    r"""
    Minimum allowed value of $\alpha_+$ for a hybrid with wall speed $v_\text{wall}$.
    Condition from coincidence of wall and shock
    """
    if isinstance(v_wall, numba.types.Float):
        return _alpha_plus_min_hybrid_scalar
    if isinstance(v_wall, numba.types.Array):
        if not v_wall.ndim:
            return _alpha_plus_min_hybrid_scalar
        return _alpha_plus_min_hybrid_arr
    if isinstance(v_wall, float):
        return _alpha_plus_min_hybrid_scalar(v_wall)
    if isinstance(v_wall, np.ndarray):
        if not v_wall.ndim:
            return _alpha_plus_min_hybrid_scalar(v_wall.item())
        return _alpha_plus_min_hybrid_arr(v_wall)
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


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
