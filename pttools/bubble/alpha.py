"""Functions for alpha_n (strength parameter at nucleation temp) and
alpha_p(lus) (strength parameter just in front of wall)
"""

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
    """
    Calculates alpha_n from alpha_plus, for given v_wall.
    v_wall can be scalar or iterable.
    alpha_p[lus] must be scalar.
    alpha = ([(3/4) difference in trace anomaly]/enthalpy).
    alpha_n is global strength parameter, alpha_plus the at-wall strength parameter.
    """
    check.check_wall_speed(v_wall)
    if sol_type == boundary.SolutionType.UNKNOWN:
        sol_type = transition.identify_solution_type_alpha_plus(v_wall, alpha_p)
    _, w, xi = fluid.fluid_shell_alpha_plus(v_wall, alpha_p, sol_type, n_xi)
    n_wall = props.find_v_index(xi, v_wall)
    return alpha_p * w[n_wall] / w[-1]


def find_alpha_plus(v_wall: th.FLOAT_OR_ARR, alpha_n_given: float, n_xi: int = const.N_XI_DEFAULT) -> th.FLOAT_OR_ARR:
    """
    Calculate alpha_plus from a given alpha_n and v_wall.
    v_wall can be scalar or iterable.
    alpha_n_given must be scalar.
    alpha = ([(3/4) difference in trace anomaly]/enthalpy)
    alpha_n is global strength parameter, alpha_plus the at-wall strength parameter.
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
    """
    Initial guess for root-finding of alpha_plus from alpha_n.
    Linear approx between alpha_n_min and alpha_n_max.
    """
    #     Doesn't do obvious checks like Detonation - needs improving?

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
    """
    Calculates alpha_N ([(3/4) difference in trace anomaly]/enthalpy) from alpha_p[lus]
    Assumes one has solution arrays w, xi
    """
    n_wall = props.find_v_index(xi, v_wall)
    return alpha_p * w[n_wall] / w[-1]


def alpha_n_max_hybrid(v_wall: float, n_xi: int = const.N_XI_DEFAULT) -> float:
    """
    Calculates maximum alpha_n for given v_wall, assuming Hybrid fluid shell
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


def alpha_n_max(v_wall: th.FLOAT_OR_ARR, Np=const.N_XI_DEFAULT) -> th.FLOAT_OR_ARR:
    """
    alpha_n_max(v_wall, Np=N_XI_DEFAULT)

    Calculates maximum alpha_n (relative trace anomaly outside bubble)
    for given v_wall, which is max alpha_n for (supersonic) deflagration.
    """
    return alpha_n_max_deflagration(v_wall, Np)


def alpha_n_max_deflagration(v_wall: th.FLOAT_OR_ARR, Np=const.N_XI_DEFAULT) -> th.FLOAT_OR_ARR:
    """
    Calculates maximum alpha_n (relative trace anomaly outside bubble)
    for given v_wall, for deflagration.
    Works also for hybrids, as they are supersonic deflagrations
    """
    check.check_wall_speed(v_wall)
    # sol_type_ = identify_solution_type(v_wall_, 1./3)
    it = np.nditer([None, v_wall], [], [['writeonly', 'allocate'], ['readonly']])
    for ww, vw in it:
        if vw > const.CS0:
            sol_type = boundary.SolutionType.HYBRID
        else:
            sol_type = boundary.SolutionType.SUB_DEF

        ap = 1. / 3 - 1.0e-10  # Warning - this is not safe.  Causes warnings for v low vw
        _, w, xi = fluid.fluid_shell_alpha_plus(vw, ap, sol_type, Np)
        n_wall = props.find_v_index(xi, vw)
        ww[...] = w[n_wall + 1] * (1. / 3)

    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalised to 1 at large xi
    # Need n_wall+1, as w is an integral of v, and lags by 1 step
    if isinstance(v_wall, np.ndarray):
        return it.operands[0]
    return type(v_wall)(it.operands[0])


def alpha_plus_max_detonation(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Maximum allowed value of alpha_plus for a detonation with wall speed v_wall.
    Comes from inverting v_w > v_Jouguet
    """

    check.check_wall_speed(v_wall)
    it = np.nditer([None, v_wall], [], [['writeonly', 'allocate'], ['readonly']])
    for bb, vw in it:
        a = 3 * (1 - vw ** 2)
        if vw < const.CS0:
            b = 0.0
        else:
            b = (1 - np.sqrt(3) * vw) ** 2
        bb[...] = b / a

    if isinstance(v_wall, np.ndarray):
        return it.operands[0]
    return type(v_wall)(it.operands[0])


def alpha_n_max_detonation(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Maximum allowed value of alpha_n for a detonation with wall speed v_wall.
    Same as alpha_plus_max_detonation, because alpha_n = alpha_plus for detonation.
    """
    return alpha_plus_max_detonation(v_wall)


def alpha_plus_min_hybrid(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Minimum allowed value of alpha_plus for a hybrid with wall speed v_wall.
    Condition from coincidence of wall and shock
    """
    check.check_wall_speed(v_wall)
    b = (1 - np.sqrt(3) * v_wall) ** 2
    c = 9 * v_wall ** 2 - 1
    # for bb, vw in np.nditer([b,v_wall_]):
    if isinstance(b, np.ndarray):
        b[np.where(v_wall < 1. / np.sqrt(3))] = 0.0
    else:
        if v_wall < const.CS0:
            b = 0.0
    return b / c


def alpha_n_min_hybrid(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Minimum alpha_n for a hybrid. Equal to maximum alpha_n for a detonation.
    Same as aalpha_n_min_deflagration, as a hybrid is a supersonic deflagration.
    """
    check.check_wall_speed(v_wall)
    return alpha_n_max_detonation(v_wall)


def alpha_n_min_deflagration(v_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Minimum alpha_n for a deflagration. Equal to maximum alpha_n for a detonation.
    Same as alpha_n_min_hybrid, as a hybrid is a supersonic deflagration.
    """
    check.check_wall_speed(v_wall)
    return alpha_n_max_detonation(v_wall)
