"""Functions for calculating approximate solutions

TODO: remove duplicate check_wall_speed checks
"""

import numpy as np

import pttools.type_hints as th
from . import check
from . import const


def xi_zero(v_wall: th.FLOAT_OR_ARR, v_xi_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Used in approximate solution near v(xi) = xi: defined as solution to v(xi0) = xi0
    """
    check.check_wall_speed(v_wall)
    xi0 = (v_xi_wall + 2 * v_wall) / 3.
    return xi0


def v_approx_high_alpha(xi: th.FLOAT_OR_ARR, v_wall: th.FLOAT_OR_ARR, v_xi_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Approximate solution for fluid velocity v(xi) near v(xi) = xi.
    """
    check.check_wall_speed(v_wall)
    xi0 = xi_zero(v_wall, v_xi_wall)
    A2 = 3 * (2 * xi0 - 1) / (1 - xi0 ** 2)
    dv = (xi - xi0)
    return xi0 - 2 * dv - A2 * dv ** 2


def v_approx_hybrid(xi: th.FLOAT_OR_ARR, v_wall: th.FLOAT_OR_ARR, v_xi_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Approximate solution for fluid velocity v(xi) near v(xi) = xi (same as v_approx_high_alpha).
    """
    check.check_wall_speed(v_wall)
    xi0 = xi_zero(v_wall, v_xi_wall)
    A2 = 3 * (2 * xi0 - 1) / (1 - xi0 ** 2)
    dv = (xi - xi0)
    return xi - 2 * dv - A2 * dv ** 2


def w_approx_high_alpha(
        xi: th.FLOAT_OR_ARR,
        v_wall: th.FLOAT_OR_ARR,
        v_xi_wall: th.FLOAT_OR_ARR,
        w_xi_wall: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Approximate solution for ehthalpy w(xi) near v(xi) = xi.
    """
    check.check_wall_speed(v_wall)
    xi0 = xi_zero(v_wall, v_xi_wall)
    return w_xi_wall * np.exp(-12 * (xi - xi0) ** 2 / (1 - xi0 ** 2) ** 2)


def v_approx_low_alpha(xi: np.ndarray, v_wall: float, alpha: float) -> np.ndarray:
    """
    Approximate solution for fluid velocity v(xi) at low alpha_plus = alpha_n.
    """
    def v_approx_fun(x, v_w):
        # Shape of linearised solution for v(xi)
        return (v_w / x) ** 2 * (const.CS0 ** 2 - x ** 2) / (const.CS0 ** 2 - v_w ** 2)

    v_app = np.zeros_like(xi)
    v_max = 3 * alpha * v_wall / abs(3 * v_wall ** 2 - 1)
    shell = np.where(np.logical_and(xi > min(v_wall, const.CS0), xi < max(v_wall, const.CS0)))
    v_app[shell] = v_max * v_approx_fun(xi[shell], v_wall)

    return v_app


def w_approx_low_alpha(xi: np.ndarray, v_wall: float, alpha: float) -> np.ndarray:
    """
    Approximate solution for enthalpy w(xi) at low alpha_plus = alpha_n.
    (Not complete for xi < min(v_wall, cs0)).
    """

    v_max = 3 * alpha * v_wall / abs(3 * v_wall ** 2 - 1)
    gaws2 = 1. / (1. - 3 * v_wall ** 2)
    w_app = np.exp(8 * v_max * gaws2 * v_wall ** 2 * (1 / xi - 1 / const.CS0))

    w_app[np.where(xi > max(v_wall, const.CS0))] = 1.0
    w_app[np.where(xi < min(v_wall, const.CS0))] = np.nan

    return w_app
