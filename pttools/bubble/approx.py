"""Functions for calculating approximate solutions

TODO: remove duplicate check_wall_speed checks
"""

import numpy as np

import pttools.type_hints as th
from . import check
from . import const


def A2_approx(xi0: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Approximate solution for A2.
    $A2_\text{approx} = \frac{3(2\xi_0 - 1)}{1 - \xi_0^2}$

    :param xi0: $\xi_0$
    :return: A2
    """
    return 3 * (2 * xi0 - 1) / (1 - xi0 ** 2)


def v_approx_high_alpha(xi: th.FloatOrArr, v_wall: th.FloatOrArr, v_xi_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Approximate solution for fluid velocity $v(\xi)$ near $v(\xi) = \xi$.

    :return: $v_\text{approx}$
    """
    check.check_wall_speed(v_wall)
    xi0 = xi_zero(v_wall, v_xi_wall)
    dv = (xi - xi0)
    return xi0 - 2 * dv - A2_approx(xi0) * dv ** 2


def v_approx_hybrid(xi: th.FloatOrArr, v_wall: th.FloatOrArr, v_xi_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Approximate solution for fluid velocity $v(\xi)$ near $v(\xi) = \xi$ (same as v_approx_high_alpha).

    :return: $v_\text{approx}$
    """
    check.check_wall_speed(v_wall)
    xi0 = xi_zero(v_wall, v_xi_wall)
    dv = (xi - xi0)
    return xi - 2 * dv - A2_approx(xi0) * dv ** 2


def v_approx_low_alpha(xi: np.ndarray, v_wall: float, alpha: float) -> np.ndarray:
    r"""
    Approximate solution for fluid velocity $v(\xi)$ at low $\alpha_+ = \alpha_n$.

    :return: $v_\text{approx}$
    """
    def v_approx_fun(x, v_w):
        # Shape of linearised solution for v(xi)
        return (v_w / x) ** 2 * (const.CS0 ** 2 - x ** 2) / (const.CS0 ** 2 - v_w ** 2)

    v_app = np.zeros_like(xi)
    v_max = 3 * alpha * v_wall / abs(3 * v_wall ** 2 - 1)
    shell = np.where(np.logical_and(xi > min(v_wall, const.CS0), xi < max(v_wall, const.CS0)))
    v_app[shell] = v_max * v_approx_fun(xi[shell], v_wall)

    return v_app


def w_approx_high_alpha(
        xi: th.FloatOrArr,
        v_wall: th.FloatOrArr,
        v_xi_wall: th.FloatOrArr,
        w_xi_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Approximate solution for enthalpy $w(\xi)$ near $v(\xi) = \xi$.

    :return: $w_\text{approx}$
    """
    check.check_wall_speed(v_wall)
    xi0 = xi_zero(v_wall, v_xi_wall)
    return w_xi_wall * np.exp(-12 * (xi - xi0) ** 2 / (1 - xi0 ** 2) ** 2)


def w_approx_low_alpha(xi: np.ndarray, v_wall: float, alpha: float) -> np.ndarray:
    r"""
    Approximate solution for enthalpy $w(\xi)$ at low $\alpha_+ = \alpha_n$.
    (Not complete for $\xi < \min(v _\text{wall}, cs_0)$).

    :return: $w_\text{approx}$
    """
    v_max = 3 * alpha * v_wall / abs(3 * v_wall ** 2 - 1)
    gaws2 = 1. / (1. - 3 * v_wall ** 2)
    w_app = np.exp(8 * v_max * gaws2 * v_wall ** 2 * (1 / xi - 1 / const.CS0))

    w_app[np.where(xi > max(v_wall, const.CS0))] = 1.0
    w_app[np.where(xi < min(v_wall, const.CS0))] = np.nan

    return w_app


def xi_zero(v_wall: th.FloatOrArr, v_xi_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Used in approximate solution near $v(\xi) = \xi$: defined as solution to $v(\xi_0) = \xi_0$.

    $\xi_0 = \frac{1}{3} (v(\xi_\text{wall} + 2 v_\text{wall})$

    :param v_wall: $v_\text{wall}$
    :param v_xi_wall: $v(\xi_\text{wall})$
    :return: $\xi_0$
    """
    check.check_wall_speed(v_wall)
    xi0 = (v_xi_wall + 2 * v_wall) / 3.
    return xi0
