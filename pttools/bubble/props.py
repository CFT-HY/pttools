"""Useful functions for finding the properties of a solution."""

import typing as tp

import numba.types
import numpy as np

import pttools.type_hints as th
from . import boundary
from . import check
from . import const
from . import relativity


@numba.njit
def find_shock_index(v_f: np.ndarray, xi: np.ndarray, v_wall: float, sol_type: boundary.SolutionType) -> int:
    r"""
    Array index of shock from first point where fluid velocity $v_f$ goes below $v_\text{shock}$.
    For detonation, returns wall position.

    :param v_f: fluid velocity $v_f$
    :param xi: $\xi$
    :param v_wall: wall velocity $v_\text{wall}$
    :param sol_type: solution type (detonation etc.)
    :return: shock index
    """
    check.check_wall_speed(v_wall)

    n_shock = 0
    if sol_type == boundary.SolutionType.DETON:
        n_shock = find_v_index(xi, v_wall)
    else:
        for i, (v, x) in enumerate(zip(v_f, xi)):
            if x > v_wall and v <= v_shock(x):
                n_shock = i
                break

    return n_shock


@numba.njit
def find_v_index(xi: np.ndarray, v_target: float) -> int:
    r"""
    The first array index of $\xi$ where value is just above $v_\text{target}$
    """
    greater = np.where(xi >= v_target)[0]
    if greater.size:
        return greater[0]
    return 0


@numba.njit
def shock_zoom_last_element(
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Replaces last element of $(v,w,\xi)$ arrays by better estimate of
    shock position and values of $v, w$ there.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :return: given $v, w, \xi$ arrays with the last elements replaced
    """
    # TODO: Edit this so that it won't edit the original arrays.

    v_sh = v_shock(xi)
    # First check if last two elements straddle shock
    if v[-1] < v_sh[-1] and v[-2] > v_sh[-2] and xi[-1] > xi[-2]:
        dxi = xi[-1] - xi[-2]
        dv = v[-1] - v[-2]
        dv_sh = v_sh[-1] - v_sh[-2]
        dw_sh = w[-1] - w[-2]
        dxi_sh = dxi * (v[-2] - v_sh[-2])/(dv_sh - dv)
        # now replace final element
        xi[-1] = xi[-2] + dxi_sh
        v[-1] = v[-2] + (dv_sh/dxi)*dxi_sh
        w[-1] = w[-2] + (dw_sh/dxi)*dxi_sh
    # If not, do nothing
    return v, w, xi


@numba.vectorize
def v_max_behind(xi: th.FloatOrArr, cs: float):
    r"""Maximum fluid velocity behind the wall.
    Given by the condition $\mu(\xi, v) = c_s$.
    This results in:
    $$ v_\text{max} = \frac{c_s-\xi}{c_s \xi - 1} $$

    :param xi: $\xi$
    :param cs: $c_s$, speed of sound behind the wall (=in the broken phase)
    :return: $v_\text{max,behind}$
    """
    return relativity.lorentz(xi, cs)


@numba.njit
def _v_shock_scalar(xi: float, cs2: float):
    # TODO: Maybe should return a nan?
    # Squaring is used instead of a square root, as it's faster.
    if xi**2 < cs2:
        return 0

    # This generalization is wrong. The shock velocity requires significantly more complex treatment.
    # TODO: implement v_shock
    # return (xi**2 - cs2) / (xi * (1 - cs2))

    # For bag EoS
    return (3 * xi**2 - 1) / (2 * xi)


@numba.njit
def _v_shock_arr(xi: np.ndarray, cs2: float):
    ret = np.zeros_like(xi)
    for i in range(xi.size):
        ret[i] = _v_shock_scalar(xi[i], cs2)
    return ret


@numba.generated_jit(nopython=True)
def v_shock(xi: th.FloatOrArr, cs2: float = const.CS0_2):
    r"""
    Fluid velocity at a shock at xi.
    No shocks exist for $\xi < cs$, so returns zero.
    $$ v_{sh}(\xi) = \frac{\xi^2 - c_s^2}{\xi (1 - c_s^2)} $$

    For the bag equation of state ($c_s = \frac{1}{3}$) this gives eq. B.17 of :gw_pt_ssm:`\ `.

    $$ v_{sh}(\xi) = \frac{3 \xi^2 - 1}{2\xi} $$

    :param xi: $\xi$
    :param cs2: $c_s^2$
    :return: $v_{sh}$
    """
    if isinstance(xi, numba.types.Float):
        return _v_shock_scalar
    if isinstance(xi, numba.types.Array):
        return _v_shock_arr
    if isinstance(xi, float):
        return _v_shock_scalar(xi, cs2)
    if isinstance(xi, np.ndarray):
        return _v_shock_arr(xi, cs2)
    raise TypeError(f"Unknown type for xi: {type(xi)}")


@numba.njit
def _w_shock_scalar(xi: float, w_n: float) -> float:
    if xi < const.CS0:
        return np.nan
    return w_n * (9*xi**2 - 1)/(3*(1-xi**2))


@numba.njit
def _w_shock_arr(xi: np.ndarray, w_n: float) -> np.ndarray:
    ret = np.zeros_like(xi)
    for i in range(xi.size):
        ret[i] = _w_shock_scalar(xi[i], w_n)
    return ret


# This cannot be vectorized with numba.vectorize due to the keyword argument, but guvectorize might work
@numba.generated_jit(nopython=True)
def w_shock(xi: th.FloatOrArr, w_n: float = 1.) -> th.FloatOrArrNumba:
    r"""
    Fluid enthalpy at a shock at $\xi$.
    No shocks exist for $\xi < cs$, so returns nan.
    Equation B.18 of :gw_pt_ssm:`\ `.

    $$ w_{sh}(\xi) = w_n \frac{9\xi^2 - 1}{3(1 - \xi^2)} $$

    :param xi: $\xi$
    :param w_n: enthalpy on the TODO side of the wall
    :return: $w_\text{shock}$, fluid enthalpy at the shock
    """
    if isinstance(xi, numba.types.Float):
        return _w_shock_scalar
    if isinstance(xi, numba.types.Array):
        if not xi.ndim:
            return _w_shock_scalar
        return _w_shock_arr
    if isinstance(xi, float):
        return _w_shock_scalar(xi, w_n)
    if isinstance(xi, np.ndarray):
        if not xi.ndim:
            return _w_shock_scalar(xi.item(), w_n)
        return _w_shock_arr(xi, w_n)
    raise TypeError(f"Unknown type for xi: {type(xi)}")
