"""Useful functions for finding properties of solution """

import typing as tp

import numba.types
import numpy as np

import pttools.type_hints as th
from . import boundary
from . import check
from . import const


@numba.njit
def find_v_index(xi: np.ndarray, v_target: float) -> int:
    r"""
    The first array index of $\xi$ where value is just above $v_\text{target}$
    """
    greater = np.where(xi >= v_target)[0]
    if greater.size:
        return greater[0]
    return 0


@numba.vectorize
def v_shock(xi: th.FLOAT_OR_ARR):
    r"""
    Fluid velocity at a shock at xi.
    No shocks exist for $\xi < cs$, so returns zero.
    """
    # TODO: Maybe should return a nan?
    if xi < const.CS0:
        return 0
    return (3 * xi ** 2 - 1) / (2 * xi)


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
def w_shock(xi: th.FLOAT_OR_ARR, w_n: float = 1.) -> th.FLOAT_OR_ARR_NUMBA:
    r"""
    Fluid enthalpy at a shock at $\xi$.
    No shocks exist for $\xi < cs$, so returns nan.
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


@numba.njit
def find_shock_index(v_f: np.ndarray, xi: np.ndarray, v_wall: float, sol_type: boundary.SolutionType) -> int:
    r"""
    Array index of shock from first point where fluid velocity $v_f$ goes below $v_\text{shock}$.
    For detonation, returns wall position.
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
def shock_zoom_last_element(
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Replaces last element of $(v,w,\xi)$ arrays by better estimate of
    shock position and values of $v, w$ there.
    """
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
