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


@numba.vectorize(nopython=True)
def v_shock(xi: th.FLOAT_OR_ARR):
    r"""
    Fluid velocity at a shock at xi.
    No shocks exist for $\xi < cs$, so returns zero.
    """
    # TODO: Maybe should return a nan?
    if xi < const.CS0:
        return 0
    return (3 * xi ** 2 - 1) / (2 * xi)


def w_shock(xi: th.FLOAT_OR_ARR, w_n: float = 1.) -> th.FLOAT_OR_ARR:
    r"""
    Fluid enthalpy at a shock at $\xi$.
    No shocks exist for $\xi < cs$, so returns nan.
    """
    w_sh = w_n * (9*xi**2 - 1)/(3*(1-xi**2))

    if isinstance(w_sh, np.ndarray):
        w_sh[np.where(xi < const.CS0)] = np.nan
    else:
        if xi < const.CS0:
            w_sh = np.nan

    return w_sh


def find_shock_index(v_f: np.ndarray, xi: np.ndarray, v_wall: float, sol_type: boundary.SolutionType) -> int:
    r"""
    Array index of shock from first point where fluid velocity $v_f$ goes below $v_\text{shock}$.
    For detonation, returns wall position.
    """
    check.check_wall_speed(v_wall)
    n_shock = 0

    if not (sol_type == boundary.SolutionType.DETON):
        it = np.nditer([v_f, xi], flags=['c_index'])
        for v, x in it:
            if x > v_wall:
                if v <= v_shock(x):
                    n_shock = it.index
                    break
    else:
        n_shock = find_v_index(xi, v_wall)

    return n_shock


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
