"""Useful functions for finding properties of solution """

import numpy as np

from . import check
from . import const


def find_v_index(xi, v_target):
    """
     The first array index of xi where value is just above v_target
    """
    n = 0
    it = np.nditer(xi, flags=['c_index'])
    for x in it:
        if x >= v_target:
            n = it.index
            break
    return n


def v_shock(xi):
    """
     Fluid velocity at a shock at xi.  No shocks exist for xi < cs, so returns zero.
    """
    # Maybe should return a nan?
    v_sh = (3*xi**2 - 1)/(2*xi)

    if isinstance(v_sh, np.ndarray):
        v_sh[np.where(xi < const.cs0)] = 0.0
    else:
        if xi < const.cs0:
            v_sh = 0.0

    return v_sh


def w_shock(xi, w_n=1.):
    """
     Fluid enthalpy at a shock at xi.  No shocks exist for xi < cs, so returns nan.
    """
    w_sh = w_n * (9*xi**2 - 1)/(3*(1-xi**2))

    if isinstance(w_sh, np.ndarray):
        w_sh[np.where(xi < const.cs0)] = np.nan
    else:
        if xi < const.cs0:
            w_sh = np.nan

    return w_sh


def find_shock_index(v_f, xi, v_wall, wall_type):
    """
     Array index of shock from first point where fluid velocity v_f goes below v_shock
     For detonation, returns wall position.
    """
    check.check_wall_speed(v_wall)
    n_shock = 0

    if not (wall_type == "Detonation"):
        it = np.nditer([v_f, xi], flags=['c_index'])
        for v, x in it:
            if x > v_wall:
                if v <= v_shock(x):
                    n_shock = it.index
                    break
    else:
        n_shock = find_v_index(xi, v_wall)

    return n_shock


def shock_zoom_last_element(v, w, xi):
    """
     Replaces last element of (v,w,xi) arrays by better estimate of
     shock position and values of v, w there.
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
        v[-1]  = v[-2] + (dv_sh/dxi)*dxi_sh
        w[-1]  = w[-2] + (dw_sh/dxi)*dxi_sh
    # If not, do nothing
    return v, w, xi
