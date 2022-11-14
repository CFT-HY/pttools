"""Functions for shocks"""

import typing as tp

import numba.types
import numpy as np

import pttools.type_hints as th
from . import boundary
from . import check
from . import const
from . import props


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
        n_shock = props.find_v_index(xi, v_wall)
    else:
        for i, (v, x) in enumerate(zip(v_f, xi)):
            if x > v_wall and v <= v_shock_bag(x):
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

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :return: given $v, w, \xi$ arrays with the last elements replaced
    """
    # TODO: Edit this so that it won't edit the original arrays.

    v_sh = v_shock_bag(xi)
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


def solve_shock(
        model: "Model",
        vp: float,
        wp: float,
        vm_guess: float = None,
        wm_guess: float = None,
        phase: boundary.Phase = boundary.Phase.SYMMETRIC,
        allow_failure: bool = False):
    r"""Solve the boundary conditions at a shock

    :param vp: $\tilde{v}_{+,sh} = \xi_{sh}$
    :param wp: $w_{+,sh} = w_n$
    """
    vm_guess = 1/(3*vp) if vm_guess is None else vm_guess
    wm_guess = wm_shock_bag(vp, wp) if wm_guess is None else wm_guess
    return boundary.solve_junction(
        model,
        v1=vp, w1=wp,
        phase1=phase, phase2=phase,
        v2_guess=vm_guess, w2_guess=wm_guess,
        allow_failure=allow_failure
    )


@numba.njit
def _v_shock_bag_scalar(xi: float) -> float:
    # const.CS0 is used only because it corresponds to the 1/sqrt(3) we need.
    # This has nothing to do with the sound speed!
    if xi < const.CS0:
        return np.nan

    v = (3 * xi**2 - 1) / (2 * xi)
    return v


@numba.njit
def _v_shock_bag_arr(xi: np.ndarray) -> np.ndarray:
    ret = np.zeros_like(xi)
    for i in range(xi.size):
        ret[i] = _v_shock_bag_scalar(xi[i])
    return ret


@numba.generated_jit(nopython=True)
def v_shock_bag(xi: th.FloatOrArr):
    r"""
    Fluid velocity at a shock at $\xi$.
    No shocks exist for $\xi < \frac{1}{\sqrt{3}}$, so this returns zero.
    $$ v_{sh}(\xi) = \frac{3 \xi^2 - 1}{2\xi} $$
    :gw_pt_ssm:`\ `, eq. B.17.

    :param xi: $\xi$
    :return: $v_{sh}$
    """
    if isinstance(xi, numba.types.Float):
        return _v_shock_bag_scalar
    if isinstance(xi, numba.types.Array):
        return _v_shock_bag_arr
    if isinstance(xi, float):
        return _v_shock_bag_scalar(xi)
    if isinstance(xi, np.ndarray):
        return _v_shock_bag_arr(xi)
    raise TypeError(f"Unknown type for xi: {type(xi)}")


@numba.njit
def _wm_shock_bag_scalar(xi: float, w_n: float) -> float:
    # const.CS0 is used only because it corresponds to the 1/sqrt(3) we need.
    # This has nothing to do with the sound speed!
    if xi < const.CS0:
        return np.nan
    return w_n * (9*xi**2 - 1)/(3*(1-xi**2))


@numba.njit
def _wm_shock_bag_arr(xi: np.ndarray, w_n: float) -> np.ndarray:
    ret = np.zeros_like(xi)
    for i in range(xi.size):
        ret[i] = _wm_shock_bag_scalar(xi[i], w_n)
    return ret


# This cannot be vectorized with numba.vectorize due to the keyword argument, but guvectorize might work
@numba.generated_jit(nopython=True)
def wm_shock_bag(xi: th.FloatOrArr, w_n: float = 1.) -> th.FloatOrArrNumba:
    r"""
    Fluid enthalpy behind a shock at $\xi$ in the bag model.
    No shocks exist for $\xi < c_s$, so returns nan.
    Equation B.18 of :gw_pt_ssm:`\ `.

    $$ w_{sh}(\xi) = w_n \frac{9\xi^2 - 1}{3(1 - \xi^2)} $$

    :param xi: $\xi$
    :param w_n: enthalpy in front of the shock
    :return: $w_{sh}$, enthalpy behind the shock
    """
    if isinstance(xi, numba.types.Float):
        return _wm_shock_bag_scalar
    if isinstance(xi, numba.types.Array):
        if not xi.ndim:
            return _wm_shock_bag_scalar
        return _wm_shock_bag_arr
    if isinstance(xi, float):
        return _wm_shock_bag_scalar(xi, w_n)
    if isinstance(xi, np.ndarray):
        if not xi.ndim:
            return _wm_shock_bag_scalar(xi.item(), w_n)
        return _wm_shock_bag_arr(xi, w_n)
    raise TypeError(f"Unknown type for xi: {type(xi)}")


@numba.njit
def _wp_shock_bag_scalar(xi: float, wm: float) -> float:
    # const.CS0 is used only because it corresponds to the 1/sqrt(3) we need.
    # This has nothing to do with the sound speed!
    if xi < const.CS0:
        return np.nan
    return wm * (3*(1-xi**2))/(9*xi**2 - 1)


@numba.njit
def _wp_shock_bag_arr(xi: np.ndarray, wm: float) -> np.ndarray:
    ret = np.zeros_like(xi)
    for i in range(xi.size):
        ret[i] = _wp_shock_bag_scalar(xi[i], wm)
    return ret


# This cannot be vectorized with numba.vectorize due to the keyword argument, but guvectorize might work
@numba.generated_jit(nopython=True)
def wp_shock_bag(xi: th.FloatOrArr, wm: float) -> th.FloatOrArrNumba:
    r"""
    Fluid enthalpy in front of a shock at $\xi$ in the bag model.
    No shocks exist for $\xi < cs$, so returns nan.
    Derived from :gw_pt_ssm:`\ ` eq. B.18.

    $$ w_n(\xi) = w_{-,sh} \frac{3(1 - \xi^2)}{9\xi^2 - 1} $$

    :param xi: $\xi$
    :param wm: $w_{-,sh}$, enthalpy behind the shock
    :return: $w_{+,sh}$, enthalpy in front of the shock
    """
    if isinstance(xi, numba.types.Float):
        return _wp_shock_bag_scalar
    if isinstance(xi, numba.types.Array):
        if not xi.ndim:
            return _wp_shock_bag_scalar
        return _wp_shock_bag_arr
    if isinstance(xi, float):
        return _wp_shock_bag_scalar(xi, wm)
    if isinstance(xi, np.ndarray):
        if not xi.ndim:
            return _wp_shock_bag_scalar(xi.item(), wm)
        return _wp_shock_bag_arr(xi, wm)
    raise TypeError(f"Unknown type for xi: {type(xi)}")
