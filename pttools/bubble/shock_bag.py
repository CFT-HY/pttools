"""Functions for shocks"""

import logging

from numba.extending import overload
import numba.types
import numpy as np

from pttools.bubble import check
from pttools.bubble import const
from pttools.bubble import props
from pttools.bubble.solution_type import SolutionType
from pttools.speedup import NUMBA_ENABLE_CACHE
import pttools.type_hints as th

logger = logging.getLogger(__name__)


@numba.njit
def find_shock_index_bag(v_f: th.FloatArr1D, xi: th.FloatArr1D, v_wall: float, sol_type: SolutionType) -> int:
    r"""
    Array index of shock from first point where fluid velocity $v_f$ goes below $v_\text{shock}$.
    For detonation, returns wall position.

    :param v_f: fluid velocity $v_f$
    :param xi: $\xi$
    :param v_wall: wall velocity $v_\text{wall}$
    :param sol_type: solution type (detonation etc.)
    :return: shock index
    """
    logger.warning("DEPRECATED")
    check.check_wall_speed(v_wall)

    n_shock = 0
    if sol_type == SolutionType.DETON:
        n_shock = props.find_v_index(xi, v_wall)
    else:
        for i, (v, x) in enumerate(zip(v_f, xi)):
            if x > v_wall and v <= v_shock_bag(x):
                n_shock = i
                break

    return n_shock


def _v_shock_bag_scalar(xi: th.FloatOrArr) -> th.FloatOrArr:
    # const.CS0 is used only because it corresponds to the 1/sqrt(3) we need.
    # This has nothing to do with the sound speed!
    if xi < const.CS0:
        return np.nan

    v = (3 * xi**2 - 1) / (2 * xi)
    return v


_v_shock_bag_scalar_numba = numba.njit(_v_shock_bag_scalar)


def _v_shock_bag_arr(xi: th.FloatOrArr) -> th.FloatArr:
    ret = np.zeros_like(xi)
    for i in numba.prange(xi.size):
        ret[i] = _v_shock_bag_scalar_numba(xi[i])
    return ret


def v_shock_bag(xi: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Fluid velocity at a shock at $\xi$.
    No shocks exist for $\xi < \frac{1}{\sqrt{3}}$, so this returns zero.
    $$ v_{sh}(\xi) = \frac{3 \xi^2 - 1}{2\xi} $$
    :gw_pt_ssm:`\ `, eq. B.17.

    :param xi: $\xi$
    :return: $v_{sh}$
    """
    if isinstance(xi, float):
        return _v_shock_bag_scalar(xi)
    if isinstance(xi, np.ndarray):
        return _v_shock_bag_arr(xi)
    raise TypeError(f"Unknown type for xi: {type(xi)}")


@overload(v_shock_bag, jit_options={"nopython": True, "cache": NUMBA_ENABLE_CACHE})
def _v_shock_bag_numba(xi: th.FloatOrArr) -> th.NumbaFunc:
    if isinstance(xi, numba.types.Float):
        return _v_shock_bag_scalar
    if isinstance(xi, numba.types.Array):
        return _v_shock_bag_arr
    raise TypeError(f"Unknown type for xi: {type(xi)}")


def _wm_shock_bag_scalar(xi: th.FloatOrArr, w_n: float = 1., nan_on_negative: bool = True) -> th.FloatOrArr:
    # const.CS0 is used only because it corresponds to the 1/sqrt(3) we need.
    # This has nothing to do with the sound speed!
    if nan_on_negative and xi < const.CS0:
        return np.nan
    if xi == 1:
        return np.inf
    return w_n * (9*xi**2 - 1)/(3*(1-xi**2))


def _wm_shock_bag_arr(xi: th.FloatOrArr, w_n: float = 1., nan_on_negative: bool = True) -> th.FloatArr:
    ret = np.zeros_like(xi)
    for i in range(xi.size):
        ret[i] = _wm_shock_bag_scalar(xi[i], w_n, nan_on_negative)
    return ret


# This cannot be vectorized with numba.vectorize due to the keyword argument, but guvectorize might work
def wm_shock_bag(xi: th.FloatOrArr, w_n: float = 1., nan_on_negative: bool = True) -> th.FloatOrArr:
    r"""
    Fluid enthalpy behind a shock at $\xi$ in the bag model.
    No shocks exist for $\xi < c_s$, so returns nan.
    Equation B.18 of :gw_pt_ssm:`\ `.

    $$ w_{sh}(\xi) = w_n \frac{9\xi^2 - 1}{3(1 - \xi^2)} $$

    :param xi: $\xi$
    :param w_n: enthalpy in front of the shock
    :return: $w_{sh}$, enthalpy behind the shock
    """
    if isinstance(xi, float):
        return _wm_shock_bag_scalar(xi, w_n, nan_on_negative)
    if isinstance(xi, np.ndarray):
        if not xi.ndim:
            return _wm_shock_bag_scalar(xi.item(), w_n, nan_on_negative)
        return _wm_shock_bag_arr(xi, w_n, nan_on_negative)
    raise TypeError(f"Unknown type for xi: {type(xi)}")


@overload(wm_shock_bag, jit_options={"nopython": True, "cache": NUMBA_ENABLE_CACHE})
def _wm_shock_bag_numba(xi: th.FloatOrArr, w_n: float = 1., nan_on_negative: bool = True) -> th.NumbaFunc:
    if isinstance(xi, numba.types.Float):
        return _wm_shock_bag_scalar
    if isinstance(xi, numba.types.Array):
        if not xi.ndim:
            return _wm_shock_bag_scalar
        return _wm_shock_bag_arr
    raise TypeError(f"Unknown type for xi: {type(xi)}")


def _wp_shock_bag_scalar(xi: float, wm: float) -> float:
    # const.CS0 is used only because it corresponds to the 1/sqrt(3) we need.
    # This has nothing to do with the sound speed!
    if xi < const.CS0:
        return np.nan
    return wm * (3*(1-xi**2)) / (9*xi**2 - 1)


def _wp_shock_bag_arr(xi: np.ndarray, wm: float) -> np.ndarray:
    ret = np.zeros_like(xi)
    for i in range(xi.size):
        ret[i] = _wp_shock_bag_scalar(xi[i], wm)
    return ret


# This cannot be vectorized with numba.vectorize due to the keyword argument, but guvectorize might work
def wp_shock_bag(xi: th.FloatOrArr, wm: float) -> th.FloatOrArr:
    r"""
    Fluid enthalpy in front of a shock at $\xi$ in the bag model.
    No shocks exist for $\xi < cs$, so returns nan.
    Derived from :gw_pt_ssm:`\ ` eq. B.18.

    $$ w_n(\xi) = w_{-,sh} \frac{3(1 - \xi^2)}{9\xi^2 - 1} $$

    :param xi: $\xi$
    :param wm: $w_{-,sh}$, enthalpy behind the shock
    :return: $w_{+,sh}$, enthalpy in front of the shock
    """
    if isinstance(xi, float):
        return _wp_shock_bag_scalar(xi, wm)
    if isinstance(xi, np.ndarray):
        if not xi.ndim:
            return _wp_shock_bag_scalar(xi.item(), wm)
        return _wp_shock_bag_arr(xi, wm)
    raise TypeError(f"Unknown type for xi: {type(xi)}")


@overload(wp_shock_bag, jit_options={"nopython": True, "cache": NUMBA_ENABLE_CACHE})
def _wp_shock_bag_numba(xi: th.FloatOrArr, wm: float) -> th.NumbaFunc:
    if isinstance(xi, numba.types.Float):
        return _wp_shock_bag_scalar
    if isinstance(xi, numba.types.Array):
        if not xi.ndim:
            return _wp_shock_bag_scalar
        return _wp_shock_bag_arr
    raise TypeError(f"Unknown type for xi: {type(xi)}")
