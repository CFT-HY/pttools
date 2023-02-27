"""Functions for shocks"""

import logging
import typing as tp

import numba.types
import numpy as np

import pttools.type_hints as th
from .boundary import Phase, SolutionType, solve_junction
from . import check
from . import const
from . import props
from . import relativity
if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


@numba.njit
def find_shock_index_bag(v_f: np.ndarray, xi: np.ndarray, v_wall: float, sol_type: SolutionType) -> int:
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


def find_shock_index(
        model: "Model",
        v: np.ndarray, xi: np.ndarray,
        v_wall: float, wn: float,
        sol_type: SolutionType,
        allow_failure: bool = False,
        warn_if_barely_exists: bool = True) -> int:
    if sol_type is SolutionType.DETON:
        return props.find_v_index(xi, v_wall)
    # Todo: replace this with isinstance()
    if model.name == "bag":
        return np.argmax(np.logical_and(xi > v_wall, v <= v_shock_bag(xi)))

    # Trim the integration to the shock
    i_shock = 0
    # The shock curve hits v=0 here
    cs_n = np.sqrt(model.cs2(wn, Phase.SYMMETRIC))
    for i, xi_i in enumerate(xi):
        if xi_i < cs_n:
            continue
        # This can emit a lot of log spam if the warning of a barely existing shock is enabled.
        # pylint: disable=unused-variable
        v_shock_tilde, w_shock = solve_shock(
            model, xi_i, wn,
            backwards=True, warn_if_barely_exists=warn_if_barely_exists)
        v_shock = relativity.lorentz(xi_i, v_shock_tilde)
        if v[i] <= v_shock:
            i_shock = i
            break

    if i_shock == 0:
        if np.max(xi) < const.CS0:
            msg = \
                "The curve turns backwards before reaching the shock. " + \
                "Probably the model does not allow this solution type with these parameters."
            logger.error(msg)
            if not allow_failure:
                raise RuntimeError(msg)
        i_shock = -1

    return i_shock


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
            v1: float,
            w1: float,
            backwards: bool,
            v2_guess: float = None,
            w2_guess: float = None,
            phase: Phase = Phase.SYMMETRIC,
            allow_failure: bool = False,
            warn_if_barely_exists: bool = True) -> tp.Tuple[float, float]:
    r"""Solve the boundary conditions at a shock

    :param model: Hydrodynamics model
    :param v1: $\tilde{v}_{1,sh}$
    :param w1: $w_{1,sh}$
    :param v2_guess: Starting guess for $\tilde{v}_{2,sh}$
    :param w2_guess: Starting guess for $w_{2,sh}$
    :param phase: Phase in which the shock propagates
    :param backwards: whether to solve from $+$ to $-$ instead of from $-$ to $+$
    :param allow_failure: Whether to allow invalid values
    """
    if v1 < 0 or v1 > 1 or np.isclose(v1, 0) or np.isnan(v1):
        logger.error(f"Got invalid v1={v1} for shock solver.")
        return np.nan, np.nan
    if np.isclose(w1, 0) or np.isnan(w1):
        logger.error(f"Got invalid w1={w1} for shock solver.")
        return np.nan, np.nan

    if np.isclose(v1, 1):
        logger.error(f"Got v1={v1} for shock solver.")
        return 1, np.nan

    cs = np.sqrt(model.cs2(w1, phase))
    if np.isclose(v1, cs):
        if warn_if_barely_exists:
            logger.warning(f"The shock barely exists. Got v1={v1}, w1={w1}")
        return v1, w1

    # Bag model guess
    if v2_guess is None:
        v2_guess = 1/(3*v1)
    if np.isclose(v2_guess, 0) or np.isclose(v2_guess, 1):
        logger.error(f"Got invalid estimate for v2={v2_guess}")
        return np.nan, np.nan

    if backwards:
        if v1 < cs:
            logger.error(f"The shock must be supersonic. Got v1=vp={v1}, w1=wp={w1}, cs(wp)={cs}")
            return np.nan, np.nan
        if w2_guess is None:
            w2_guess = wm_shock_bag(v1, w1)
        if np.isnan(w2_guess):
            w2_guess = 0.1*w1
    else:
        if w2_guess is None:
            w2_guess = wp_shock_bag(v1, w1)
        if np.isnan(w2_guess):
            w2_guess = 2*w1
    if w2_guess < 0 or np.isclose(w2_guess, 0):
        logger.error(f"Got invalid estimate for w2={w2_guess}")
        return np.nan, np.nan

    return solve_junction(
        model,
        v1=v1, w1=w1,
        phase1=phase, phase2=phase,
        v2_guess=v2_guess, w2_guess=w2_guess,
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
def _wm_shock_bag_scalar(xi: float, w_n: float, nan_on_negative: bool) -> float:
    # const.CS0 is used only because it corresponds to the 1/sqrt(3) we need.
    # This has nothing to do with the sound speed!
    if nan_on_negative and xi < const.CS0:
        return np.nan
    if xi == 1:
        return np.inf
    return w_n * (9*xi**2 - 1)/(3*(1-xi**2))


@numba.njit
def _wm_shock_bag_arr(xi: np.ndarray, w_n: float, nan_on_negative: bool) -> np.ndarray:
    ret = np.zeros_like(xi)
    for i in range(xi.size):
        ret[i] = _wm_shock_bag_scalar(xi[i], w_n, nan_on_negative)
    return ret


# This cannot be vectorized with numba.vectorize due to the keyword argument, but guvectorize might work
@numba.generated_jit(nopython=True)
def wm_shock_bag(xi: th.FloatOrArr, w_n: float = 1., nan_on_negative: bool = True) -> th.FloatOrArrNumba:
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
        return _wm_shock_bag_scalar(xi, w_n, nan_on_negative)
    if isinstance(xi, np.ndarray):
        if not xi.ndim:
            return _wm_shock_bag_scalar(xi.item(), w_n, nan_on_negative)
        return _wm_shock_bag_arr(xi, w_n, nan_on_negative)
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
