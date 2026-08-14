"""Functions for shocks"""

import logging
import typing as tp

import numba.types
import numpy as np

from pttools.bubble.junction import solve_junction, w2_junction
from pttools.bubble import const
from pttools.bubble import props
from pttools.bubble.phase import Phase
from pttools.bubble import relativity
from pttools.bubble.shock_bag import v_shock_bag
from pttools.bubble.solution_type import SolutionType
import pttools.type_hints as th

if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


def find_shock_index(
        model: "Model",
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        wn: float,
        cs_n: float,
        sol_type: SolutionType,
        v_shock_atol: float = 3.5e-8,
        error_on_failure: bool = True,
        zero_on_failure: bool = True,
        log_failure: bool = True,
        warn_if_barely_exists: bool = True) -> int | np.signedinteger:
    if sol_type is SolutionType.DETON:
        return props.find_v_index(xi, v_wall)
    # Todo: replace this with isinstance()
    if model.name == "bag":
        points_after_shock = np.logical_and(xi > v_wall, v <= v_shock_bag(xi))
        if np.sum(points_after_shock):
            return np.argmax(points_after_shock)
        points_near_cs = np.logical_and(np.isclose(xi, const.CS0), np.isclose(v, 0))
        if np.sum(points_near_cs):
            return np.argmax(points_near_cs)
        raise RuntimeError("Did not find shock for the bag model.")

    params = f"model={model.label_unicode}, sol_type={sol_type}, v_wall={v_wall}, wn={wn}, cs_n={cs_n}"

    xi_nan = np.any(np.isnan(xi))
    v_nan = np.any(np.isnan(v))
    if xi_nan or v_nan:
        if xi_nan and v_nan:
            msg = f"xi and v have nan values for {params}"
        elif xi_nan:
            msg = f"xi has nan values for {params}"
        else:  # if v_nan
            msg = f"v has nan values for {params}"

        if log_failure:
            logger.error(msg)
        if error_on_failure:
            raise RuntimeError(msg)
        if zero_on_failure:
            return 0

    if not np.isclose(xi[0], v_wall):
        msg = f"Invalid xi data: does not start with v_wall. Got xi[0]={xi[0]}, v_wall={v_wall}"
        if log_failure:
            logger.error(msg)
        if error_on_failure:
            raise RuntimeError(msg)
        if zero_on_failure:
            return 0

    # Index of highest xi = where the curve turns backwards
    i_right: np.signedinteger = np.argmax(xi)

    # If the curve starts by going to the left, then it's a detonation.
    if i_right == 0:
        msg = f"The given xi array seems to be for a detonation, but got {params}"
        if log_failure:
            logger.error(msg)
        if error_on_failure:
            raise RuntimeError(msg)
        if zero_on_failure:
            return 0

    # First index close to xi=cs_n, v=0
    i_close: np.signedinteger = np.argmax(np.logical_and(np.isclose(xi, cs_n), np.isclose(v, 0)))
    # First index where xi > cs_n
    i_cs_n: np.signedinteger = np.argmax(xi > cs_n)

    # If the curve goes directly to zero at cs_n
    # = (if a point close to xi=cs_n, v=0 exists) and ((all points < cs_n) or (approaches cs_n from the left))
    if i_close != 0 and ((i_cs_n == 0 and xi[0] < cs_n) or i_close <= i_cs_n):
        return i_close

    # The lower limit for the shock search is where the shock curve hits v=0
    i_left = i_cs_n

    if i_left > i_right:
        msg = \
            "Shock index finder started with invalid values: " \
            f"i_left={i_left}, i_right={i_right}, cs_n={cs_n}, xi={xi}"
        if log_failure:
            logger.error(msg)
        if error_on_failure:
            raise RuntimeError(msg)
        if zero_on_failure:
            return 0

    # The curve should go to the right before turning back at xi_max
    if not np.all(np.diff(xi[i_left:i_right])):
        msg = f"xi is not monotonic from cs_n to xi_max for {params}"
        if log_failure:
            logger.error(msg)
        if error_on_failure:
            raise RuntimeError(msg)
        if zero_on_failure:
            return 0

    v_sh = v_shock(model, wn=wn, xi=xi[0], cs_n=cs_n, warn_if_barely_exists=warn_if_barely_exists)
    if v[0] < v_sh:
        msg = f"The starting point is already beyond the shock: xi={xi[0]}, v={v[0]}, w={w[0]}, v_shock={v_sh}"
        if log_failure:
            logger.error(msg)
        if error_on_failure:
            raise RuntimeError(msg)
        if zero_on_failure:
            return 0

    # if v[i_left] < v_shock(model, wn=wn, xi=xi[i_left], warn_if_barely_exists=warn_if_barely_exists):
    #     msg = "v at cs_n > v_shock"
    #     if not allow_failure:
    #         raise RuntimeError(msg)

    v_sh_xi_max = v_shock(model, wn=wn, xi=xi[i_right], cs_n=cs_n, warn_if_barely_exists=warn_if_barely_exists)
    # If the shock curve is not reached before the curve turns backwards
    if v[i_right] > v_sh_xi_max:
        # If the intersection is close to the point where the curve turns backwards,
        # we can use the right-most point as the point of the intersection.
        if v_sh_xi_max - v[i_right] < v_shock_atol:
            return i_right

        # Fix for tiny shocks
        # if np.isclose(xi[i_right], cs_n, atol=0.02):
        #     return i_close
        # if np.isclose(v[i_cs_n], 0, atol=0.01):
        #     return i_cs_n

        v_sh_cs_n = v_shock(model, wn=wn, xi=cs_n, cs_n=cs_n, warn_if_barely_exists=warn_if_barely_exists)
        msg = \
            "The curve does not reach the shock. (E.g. it turns backwards before reaching the shock.) " + \
            f"i_cs_n={i_cs_n}, i_xi_max={i_right}, i_close={i_close} " + \
            f"cs_n={cs_n}, xi_max={xi[i_right]}, " + \
            f"v_cs_n={v[i_cs_n]}, v_xi_max={v[i_right]}, " + \
            f"v_sh_cs_n={v_sh_cs_n}, v_sh_xi_max={v_sh_xi_max}"
        if log_failure:
            logger.error(msg)

        # Manual fallback to the old shock finder
        # This may not find the shock correctly until after the curve has turned backwards.

        # i_sh = 0
        # for i, (xi_i, v_i) in enumerate(zip(xi, v)):
        #     # The shock can be so tiny that it cannot be distinguished.
        #     if np.isclose(xi_i, cs_n) and np.isclose(v_i, 0):
        #         i_sh = i
        #         break
        #
        #     if xi_i < cs_n:
        #         continue
        #     # This can emit a lot of log spam if the warning of a barely existing shock is enabled.
        #     # pylint: disable=unused-variable
        #     v_shock_tilde, w_shock = solve_shock(
        #         model,
        #         v1_tilde=xi_i, w1=wn,
        #         backwards=True, warn_if_barely_exists=warn_if_barely_exists
        #     )
        #     v_sh = relativity.lorentz(xi_i, v_shock_tilde)
        #     if v[i] <= v_sh:
        #         i_sh = i
        #         break
        # return i_sh

        if error_on_failure:
            raise RuntimeError(msg)
        if zero_on_failure:
            return 0

    # -----
    # Binary search
    # -----
    i_sh = 0
    while i_right - i_left > 1:
        i_sh = i_left + (i_right - i_left) // 2
        # logger.debug(f"i_left={i_left}, i_right={i_right}, i_sh={i_sh}")
        v_i = v[i_sh]
        v_sh = v_shock(model, wn=wn, xi=xi[i_sh], cs_n=cs_n, warn_if_barely_exists=warn_if_barely_exists)
        if v_i > v_sh:
            i_left = i_sh
        elif v_i < v_sh:
            i_right = i_sh
        else:
            return i_sh + 1

    # -----
    # Output validation
    # -----
    if i_sh == 0:
        msg = \
            "Shock index finder ended up in an invalid state with " \
            f"i_left={i_left}, i_right={i_right}, i_sh={i_sh} for {params}."
        if log_failure:
            logger.error(msg)
        if error_on_failure:
            raise RuntimeError(msg)
        if zero_on_failure:
            return 0

    if v[i_sh] > v_shock(model, wn=wn, xi=xi[i_sh], cs_n=cs_n, warn_if_barely_exists=warn_if_barely_exists):
        # logger.warning("Manually correcting i_sh with +1 to %s", i_sh)
        if v[i_sh + 1] > v_shock(model, wn, xi[i_sh + 1], cs_n=cs_n, warn_if_barely_exists=warn_if_barely_exists):
            msg = f"i_sh + 1 should be beyond the shock for {params}"
            if log_failure:
                logger.error(msg)
            if error_on_failure:
                raise RuntimeError(msg)
            if zero_on_failure:
                return 0
        return i_sh + 1
    return i_sh


@numba.njit
def shock_zoom_last_element(
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]:
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
            v1_tilde: float,
            w1: float,
            backwards: bool,
            v2_tilde_guess: float | None = None,
            w2_guess: float | None = None,
            csp: float | None = None,
            phase: Phase = Phase.SYMMETRIC,
            allow_failure: bool = False,
            warn_if_barely_exists: bool = True) -> tuple[float, float]:
    r"""Solve the boundary conditions at a shock

    :param model: Hydrodynamics model
    :param v1_tilde: $\tilde{v}_{1,sh}$
    :param w1: $w_{1,sh}$
    :param backwards: whether to solve from $+$ to $-$ instead of from $-$ to $+$
    :param v2_tilde_guess: Starting guess for $\tilde{v}_{2,sh}$
    :param w2_guess: Starting guess for $w_{2,sh}$
    :param csp: Speed of sound in front of the shock. Will be computed from the model if not given.
        The computation is only an approximation when solving forwards.
    :param phase: Phase in which the shock propagates
    :param allow_failure: Whether to allow invalid values
    :param warn_if_barely_exists: Warn if the shock barely exists
    :return: $\tilde{v}_{2,sh},w2$
    """
    # Handle invalid inputs
    if v1_tilde < 0 or v1_tilde > 1 or np.isclose(v1_tilde, 0) or np.isnan(v1_tilde):
        logger.error("Got invalid v1=%s for shock solver.", v1_tilde)
        return np.nan, np.nan
    if np.isclose(w1, 0) or np.isnan(w1):
        logger.error("Got invalid w1=%s for shock solver.", w1)
        return np.nan, np.nan

    if np.isclose(v1_tilde, 1):
        logger.error("Got v1=%s for shock solver.", v1_tilde)
        return 1, np.nan

    if csp is None:
        # When solving forwards, this is only an approximation, but a valid one for tiny shocks.
        csp2 = model.cs2(w1, phase)
        csp = np.sqrt(csp2)
    else:
        csp2 = csp**2

    # If the shock barely exists
    if np.isclose(v1_tilde, csp):
        if warn_if_barely_exists:
            logger.warning("The shock barely exists. Got v1=%s, w1=%s", v1_tilde, w1)
        return v1_tilde, w1

    if v2_tilde_guess is None:
        # Bag model guess
        # v2_tilde_guess = 1 / (3 * v1_tilde)
        # General guess
        v2_tilde_guess = csp2 * v1_tilde
    if np.isclose(v2_tilde_guess, 0) or np.isclose(v2_tilde_guess, 1):
        logger.error("Got invalid estimate for v2=%s", v2_tilde_guess)
        return np.nan, np.nan

    if backwards and v1_tilde < csp:
        logger.error(
            "The shock must be supersonic. Got v1=vp=%s, w1=wp=%s, cs(wp)=%s",
            v1_tilde, w1, csp
        )
        return np.nan, np.nan

    # Old guess based on the bag model. It does not work for xi**2 < 1/3.
    # if backwards:
    #     if w2_guess is None:
    #         # The bag guess does not work when
    #         w2_guess = wm_shock_bag(v1_tilde, w1)
    #     if np.isnan(w2_guess):
    #         w2_guess = 0.1*w1
    # else:
    #     if w2_guess is None:
    #         w2_guess = wp_shock_bag(v1_tilde, w1)

    if w2_guess is None:
        w2_guess = w2_junction(v1_tilde, w1, v2_tilde_guess)

    if w2_guess < 0 or np.isclose(w2_guess, 0):
        logger.error("Got invalid estimate for w2=%s", w2_guess)
        return np.nan, np.nan

    return solve_junction(
        model,
        v1_tilde=v1_tilde, w1=w1,
        phase1=phase, phase2=phase,
        v2_tilde_guess=v2_tilde_guess, w2_guess=w2_guess,
        allow_failure=allow_failure
    )


@np.vectorize
def v_shock(model: "Model", wn: float, xi: float, cs_n: float, warn_if_barely_exists: bool = True) -> float:
    r"""Shock velocity $v_\text{sh}$"""
    if xi <= cs_n or np.isclose(xi, cs_n):
        return 0.
    if np.isclose(xi, 1.):
        return 1.

    # This can emit a lot of log spam if the warning of a barely existing shock is enabled.
    # pylint: disable=unused-variable
    v_shock_tilde, w_shock = solve_shock(
        model,
        v1_tilde=xi, w1=wn,
        csp=cs_n,
        backwards=True, warn_if_barely_exists=warn_if_barely_exists
    )
    v_sh = relativity.lorentz(xi, v_shock_tilde)
    # print(f"v1=0, v2={ret}, v2_tilde={v_shock_tilde}, wn={wn}, w_sh={w_shock}")

    if v_sh > 1.:
        return 1.
    if v_sh < 0.:
        return 0.
    return v_sh


def v_shock_curve(
        model: "Model",
        wn: float,
        xi: th.FloatArr1D | None = None,
        n_points: int = 20,
        warn_if_barely_exists: bool = False) -> tuple[th.FloatArr1D, th.FloatArr1D]:
    r"""Shock velocity curve $(\xi, v_{\text{sh}})$"""
    if xi is None:
        cs_n = np.sqrt(model.cs2(wn, Phase.SYMMETRIC))
        # Create more points near cs_n, as there the accuracy is the most critical
        xi = cs_n + np.logspace(-4, 0, num=n_points) * (1 - cs_n)
        # Ensure that the shock curve starts from xi=cs_n, v=0
        xi[0] = cs_n
    return xi, v_shock(model, wn, xi, warn_if_barely_exists)
