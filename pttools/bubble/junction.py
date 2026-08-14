"""Junction conditions

At the bubble wall (phase boundary), or shock.

.. plot:: fig/vm_vp_plane.py
"""

import functools
import logging
import typing as tp

import numba
import numpy as np

from pttools.bubble import const
from pttools.bubble.junction_entropy import check_entropy_fluxes
from pttools.bubble.relativity import gamma2, lorentz
from pttools.bubble.phase import Phase
from pttools.speedup.solvers import fsolve_vary
import pttools.type_hints as th
if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


@numba.njit(cache=True)
def enthalpy_ratio(v1: th.FloatOrArr, v2: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Ratio of enthalpies $w$ on both sides of a transition front.
    Uses conservation of momentum in moving frame.
    $$\frac{w_1}{w_2} = \frac{\gamma^2 (v_1) v_1}{\gamma^2 (v_2) v_2}$$

    :param v1: $\tilde{v}_1$
    :param v2: $\tilde{v}_2$
    :return: Ratio of enthalpies $\frac{w_1}{w_2}$
    """
    return gamma2(v1) * v1 / (gamma2(v2) * v2)


def junction_conditions_deviation(vp: th.FloatOrArr, vm: th.FloatOrArr, ap: th.FloatOrArr) -> th.FloatOrArr:
    r"""Deviation from the combined junction conditions
    $$\Delta = \left( \frac{1}{\tilde{v}_-} + 3\tilde{v}_- \right)
    \tilde{v}_+ - 3(1 + \alpha_+) \tilde{v}_+^2 - \alpha_+ + 1$$

    :param vp: $\tilde{v}_+$
    :param vm: $\tilde{v}_-$
    :param ap: $\alpha_+$
    """
    dev = (1/vm + 3*vm) * vp - 3 * (1 + ap) * vp**2 + 3 * ap - 1
    if not np.allclose(dev, 0):
        if np.isscalar(dev):
            logger.error(
                "Non-zero deviation from junction conditions: %s for vp=%s, vm=%s, ap=%s",
                dev, vp, vm, ap
            )
        else:
            logger.error("Non-zero deviation from junction conditions")
    return dev


def junction_conditions_solvable(
        params: th.FloatArr1D,
        model: "Model",
        v1: float,
        w1: float,
        phase1: float,
        phase2: float) -> th.FloatArr1D:
    """Get the deviation from both boundary conditions simultaneously."""
    v2 = params[0]
    w2 = params[1]
    return junction_condition_deviations(
        v1=v1, w1=w1, p1=model.p(w=w1, phase=phase1),
        v2=v2, w2=w2, p2=model.p(w=w2, phase=phase2)
    )


@numba.njit(cache=True)
def junction_condition_deviation1(
        v1: th.FloatOrArr, w1: th.FloatOrArr,
        v2: th.FloatOrArr, w2: th.FloatOrArr) -> th.FloatOrArr:
    r"""Deviation from the first junction condition
    $$w_1 \tilde{\gamma}_1^2 \tilde{v}_1 - w_2 \tilde{\gamma}_2^2 \tilde{v}_2$$
    :notes:`\ `, eq. 7.22
    :cutting_2022:`\ `, eq. 19

    :param v1: $\tilde{v}_1$
    :param w1: $w_1$
    :param v2: $\tilde{v}_2$
    :param w2: $w_2$
    """
    return w1 * gamma2(v1) * v1 - w2 * gamma2(v2) * v2


@numba.njit(cache=True)
def junction_condition_deviation2(
        v1: th.FloatOrArr, w1: th.FloatOrArr, p1: th.FloatOrArr,
        v2: th.FloatOrArr, w2: th.FloatOrArr, p2: th.FloatOrArr
    ):
    r"""Deviation from the second junction condition
    $$w_1 \tilde{\gamma}_1^2 \tilde{v}_1^2 + {p}_1 - {w}_2 \tilde{\gamma}_2^2 \tilde{v}_2^2 - {p}_2$$
    :notes:`\ `, eq. 7.22
    :notes:`\ `, eq. 18

    :param v1: $\tilde{v}_1$
    :param w1: $w_1$
    :param p1: $p_1$
    :param v2: $\tilde{v}_2$
    :param w2: $w_2$
    :param p2: $p_2$
    """
    return w1 * gamma2(v1) * v1**2 + p1 - w2 * gamma2(v2) * v2**2 - p2


@numba.njit(cache=True)
def junction_condition_deviations(
        v1: th.FloatOrArr, w1: th.FloatOrArr, p1: th.FloatOrArr,
        v2: th.FloatOrArr, w2: th.FloatOrArr, p2: th.FloatOrArr) -> th.FloatArr1D:
    # This would avoid invalid values in the inner functions, but it may make the solver not to find the solution.
    # if v2 < 0 or w2 < 0:
    #     return np.array([np.nan, np.nan])
    return np.array([
        junction_condition_deviation1(v1=v1, w1=w1, v2=v2, w2=w2),
        junction_condition_deviation2(
            v1=v1, w1=w1, p1=p1,
            v2=v2, w2=w2, p2=p2
        )
    ])


def solve_junction(
        model: "Model",
        v1_tilde: float,
        w1: float,
        phase1: Phase,
        phase2: Phase,
        v2_tilde_guess: float,
        w2_guess: float,
        v2_tilde_min: float | None = None,
        v2_tilde_max: float | None = None,
        w2_min: float | None = None,
        w2_max: float | None = None,
        allow_failure: bool = False,
        allow_negative_entropy_flux_change: bool = False,
        rtol: float = const.JUNCTION_RTOL,
        # atol: float = const.JUNCTION_ATOL,
        debug: bool = False) -> tuple[float, float]:
    r"""Model-independent junction condition solver

    Velocities are in the wall frame.

    :param model: Equation of state
    :param v1_tilde: $\tilde{v}_1$
    :param w1: $w_1$
    :param phase1: $\phi_1$
    :param phase2: $\phi_2$
    :param v2_tilde_guess: $\tilde{v}_{2,\text{guess}}$
    :param w2_guess: $w_{2,\text{guess}}$
    :param v2_tilde_min: $v_{2,\text{min}}$
    :param v2_tilde_max: $v_{2,\text{max}}$
    :param w2_min: $w_{2,\text{min}}$
    :param w2_max: $w_{2,\text{max}}$
    :param allow_failure: Whether to allow or return nan on failure
    :param allow_negative_entropy_flux_change: Whether to allow or return nan on negative entropy flux change
    :param rtol: Relative tolerance
    :param debug: Whether to enable debug logging
    :return: $\tilde{v}_2, w_2$
    """
    if np.isnan(v1_tilde) or np.isnan(w1) or np.isnan(v2_tilde_guess) or np.isnan(w2_guess) \
            or v1_tilde < 0 or v1_tilde > 1 or w1 < 0 or v2_tilde_guess < 0 or v2_tilde_guess > 1 or w2_guess < 0 \
            or np.isclose(v1_tilde, 0) or np.isclose(v1_tilde, 1) \
            or np.isclose(v2_tilde_guess, 0) or np.isclose(v2_tilde_guess, 1) \
            or np.isclose(w1, 0) or np.isclose(w2_guess, 0):
        logger.warning(
            "Got invalid input for the junction solver. The inputs must have 0<v<1 and w>0. "
            "Got: v1=%s, w1=%s, v2_guess=%s, w2_guess=%s",
            v1_tilde, w1, v2_tilde_guess, w2_guess
        )
        return np.nan, np.nan
    if (v2_tilde_min is not None and (v2_tilde_min < 0 or (v2_tilde_max is not None and v2_tilde_max > 0))) or \
            (v2_tilde_max is not None and (v2_tilde_max < 0 or (v2_tilde_min is not None and v2_tilde_min > 0))) or \
            (w2_min is not None and w2_min < 0) or \
            (w2_max is not None and w2_max < 0) or \
            (v2_tilde_min is not None and v2_tilde_max is not None and v2_tilde_max <= v2_tilde_min) or \
            (w2_min is not None and w2_max is not None and w2_max <= w2_min):
        logger.error(
            "Got invalid limits for the junction solver. "
            "Got: v2_tilde_min=%s, v2_tilde_max=%s, w2_min=%s, w2_max=%s",
            v2_tilde_min, v2_tilde_max, w2_min, w2_max
        )

    sol = solve_junction_internal(
        model=model, v1_tilde=v1_tilde, w1=w1,
        phase1=phase1, phase2=phase2,
        v2_tilde_guess=v2_tilde_guess, w2_guess=w2_guess,
        log_status=debug
    )
    v2_tilde = sol[0][0]
    w2 = sol[0][1]
    devs = junction_conditions_solvable(np.array([v2_tilde, w2]), model, v1_tilde, w1, phase1, phase2)

    if sol[2] != 1:
        msg = \
            f"Boundary solution was not found for v1_tilde={v1_tilde}, w1={w1}, model={model.name}. " \
            f"Using v2_tilde={v2_tilde}, w2={w2}. Guess was v2_tilde={v2_tilde_guess}, w2={w2_guess}. " \
            f"Deviations={devs}. " + \
            ("" if (0 < v2_tilde < 1) else "This is unphysical! ") + \
            f"Reason: {sol[3]}"
        logger.error(msg)
        if not allow_failure:
            return np.nan, np.nan

    # TODO: add the Giese et al. junction solver option here (Giese et al. eq. 11)

    devs_rel = devs / w1
    if np.max(np.abs(devs_rel)) > rtol:
        logger.error(
            "The boundary solver gave a solution that deviates from the boundary conditions with "
            "absolute deviation %s, relative deviation %s",
            devs, devs_rel
        )
        if not allow_failure:
            return np.nan, np.nan
    if v2_tilde < 0 or v2_tilde > 1 or w2 < 0:
        logger.error(
            "The boundary solver gave an unphysical solution with "
            "v2_tilde=%s, w2=%s for model=%s, v1_tilde=%s, w1=%s, phase1=%s, phase2=%s.",
            v2_tilde, w2, model.name, v1_tilde, w1, phase1, phase2
        )
        if not allow_failure:
            return np.nan, np.nan

    fail, s_flux1, s_flux2, s_flux_diff = check_entropy_fluxes(
        model=model,
        v1_tilde=v1_tilde, v2_tilde=v2_tilde,
        w1=w1, w2=w2,
        phase1=phase1, phase2=phase2,
        allow_negative_entropy_flux_change=allow_negative_entropy_flux_change
    )
    if fail:
        logger.error(
            "The boundary solver gave an unphysical solution with "
            "v2_tilde=%s, w2=%s for model=%s, v1_tilde=%s, w1=%s, phase1=%s, phase2=%s, "
            "s_flux1=%s, s_flux2=%s, s_flux_diff=%s, deviation1=%s, deviation2=%s, dev1_rel=%s, dev2_rel=%s.",
            v2_tilde, w2, model.name, v1_tilde, w1, phase1, phase2,
            s_flux1, s_flux2, s_flux_diff, devs[0], devs[1], devs_rel[0], devs_rel[1]
        )
        if not allow_failure:
            return np.nan, np.nan

    return v2_tilde, w2


@functools.lru_cache(maxsize=const.JUNCTION_CACHE_SIZE)
def solve_junction_internal(
        model: "Model",
        v1_tilde: float,
        w1: float,
        phase1: Phase,
        phase2: Phase,
        v2_tilde_guess: float,
        w2_guess: float,
        log_status: bool = False) -> th.FSolveOutput:
    # Using fsolve_vary helps in finding the solutions, but it can also make the overall solver a lot slower.
    return fsolve_vary(
        junction_conditions_solvable,
        x0=np.array([v2_tilde_guess, w2_guess]),
        args=(model, v1_tilde, w1, phase1, phase2),
        # This would create a lot of log spam
        log_status=log_status
    )


def v_plus_hybrid(
        model: "Model",
        v_wall: float,
        wm: float,
        vp_tilde_guess: float,
        wp_guess: float,
        allow_failure: bool = False,
        allow_negative_entropy_flux_change: bool = False) -> float:
    """Find $v_+$ for a hybrid"""
    # Exit velocity is at the sound speed
    vm_tilde = np.sqrt(model.cs2(wm, Phase.BROKEN))

    # Solve the boundary conditions at the wall
    vp_tilde, wp = solve_junction(
        model, vm_tilde, wm,
        Phase.BROKEN, Phase.SYMMETRIC,
        v2_tilde_guess=vp_tilde_guess, w2_guess=wp_guess,
        allow_failure=allow_failure,
        allow_negative_entropy_flux_change=allow_negative_entropy_flux_change
    )
    return -lorentz(vp_tilde, v_wall)


@numba.njit(cache=True)
def w2_junction(v1: th.FloatOrArr, w1: th.FloatOrArr, v2: th.FloatOrArr) -> th.FloatOrArr:
    r"""Get $w_-$ from the junction condition 1
    $$w_1 = w_2 \frac{\tilde{\gamma}_2^2 \tilde{v}_2}{\tilde{\gamma}_1^2 \tilde{v}_1}$$
    :notes:`\ `, eq. 7.22
    """
    return w1 * enthalpy_ratio(v1, v2)
