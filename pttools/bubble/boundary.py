"""Functions for calculating the properties of the bubble boundaries

.. plot:: fig/vm_vp_plane.py
"""

import functools
import logging
import typing as tp

import numba
import numpy as np

from pttools.bubble import const
from pttools.bubble.relativity import gamma, gamma2, lorentz
from pttools.bubble.phase import Phase
from pttools.bubble.solution_type import SolutionType
from pttools.bubble.v_plus import v_plus
from pttools.bubble.v_minus import v_minus
from pttools.speedup.solvers import fsolve_vary
import pttools.type_hints as th
if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


def check_entropy_fluxes(
        model: "Model",
        v1_tilde: float,
        v2_tilde: float,
        w1: float,
        w2: float,
        phase1: Phase,
        phase2: float,
        allow_negative_entropy_flux_change: bool = False) -> tuple[bool, float, float]:
    """False = OK, True = fail"""
    s1 = model.s(w1, phase1)
    s2 = model.s(w2, phase2)
    entropy_flux1 = entropy_flux(v1_tilde, s1)
    entropy_flux2 = entropy_flux(v2_tilde, s2)
    fail_individual = entropy_flux1 < 0 or entropy_flux2 < 0
    if allow_negative_entropy_flux_change:
        fail_total = False
    else:
        fail_total = (
            (phase1 == Phase.SYMMETRIC and phase2 == Phase.BROKEN and entropy_flux1 - entropy_flux2 < 0) or
            (phase1 == Phase.BROKEN and phase2 == Phase.SYMMETRIC and entropy_flux2 - entropy_flux1 < 0)
        )
    # fail_total = False
    return fail_individual or fail_total, entropy_flux1, entropy_flux2


@numba.njit
def enthalpy_ratio(v_m: th.FloatOrArr, v_p: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Ratio of enthalpies behind ($w_-$) and ahead $(w_+)$ of a shock or
    transition front, $w_-/w_+$.
    Uses conservation of momentum in moving frame.

    $$\frac{\gamma^2 (v_m) v_m}{\gamma^2 (v_p) v_p}$$

    :param v_m: $v_-$
    :param v_p: $v_+$
    :return: enthalpy ratio
    """
    return gamma2(v_m) * v_m / (gamma2(v_p) * v_p)


def entropy_flux(v_tilde: th.FloatOrArr, s: th.FloatOrArr) -> th.FloatOrArr:
    r"""Entropy flux $\gamma(\tilde{v}) \tilde{v} s$"""
    return gamma(v_tilde) * v_tilde * s


@numba.njit
def fluid_speeds_at_wall(
        v_wall: float,
        alpha_plus: float,
        sol_type: SolutionType) -> tuple[float, float, float, float]:
    r"""
    Solves fluid speed boundary conditions at the wall to obtain
    the fluid speeds both in the universe (plasma frame): $v_+$ and $v_+$
    and in the wall frame: $\tilde{v}_+, \tilde{v}_-$.

    Bag model only!

    The abbreviations are: fluid speed (vf) just behind (m=minus) and just ahead (p=plus) of wall,
    in wall (_w) and plasma/universe (_p) frames.

    TODO: add a validity check for v_minus

    :param v_wall: $v_\text{wall}$
    :param alpha_plus: $\alpha_+$
    :param sol_type: solution type
    :return: $\tilde{v}_+,\tilde{v}_-,v_+,v_-$
    """
    if v_wall > 1:
        # Todo: better error handling and logging
        # with numba.objmode:
        #     logger.error("v_wall > 1: v_wall = %s", v_wall)
        raise ValueError(f"Got v_wall = {v_wall} > 1")

    # print("max_speed_deflag(alpha_plus)=", max_speed_deflag(alpha_plus))
    # if v_wall < max_speed_deflag(alpha_plus) and v_wall <= cs and alpha_p <= 1/3.:
    if sol_type == SolutionType.SUB_DEF.value:
        # For clarity these are defined here in the same order as returned
        vfp_w = v_plus(v_wall, alpha_plus, sol_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
        vfm_w = v_wall  # Fluid velocity just behind the wall in wall frame (v-)
        vfp_p = lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        vfm_p = lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
    elif sol_type == SolutionType.HYBRID.value:
        vfp_w = v_plus(const.CS0, alpha_plus, sol_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
        vfm_w = const.CS0  # Fluid velocity just behind the wall in plasma frame (hybrid)
        vfp_p = lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        vfm_p = lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
    elif sol_type == SolutionType.DETON.value:
        vfp_w = v_wall  # Fluid velocity just ahead of the wall in wall frame (v+)
        vfm_w = v_minus(v_wall, alpha_plus)  # Fluid velocity just behind the wall in wall frame (v-)
        vfp_p = lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        vfm_p = lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
    else:
        # Todo: better error handling and logging
        # with numba.objmode:
        #     logger.error("Unknown sol_type: %s", sol_type)
        raise ValueError(f"Unknown sol_type={sol_type}")

    return vfp_w, vfm_w, vfp_p, vfm_p


def junction_conditions_deviation(vp: th.FloatOrArr, vm: th.FloatOrArr, ap: th.FloatOrArr) -> th.FloatOrArr:
    r"""Deviation from the combined junction conditions
    $$\Delta = \left( \frac{1}{\tilde{v}_-} + 3\tilde{v}_- \right)
    \tilde{v}_+ - 3(1 + \alpha_+) \tilde{v}_+^2 - \alpha_+ + 1$$
    """
    dev = (1/vm + 3*vm)*vp - 3*(1 + ap)*vp**2 + 3*ap - 1
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
    # This would avoid invalid values in the inner functions, but it may make the solver not to find the solution.
    # if v2 < 0 or w2 < 0:
    #     return np.array([np.nan, np.nan])
    return np.array([
        junction_condition_deviation1(v1, w1, v2, w2),
        junction_condition_deviation2(
            v1, w1, model.p(w1, phase1),
            v2, w2, model.p(w2, phase2)
        )
    ])


@numba.njit
def junction_condition_deviation1(
        v1: th.FloatOrArr, w1: th.FloatOrArr,
        v2: th.FloatOrArr, w2: th.FloatOrArr) -> th.FloatOrArr:
    r"""Deviation from the first junction condition
    $$w_- \tilde{\gamma}_-^2 \tilde{v}_- - w_+ \tilde{\gamma}_-^2 \tilde{v}_+$$
    :notes:`\ `, eq. 7.22
    :cutting_2022:`\ `, eq. 19
    """
    return w1 * gamma2(v1) * v1 - w2 * gamma2(v2) * v2


@numba.njit
def junction_condition_deviation2(
        v1: th.FloatOrArr, w1: th.FloatOrArr, p1: th.FloatOrArr,
        v2: th.FloatOrArr, w2: th.FloatOrArr, p2: th.FloatOrArr
    ):
    r"""Deviation from the second junction condition
    $$w_1 \tilde{\gamma}_1^2 \tilde{v}_1^2 + {p}_1 - {w}_2 \tilde{\gamma}_2^2 \tilde{v}_2^2 - {p}_2$$
    :notes:`\ `, eq. 7.22
    :notes:`\ `, eq. 18
    """
    return w1 * gamma2(v1) * v1**2 + p1 - w2 * gamma2(v2) * v2**2 - p2


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
    """Model-independent junction condition solver

    Velocities are in the wall frame!
    """
    if np.isnan(v1_tilde) or np.isnan(w1) or np.isnan(v2_tilde_guess) or np.isnan(w2_guess) \
            or v1_tilde < 0 or v1_tilde > 1 or w1 < 0 or v2_tilde_guess < 0 or v2_tilde_guess > 1 or w2_guess < 0 \
            or np.isclose(v1_tilde, 0) or np.isclose(v1_tilde, 1) \
            or np.isclose(v2_tilde_guess, 0) or np.isclose(v2_tilde_guess, 1) \
            or np.isclose(w1, 0) or np.isclose(w2_guess, 0):
        logger.warning(
            "Invalid input for junction solver. The inputs must have 0<v<1 and w>0. "
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
            "Invalid limits for junction solver. "
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

    if sol[2] != 1:
        msg = \
            f"Boundary solution was not found for v1_tilde={v1_tilde}, w1={w1}, model={model.name}. " \
            f"Using v2_tilde={v2_tilde}, w2={w2}. Guess was v2_tilde={v2_tilde_guess}, w2={w2_guess}. " \
            f"Deviations={junction_conditions_solvable(np.array([v2_tilde, w2]), model, v1_tilde, w1, phase1, phase2)}. " + \
            ("" if (0 < v2_tilde < 1) else "This is unphysical! ") + \
            f"Reason: {sol[3]}"
        logger.error(msg)
        if not allow_failure:
            return np.nan, np.nan
            # logger.error("ERROR")
            # raise ValueError(msg)

    # TODO: add the Giese et al. junction solver option here (Giese et al. eq. 11)

    devs = junction_conditions_solvable(np.array([v2_tilde, w2]), model, v1_tilde, w1, phase1, phase2)
    devs_rel = devs / w1
    # print(f"v1w={v1}, v2w={v2}, w1={w1}, w2={w2}, dev={devs}")
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

    fail, s_flux1, s_flux2 = check_entropy_fluxes(
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
            "s_flux1=%s, s_flux2=%s.",
            v2_tilde, w2, model.name, v1_tilde, w1, phase1, phase2, s_flux1, s_flux2
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


def w2_junction(v1: th.FloatOrArr, w1: th.FloatOrArr, v2: th.FloatOrArr) -> th.FloatOrArr:
    r"""Get $w_-$ from the junction condition 1
    $$w_1 = w_2 \frac{\tilde{\gamma}_2^2 \tilde{v}_2}{\tilde{\gamma}_1^2 \tilde{v}_1}$$
    :notes:`\ `, eq. 7.22
    """
    # Todo: use enthalpy_ratio() for this
    wm = w1 * gamma2(v1) * v1 / (gamma2(v2) * v2)
    # wm > wp for detonations
    # if wm > wp:
    #     logger.warning(f"wm_junction resulted in wm > wp: vp={vp}, wp={wp}, vm={vm}, wm={wm}")
    return wm
