"""Some functions useful for the bag equation of state

See page 37 of the lecture notes 10.21468/SciPostPhysLectNotes.24
"""

# import enum
import logging
import typing as tp

import numba
import numpy as np

import pttools.type_hints as th
from pttools import speedup
from . import const

logger = logging.getLogger(__name__)

CS2_FUN_TYPE = tp.Callable[[th.FLOAT_OR_ARR], float]


# TODO: think about using an enum for the phases
# @enum.unique
# class Phase(enum.IntEnum):
#     SYMMETRIC = 0
#     BROKEN = 1


# def cs_w(w):
#    # Speed of sound function, another label
#    # to be adapted to more realistic equations of state, e.g. with interpolation
#    return cs0
#
#
# def cs2_w(w):
#    # Speed of sound squared function
#    # to be adapted to more realistic equations of state, e.g. with interpolation
#    return cs0_2


@numba.njit
def check_thetas(theta_s: th.FLOAT_OR_ARR, theta_b: th.FLOAT_OR_ARR) -> None:
    if np.any(theta_b >= theta_s):
        with numba.objmode:
            logger.error(
                "theta_b should always be smaller than theta_s, "
                f"but got theta_s=%s, theta_b=%s", theta_s, theta_b)
        raise ValueError("theta_b should always be smaller than theta_s")


@numba.njit
def cs2_bag_scalar(w: float) -> float:
    """The scalar versions of the bag functions have to be compiled to cfuncs if jitting is disabled,
    as otherwise the cfunc version of the differential cannot be created.
    """
    return const.CS0_2


@numba.njit
def cs2_bag_arr(w: np.ndarray) -> np.ndarray:
    return const.CS0_2 * np.ones_like(w)


@numba.generated_jit(nopython=True)
def cs2_bag(w: th.FLOAT_OR_ARR):
    r"""
    Speed of sound squared in Bag model, equal to $\frac{1}{3}$ independent of enthalpy $w$
    """
    if isinstance(w, numba.types.Float):
        return cs2_bag_scalar
    if isinstance(w, numba.types.Array):
        return cs2_bag_arr
    if isinstance(w, float):
        return cs2_bag_scalar(w)
    if isinstance(w, np.ndarray):
        return cs2_bag_arr(w)
    raise TypeError(f"Unknown type for w: {type(w)}")


def theta_bag(w: th.FLOAT_OR_ARR, phase: th.INT_OR_ARR, alpha_n: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    r"""
    Trace anomaly $\theta = \frac{1}{4} (e - 3p)$ in the Bag model.
    Equation 7.24 in the lecture notes, equation 2.10 in the article

    :param w: enthalpy $w$
    :param phase: phase(s)
    :param alpha_n: strength of the transition $\alpha_n$
    :return: trace anomaly $\theta_\text{bag}$
    """
    if isinstance(w, np.ndarray):
        w_n = w[-1]
    else:
        w_n = w
    return alpha_n * (0.75 * w_n) * (1 - phase)


@numba.njit
def get_p(
        w: th.FLOAT_OR_ARR,
        phase: th.INT_OR_ARR,
        theta_s: th.FLOAT_OR_ARR,
        theta_b: th.FLOAT_OR_ARR = 0.) -> th.FLOAT_OR_ARR:
    r"""
    Pressure as a function of enthalpy $w$, assuming bag model.
    $\theta = \frac{e - 3p}{4}$ (trace anomaly or "vacuum energy").
    Enthalpy, theta and phase can be arrays of the same shape.
    See also the equation 4.40.

    :param w: enthalpy $w$
    :param phase: phase indicator
    :param theta_s: $\theta$ for symmetric phase, ahead of bubble (phase = 0)
    :param theta_b: $\theta$ for broken phase, behind bubble (phase = 1)
    :return: pressure $p$
    """
    check_thetas(theta_s, theta_b)
    theta = theta_b * phase + theta_s * (1.0 - phase)
    return 0.25 * w - theta


@numba.njit
def get_e(
        w: th.FLOAT_OR_ARR,
        phase: th.INT_OR_ARR,
        theta_s: th.FLOAT_OR_ARR,
        theta_b: th.FLOAT_OR_ARR = 0.) -> th.FLOAT_OR_ARR:
    r"""
    Energy density as a function of enthalpy $w$, assuming bag model.
    $\theta = \frac{e - 3p}{4}$ ("vacuum energy").
    Enthalpy and phase can be arrays of the same shape.
    See also the equation 4.10.

    :param w: enthalpy $w$
    :param phase: phase indicator
    :param theta_s: $\theta$ for symmetric phase, ahead of bubble (phase = 0)
    :param theta_b: $\theta$ for broken phase, behind bubble (phase = 1)
    :return: energy density
    """
    return w - get_p(w, phase, theta_s, theta_b)


@numba.njit
def get_w(
        e: th.FLOAT_OR_ARR,
        phase: th.INT_OR_ARR,
        theta_s: th.FLOAT_OR_ARR,
        theta_b: th.FLOAT_OR_ARR = 0.) -> th.FLOAT_OR_ARR:
    r"""
    Enthalpy $w$ as a function of energy density, assuming bag model.
    $\theta = \frac{e - 3p}{4}$ ("vacuum energy").
    Enthalpy and phase can be arrays of the same shape.
    Mentioned on page 23.

    :param e: energy density
    :param phase: phase indicator
    :param theta_s: $\theta$ for symmetric phase, ahead of bubble (phase = 0)
    :param theta_b: $\theta$ for broken phase, behind bubble (phase = 1)
    :return: enthalpy $w$
    """
    check_thetas(theta_s, theta_b)
    # Actually, theta is often known only from alpha_n and w, so should
    # think about an fsolve?
    theta = theta_b * phase + theta_s * (1.0 - phase)
    return 4/3 * (e - theta)


@numba.njit
def _get_phase_scalar(xi: float, v_w: float) -> int:
    return const.BROK_PHASE if xi < v_w else const.SYMM_PHASE


@numba.njit
def _get_phase_arr(xi: np.ndarray, v_w: float) -> np.ndarray:
    ph = np.zeros_like(xi)
    ph[np.where(xi < v_w)] = const.BROK_PHASE
    return ph


@numba.generated_jit(nopython=True)
def get_phase(xi: th.FLOAT_OR_ARR, v_w: float) -> th.FLOAT_OR_ARR:
    r"""
    Returns array indicating phase of system.
    in symmetric phase $(\xi > v_w)$, phase = 0
    in broken phase $(\xi < v_w)$, phase = 1

    :return: phase
    """
    if isinstance(xi, numba.types.Float):
        return _get_phase_scalar
    if isinstance(xi, numba.types.Array):
        if not xi.ndim:
            return _get_phase_scalar
        return _get_phase_arr
    if isinstance(xi, float):
        return _get_phase_scalar(xi, v_w)
    if isinstance(xi, np.ndarray):
        return _get_phase_arr(xi, v_w)
    raise TypeError(f"Unknown type for {type(xi)}")


@numba.njit
def adiabatic_index(
        w: th.FLOAT_OR_ARR,
        phase: th.INT_OR_ARR,
        theta_s: th.FLOAT_OR_ARR,
        theta_b: th.FLOAT_OR_ARR = 0.) -> th.FLOAT_OR_ARR:
    r"""
    Returns array of float, adiabatic index (ratio of enthalpy to energy).

    :param w: enthalpy $w$
    :param phase: phase indicator
    :param theta_s: $\theta$ for symmetric phase, ahead of bubble (phase = 0)
    :param theta_b: $\theta$ for broken phase, behind bubble (phase = 1)
    :return: adiabatic index
    """
    return w / get_e(w, phase, theta_s, theta_b)
