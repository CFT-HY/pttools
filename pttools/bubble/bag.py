r"""Functions for the bag equation of state.

See page 37 of :notes:`\ `.
"""

import logging

import numba
from numba.extending import overload
import numpy as np

import pttools.type_hints as th
from .boundary import Phase
from . import const

logger = logging.getLogger(__name__)


@numba.njit
def adiabatic_index_bag(
        w: th.FloatOrArr,
        phase: th.IntOrArr,
        theta_s: th.FloatOrArr,
        theta_b: th.FloatOrArr = 0.) -> th.FloatOrArr:
    r"""
    Returns array of float, adiabatic index (ratio of enthalpy to energy).

    :param w: enthalpy $w$
    :param phase: phase indicator
    :param theta_s: $\theta$ for symmetric phase, ahead of bubble (phase = 0)
    :param theta_b: $\theta$ for broken phase, behind bubble (phase = 1)
    :return: adiabatic index
    """
    return w / e_bag(w, phase, theta_s, theta_b)


def check_thetas(theta_s: th.FloatOrArr, theta_b: th.FloatOrArr) -> None:
    r"""Check that $\theta_s \leq \theta_b$.

    :param theta_s: $\theta_s$
    :param theta_b: $\theta_b$
    """
    if np.any(theta_b > theta_s):
        _check_thetas_warning(theta_s, theta_b)


@overload(check_thetas, jit_options={"nopython": True})
def _check_thetas_numba(theta_s: th.FloatOrArr, theta_b: th.FloatOrArr):
    if isinstance(theta_s, numba.types.Array) or isinstance(theta_b, numba.types.Array):
        return check_thetas
    else:
        return _check_thetas_scalar


def _check_thetas_scalar(theta_s: th.FloatOrArr, theta_b: th.FloatOrArr) -> None:
    """This is a workaround for a bug in Numba 0.60.0.
    This fix was not needed for Numba 0.59.0.
    https://github.com/numba/numba/issues/8270
    """
    if theta_b > theta_s:
        _check_thetas_warning(theta_s, theta_b)


@numba.njit
def _check_thetas_warning(theta_s: th.FloatOrArr, theta_b: th.FloatOrArr) -> None:
    with numba.objmode:
        logger.warning(
            "theta_b should always be smaller than theta_s, "
            "but got theta_s=%s, theta_b=%s", theta_s, theta_b)


@numba.njit
# pylint: disable=unused-argument
def cs2_bag_scalar(w: float, phase: Phase) -> float:
    """The scalar versions of the bag functions have to be compiled to cfuncs if jitting is disabled,
    as otherwise the cfunc version of the differential cannot be created.
    """
    return const.CS0_2


@numba.cfunc(th.CS2FunScalarSig)
# pylint: disable=unused-argument
def cs2_bag_scalar_cfunc(w: float, phase: Phase) -> float:
    return const.CS0_2


CS2_BAG_SCALAR_PTR = cs2_bag_scalar_cfunc.address
CS2ScalarCType = cs2_bag_scalar_cfunc.ctypes


@numba.njit
# pylint: disable=unused-argument
def cs2_bag_arr(w: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return const.CS0_2 * np.ones_like(w)


@numba.njit
def cs2_bag(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Speed of sound squared in Bag model, equal to $\frac{1}{3}$ independent of enthalpy $w$.

    :param w: enthalpy $w$
    :param phase: phase $\phi$
    :return: speed of sound squared $c_s^2$
    """
    if isinstance(w, float):
        return cs2_bag_scalar(w, phase)
    if isinstance(w, np.ndarray):
        return cs2_bag_arr(w, phase)
    raise TypeError(f"Unknown type for w: {type(w)}")


@numba.njit
def e_bag(
        w: th.FloatOrArr,
        phase: th.IntOrArr,
        theta_s: th.FloatOrArr,
        theta_b: th.FloatOrArr = 0.) -> th.FloatOrArr:
    r"""
    Energy density $e$ as a function of enthalpy $w$, assuming bag model.
    $\theta = \frac{e - 3p}{4}$ ("vacuum energy").
    Enthalpy and phase can be arrays of the same shape.
    See also the equation 4.10.

    :param w: enthalpy $w$
    :param phase: phase indicator
    :param theta_s: $\theta$ for symmetric phase, ahead of bubble (phase = 0)
    :param theta_b: $\theta$ for broken phase, behind bubble (phase = 1)
    :return: energy density $e$
    """
    return w - p_bag(w, phase, theta_s, theta_b)


@numba.njit
def p_bag(
        w: th.FloatOrArr,
        phase: th.IntOrArr,
        theta_s: th.FloatOrArr,
        theta_b: th.FloatOrArr = 0.) -> th.FloatOrArr:
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
def w_bag(
        e: th.FloatOrArr,
        phase: th.IntOrArr,
        theta_s: th.FloatOrArr,
        theta_b: th.FloatOrArr = 0.) -> th.FloatOrArr:
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


# def junction_bag(v1: th.FloatOrArr, w1: th.FloatOrArr, V1: th.FloatOrArr, V2: th.FloatOrArr, greater_branch: bool) -> tp.Tuple[th.FloatOrArr, th.FloatOrArr]:
#     v2 = v2_tilde_bag(v1, w1, V1, V2, greater_branch)
#     return v2, boundary.w2_junction(v1, w1, v2)
#
#
# def v2_tilde_bag(v1: th.FloatOrArr, w1: th.FloatOrArr, V1: th.FloatOrArr, V2: th.FloatOrArr, greater_branch: bool) -> th.FloatOrArr:
#     """This doesn't seem to work properly for the greater branch."""
#     a = v1 + ((V2 - V1)/w1 + 1/4) / (relativity.gamma2(v1) * v1)
#     sign = 1 if greater_branch else -1
#     ret = (2*a + sign * np.sqrt(4*a**2 - 3)) / 3
#     if np.any(ret < 0) and not greater_branch:
#         logger.error(f"Got v2_tilde={ret} with the lesser branch. Should you select the greater branch?")
#     if np.any(ret > 1) and greater_branch:
#         logger.error(f"Got v2_tilde={ret} with the greater branch. Should you select the lesser branch?")
#     return ret


def theta_bag(w: th.FloatOrArr, phase: th.IntOrArr, alpha_n: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Trace anomaly $\theta = \frac{1}{4} (e - 3p)$ in the Bag model.
    Equation 7.24 in the lecture notes, equation 2.10 in the article.

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
