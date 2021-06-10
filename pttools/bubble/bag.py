"""Some functions useful for the bag equation of state"""

import typing as tp

import numpy as np

import pttools.type_hints as th
from . import const

CS2_FUN_TYPE = tp.Callable[[th.FLOAT_OR_ARR], float]

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


def cs2_bag(w: th.FLOAT_OR_ARR) -> float:
    """
    Speed of sound squared in Bag model, equal to 1/3 independent of enthalpy $w$
    """
    if isinstance(w, np.ndarray):
        cs2 = const.cs0_2 * np.ones_like(w)
    else:
        cs2 = const.cs0_2

    return cs2


def theta_bag(w: th.FLOAT_OR_ARR, phase: th.FLOAT_OR_ARR, alpha_n: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Trace anomaly $\theta = (e - 3p)/4$ in Bag model.
    """
    if isinstance(w, np.ndarray):
        w_n = w[-1]
    else:
        w_n = w
    return alpha_n * (0.75 * w_n) * (1 - phase)


def p(
        w: th.FLOAT_OR_ARR,
        phase: th.FLOAT_OR_ARR,
        theta_s: th.FLOAT_OR_ARR,
        theta_b: th.FLOAT_OR_ARR = 0.) -> th.FLOAT_OR_ARR:
    """
     Pressure as a function of enthalpy, assuming bag model.
     phase: phase indicator (see below).
     theta = (e - 3p)/4 (trace anomaly or "vacuum energy")
     _s = symmetric phase, ahead of bubble (phase = 0)
     _b = broken phase, behind bubble (phase = 1)
     enthalpy, theta and phase can be arrays (same shape)
    """
    theta = theta_b * phase + theta_s * (1.0 - phase)
    return 0.25 * w - theta


def e(
        w: th.FLOAT_OR_ARR,
        phase: th.FLOAT_OR_ARR,
        theta_s: th.FLOAT_OR_ARR,
        theta_b: th.FLOAT_OR_ARR = 0.) -> th.FLOAT_OR_ARR:
    """
     Energy density as a function of enthalpy, assuming bag model.
     theta = (e - 3p)/4 ("vacuum energy")
     _s = symmetric phase, ahead of bubble (phase = 0)
     _b = broken phase, behind bubble (phase = 1)
     enthalpy and phase can be arrays (same shape)
    """
    return w - p(w, phase, theta_s, theta_b)


def w(
        e: th.FLOAT_OR_ARR,
        phase: th.FLOAT_OR_ARR,
        theta_s: th.FLOAT_OR_ARR,
        theta_b: th.FLOAT_OR_ARR = 0.) -> th.FLOAT_OR_ARR:
    """
     Enthalpy as a function of energy density, assuming bag model.
     theta = (e - 3p)/4 ("vacuum energy")
     _s = symmetric phase, ahead of bubble (phase = 0)
     _b = broken phase, behind bubble (phase = 1)
     enthalpy and phase can be arrays (same shape)
    """
    #     Actually, theta is often known only from alpha_n and w, so should
    #     think about an fsolve?
    theta = theta_b * phase + theta_s * (1.0 - phase)
    return (4 / 3) * (e - theta)


def phase(xi: th.FLOAT_OR_ARR, v_w: float) -> th.FLOAT_OR_ARR:
    """
     Returns array indicating phase of system.
     in symmetric phase (xi>v_w), phase = 0
     in broken phase (xi<v_w), phase = 1
    """
    if isinstance(xi, np.ndarray):
        ph = np.zeros_like(xi)
        ph[np.where(xi < v_w)] = const.brok_phase
    else:
        ph = const.symm_phase
        if xi < v_w:
            ph = const.brok_phase

    return ph


def adiabatic_index(
        w: th.FLOAT_OR_ARR,
        phase: th.FLOAT_OR_ARR,
        theta_s: th.FLOAT_OR_ARR,
        theta_b: th.FLOAT_OR_ARR = 0.) -> th.FLOAT_OR_ARR:
    """
    Returns array of float, adiabatic index (ratio of enthalpy to energy).
    """

    return w / e(w, phase, theta_s, theta_b)
