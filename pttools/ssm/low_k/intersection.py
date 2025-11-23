"""Functions for computing the intersections of low and high frequency approximations"""

import numpy as np
from scipy.special import gamma

from pttools.ssm import const
from pttools.ssm.low_k.utils import Iv, U


def cross_z(HLf: float, cs: float, nu: float, tau_star: float, tau_end: float) -> float:
    r"""
    Calculate the cross-over point z_cross where the low and high frequency approximations are equal.

    :param HLf: $H L_f = r_*$
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    """
    int_term = 4 / 3 / cs ** 4 * (3 - 2 * cs ** 2 - 3 / cs * (1 - cs ** 2) * np.arctanh(cs)) * Iv / tau_star
    low_term = HLf ** (-1 + 2 * nu) * (0.5 * (1 + nu)) ** (-2 * nu) * (1 + nu) * gamma(0.5 + nu) ** 2 / 2 / np.pi * \
               U(tau_star / tau_end, 2 * nu) ** 2 * 16 / 15 * Iv

    return (int_term / low_term) ** (1 / (2 - 2 * nu))


def cross_z_junction(cs: float, nu: float, tau_star: float, tau_end: float) -> float:
    r"""
    Calculate the cross-over point z_cross where the low and high frequency approximations are equal.

    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    """
    if cs >= const.CS0 - 1e-6:  # if cs is close to 1/sqrt(3), use the radiation dominated kernel
        Delta_bar = 0.25 * (np.log(tau_end / tau_star)) ** 2
    else:  # if cs is not close to 1/sqrt(3), use the kernel function with cs^2 \neq 1/3
        Delta_bar = (0.5 * tau_star) ** (-2 * nu) * gamma(0.5 + nu) ** 2 / (4 * np.pi) * (
                1 - (tau_star / tau_end) ** (2 * nu)) ** 2 / (2 * nu) ** 2

    square_bracket = 3 - 2 * cs ** 2 - 3 / cs * (1 - cs ** 2) * np.arctanh(cs)

    return (5 / 8 / cs ** 4 * square_bracket / tau_star ** 2 / Delta_bar) ** (1 / (2 - 2 * nu))
