import numpy as np
from scipy.special import gamma

from pttools.ssmtools.low_k.utils import parse_params_gw, Iv, U


def cross_z(HLf, cs, tau_star, tau_end):
    """
    Calculate the cross-over point z_cross where the low and high frequency approximations are equal.
    """
    nu = (1 - 3 * cs ** 2) / (1 + 3 * cs ** 2)
    int_term = 4 / 3 / cs ** 4 * (3 - 2 * cs ** 2 - 3 / cs * (1 - cs ** 2) * np.arctanh(cs)) * Iv / tau_star
    low_term = HLf ** (-1 + 2 * nu) * (0.5 * (1 + nu)) ** (-2 * nu) * (1 + nu) * gamma(0.5 + nu) ** 2 / 2 / np.pi * U(
        tau_star / tau_end, 2 * nu) ** 2 * 16 / 15 * Iv

    z_cross = (int_term / low_term) ** (1 / (2 - 2 * nu))
    return z_cross


def cross_z_junction(params_gw):
    """
    Calculate the cross-over point z_cross where the low and high frequency approximations are equal.
    """
    cs, tau_star, tau_end = parse_params_gw(params_gw)  # unpack parameters for gravitational wave power spectrum
    nu = (1 - 3 * cs ** 2) / (1 + 3 * cs ** 2)

    if cs >= np.sqrt(1 / 3) - 1e-6:  # if cs is close to 1/sqrt(3), use the radiation dominated kernel
        Delta_bar = 0.25 * (np.log(tau_end / tau_star)) ** 2
    else:  # if cs is not close to 1/sqrt(3), use the kernel function with cs^2 \neq 1/3
        Delta_bar = (0.5 * tau_star) ** (-2 * nu) * gamma(0.5 + nu) ** 2 / (4 * np.pi) * (
                1 - (tau_star / tau_end) ** (2 * nu)) ** 2 / (2 * nu) ** 2

    square_bracket = 3 - 2 * cs ** 2 - 3 / cs * (1 - cs ** 2) * np.arctanh(cs)

    z_cross = (5 / 8 / cs ** 4 * square_bracket / tau_star ** 2 / Delta_bar) ** (1 / (2 - 2 * nu))
    return z_cross
