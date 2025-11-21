"""These are not relevant for the SSM"""

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, erfc, gamma

from pttools.ssm.low_k.utils import rho, Iv, U
from pttools.ssm.low_k.intersection import cross_z


def GW_spectral_density_approx_low(z, HLf, cs, tau_end):
    """Not relevant for the SSM, as it uses analytical approximation"""
    nu = (1 - 3 * cs ** 2) / (1 + 3 * cs ** 2)
    tau_star = (1. + nu) / HLf
    # HL = (1+nu)/tau_star
    # z = k*Lf
    if cs >= np.sqrt(1 / 3) - 1e-6:
        return (np.log(tau_star / tau_end)) ** 2 * 8 / 15 * Iv * tau_star
    else:
        return HLf ** (-1 + 2 * nu) * (0.5 * (1 + nu)) ** (-2 * nu) * z ** (-2 * nu) * (1 + nu) * gamma(
            0.5 + nu) ** 2 / 2 / np.pi * U(tau_star / tau_end, 2 * nu) ** 2 * 16 / 15 * Iv


def GW_spectral_density_approx_int(z, HLf, cs):
    nu = (1 - 3 * cs ** 2) / (1 + 3 * cs ** 2)
    tau_star = (1. + nu) / HLf
    return 4 / 3 / cs ** 4 * (3 - 2 * cs ** 2 - 3 / cs * (1 - cs ** 2) * np.arctanh(cs)) * Iv / tau_star / z ** 2


def GW_spectral_density_approx_high(z, HLf, cs, tau_end):
    nu = (1 - 3 * cs ** 2) / (1 + 3 * cs ** 2)
    tau_star = (1. + nu) / HLf
    delta = tau_end - tau_star
    xp = 0.5 * z * (1 + cs) / cs
    xm = 0.5 * z * (1 - cs) / cs
    integrand = lambda x, z, delta, tau_star, cs: (4 * np.pi * cs * z ** 3) ** (-1) * (
        rho(z, x, cs) *
        3 * np.pi / (2 * np.pi) ** 3 * (x / (2 * np.pi)) ** 2 / (1 + (x / (2 * np.pi)) ** 6) *
        3 * np.pi / (2 * np.pi) ** 3 * ((z / cs - x) / (2 * np.pi)) ** 2 / (1 + ((z / cs - x) / (2 * np.pi)) ** 6) *
        (1 + 2 * nu) ** (-1) * (1 - (1 + delta / tau_star) ** (-1 - 2 * nu)))
    integral = quad(integrand, xm, xp, args=(z, delta, tau_star, cs))[0]
    return integral


def Pgw_approx(z, HLf, cs, tau_star, tau_end):
    P_low = GW_spectral_density_approx_low(z, HLf, cs, tau_end)
    P_int = GW_spectral_density_approx_int(z, HLf, cs)
    P_high = GW_spectral_density_approx_high(z, HLf, cs, tau_end)

    nu = (1 - 3 * cs ** 2) / (1 + 3 * cs ** 2)
    z_star = 4 * cs * np.pi * (1 + nu) / HLf
    z_cross = cross_z(HLf, cs, tau_star, tau_end)

    term_low = 0.5 * erfc(2 * np.pi * tau_star * (z - z_cross)) * P_low
    term_int = 0.5 * (1 + erf(2 * np.pi * tau_star * (z - z_cross))) * P_int * 0.5 * erfc(
        2 * np.pi * tau_star * (z - 0.5 * z_star))

    return term_low + term_int + P_high
