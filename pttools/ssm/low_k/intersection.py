"""Functions for computing the intersections of low and high frequency approximations"""

import numpy as np
from scipy.special import gamma

from pttools.ssm.barotropic import Upsilon
from pttools.ssm.const import CS0
from pttools.ssm.low_k.analytical import IV_ANALYTICAL
from pttools.ssm.low_k.integration import power_spectrum_integration_int
from pttools.ssm.low_k.kernel import kernel_int_bracket, kernel_low, kernel_low_bag
from pttools.type_hints import FloatOrArr


def z_cross(HLf: float, cs: float, nu: float, tau_star: float, tau_end: float) -> float:
    r"""
    Calculate the cross-over point z_cross where the low and intermediate frequency approximations are equal.

    :param HLf: $H L_f = r_*$
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    """
    # Todo: Is this only for testing? Is this fully replaced by z_cross_junction?
    int_term = power_spectrum_integration_int(z=1., cs=cs, tau_star=tau_star, Iv=IV_ANALYTICAL)
    low_term = HLf ** (-1 + 2 * nu) * (0.5 * (1 + nu)) ** (-2 * nu) * (1 + nu) * gamma(0.5 + nu) ** 2 / 2 / np.pi * \
               Upsilon(r=tau_star / tau_end, l=2 * nu) ** 2 * 16 / 15 * IV_ANALYTICAL

    return (int_term / low_term) ** (1 / (2 - 2 * nu))


def z_cross_approx(cs: FloatOrArr, nu: FloatOrArr, eta_ratio: FloatOrArr, r_star: FloatOrArr) -> FloatOrArr:
    r"""Approximation for $z_\times$

    $$z_\times = \frac{\sqrt{5}}{\sqrt{2} c_s^2} \frac{\nu}{1 + \nu}
    \frac{\sqrt{3 - 2 c_s^2 - \frac{3}{c_s}(1 - c_s^2) \text{arctanh}(c_s)}}
    {1 - \left( 1 + \frac{\Delta \eta_\text{v}}{\eta_*} \right)^{-\nu}} r_*$$

    This is derived by multiplying $R_*$ with
    $$k_\times = \frac{\sqrt{5}}{\sqrt{2} c_s^2} \frac{\nu}{1 + \nu}
    \frac{\sqrt{3 - 2 c_s^2 - \frac{3}{c_s}(1 - c_s^2) \text{arctanh}(c_s)}}
    {1 - \left( 1 + \frac{\Delta \eta_\text{v}}{\eta_*} \right)^{-\nu}} \mathcal{H}$$
    :giombi_2024_cs:`\ ` eq. 4.10
    """
    return np.sqrt(5/2) / (cs**2 * (1 + nu)) * \
        np.sqrt(kernel_int_bracket(cs=cs)) / (-Upsilon(r=1 + eta_ratio, l=-nu)) * r_star


def z_cross_junction(cs: float, nu: float, tau_star: float, tau_end: float) -> float:
    r"""
    Calculate the cross-over point z_cross where the low and intermediate frequency approximations are equal.

    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    """
    delta = kernel_low_bag(tau_star=tau_star, tau_end=tau_end) \
        if cs >= CS0 - 1e-6 \
        else kernel_low(z=1., nu=nu, tau_star=tau_star, tau_end=tau_end)

    return (5 * kernel_int_bracket(cs=cs) / (8 * cs**4 * tau_star ** 2 * delta)) ** (1 / (2 - 2 * nu))
