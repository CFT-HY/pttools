"""Functions for computing the intersections of low and high frequency approximations."""

import numpy as np

from pttools.speedup import njit
from pttools.ssm.barotropic import Upsilon
from pttools.ssm.barotropic import tau_star as tau_star_func
from pttools.ssm.low_k.const import IV_ANALYTICAL, KP_ANALYTICAL
from pttools.ssm.low_k.integration import power_spectrum_integration_int, spec_den_gw_low
from pttools.ssm.low_k.kernel import kernel_int_bracket, kernel_low
from pttools.type_hints import FloatOrArr


@njit(nogil=True)
def z_cross(HLf: float, cs: float, nu: float, tau_star: float, tau_end: float) -> float:
    r"""
    Calculate the cross-over point z_cross where the low and intermediate frequency approximations are equal.

    $$z_\times = \left( \frac{\tilde{P}_\text{gw}^\text{int}(z=1)}{\tilde{P}_\text{gw}^\text{low}(z=1)}
    \right)^\frac{1}{2 - 2\nu},$$
    as $\tilde{P}_\text{gw}^\text{int} \propto z^{-2}$ and $\tilde{P}_\text{gw}^\text{low} \propto z^{-2\nu}$.
    Here $\tilde{P}_\text{gw}^\text{int}$ is :giombi_2024_cs:`\ ` eq. 3.11 with $\tau_*$, and
    $$\tilde{P}_\text{gw}^\text{low}(z=1) = \frac{32}{15} \tau_*^H \mathcal{I}_v
    \left( \frac{\tau_*^H}{2} \right)^{-2\nu} \frac{\Gamma^2 \left( \frac{1}{2} + \nu \right)}{4 \pi}
    \Upsilon_{2\nu}^2 \left( \frac{\tau_*}{\tau_\text{end}} \right)$$
    is :giombi_2026:`\ ` eq. 3.5a with $\tau_*^H = \frac{1 + \nu}{H L_f}$
    and the analytical $\mathcal{I}_v = \frac{1}{32\pi^2}$.
    If $\tau_* = \tau_*^H$, this is equal to :py:func:`z_cross_junction` with the same $\tau_*$.

    :param HLf: $H L_f = r_*$
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    :return: $z_\times$
    """
    # Todo: Is this only for testing? Is this fully replaced by z_cross_junction?
    int_term = power_spectrum_integration_int(z=1., cs=cs, tau_star=tau_star, Iv=IV_ANALYTICAL)
    tau_star_h = tau_star_func(r_star=HLf, nu=nu)
    # The kernel depends on tau_star and tau_end only through their ratio in Upsilon,
    # which is given by the tau_star and tau_end arguments.
    low_term = 32 * tau_star_h / 15 * IV_ANALYTICAL * kernel_low(
        z=1., nu=nu, tau_star=tau_star_h, tau_end=tau_star_h * tau_end / tau_star)
    return (int_term / low_term) ** (1 / (2 - 2 * nu))


@njit(nogil=True)
def z_cross_approx[T: FloatOrArr](cs: T, nu: float, eta_ratio: T | float, r_star: T | float) -> T:
    r"""Approximation for $z_\times$.

    $$z_\times = \frac{\sqrt{5}}{\sqrt{2} c_s^2} \frac{\nu}{1 + \nu}
    \frac{\sqrt{3 - 2 c_s^2 - \frac{3}{c_s}(1 - c_s^2) \text{arctanh}(c_s)}}
    {1 - \left( 1 + \frac{\Delta \eta_\text{v}}{\eta_*} \right)^{-\nu}} r_*$$

    This is derived by multiplying $R_*$ with
    $$k_\times = \frac{\sqrt{5}}{\sqrt{2} c_s^2} \frac{\nu}{1 + \nu}
    \frac{\sqrt{3 - 2 c_s^2 - \frac{3}{c_s}(1 - c_s^2) \text{arctanh}(c_s)}}
    {1 - \left( 1 + \frac{\Delta \eta_\text{v}}{\eta_*} \right)^{-\nu}} \mathcal{H}$$
    :giombi_2024_cs:`\ ` eq. 4.10
    Please note that this uses the low-frequency kernel of :giombi_2024_cs:`\ ` eq. 3.6,
    and not that of :py:func:`pttools.ssm.low_k.kernel.kernel_low`.
    Therefore, for $\nu > 0$ this differs from :py:func:`z_cross_junction`.

    :param cs: $c_s$, speed of sound
    :param nu: $\nu_\text{gdh2024}$, a scalar
    :param eta_ratio: $\frac{\Delta \eta_\text{v}}{\eta_*}$
    :param r_star: $r_* = \mathcal{H}_* R_*$
    :return: $z_\times$
    """
    return np.sqrt(5/2) / (cs**2 * (1 + nu)) * \
        np.sqrt(kernel_int_bracket(cs=cs)) / (-Upsilon(r=1 + eta_ratio, l=-nu)) * r_star


@njit(nogil=True)
def z_cross_junction(cs: float, nu: float, tau_star: float, tau_end: float, cs_friction: bool = False) -> float:
    r"""
    Calculate the cross-over point z_cross where the low and intermediate frequency approximations are equal.

    $$z_\times = \left( \frac{\tilde{P}_\text{gw}^\text{int}(z=1)}{\tilde{P}_\text{gw}^\text{low}(z=1)}
    \right)^\frac{1}{2 - 2\nu}
    = \left( \frac{5 \left[ 3 - 2c_s^2 - \frac{3}{c_s}(1 - c_s^2) \text{arctanh}(c_s) \right]}
    {8 c_s^4 \tau_*^2 \Delta_\text{low}(z=1)} \right)^\frac{1}{2 - 2\nu},$$
    as $\tilde{P}_\text{gw}^\text{int} \propto z^{-2}$ and $\tilde{P}_\text{gw}^\text{low} \propto z^{-2\nu}$.
    This is independent of $\mathcal{I}_v$, which is the same for both.

    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    :param cs_friction: whether to compute the acoustic friction from $c_s$ instead of $\nu$,
        see :py:func:`pttools.ssm.low_k.kernel.kernel_low`. Disabled by default.
    :return: $z_\times$
    """
    p_int = power_spectrum_integration_int(z=1., cs=cs, tau_star=tau_star, Iv=1.)
    p_low = spec_den_gw_low(z=1., cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end, Iv=1., cs_friction=cs_friction)
    return (p_int / p_low) ** (1 / (2 - 2 * nu))


@njit(cache=True)
def z_star_approx(cs: float, kp: float = KP_ANALYTICAL) -> float:
    r"""$z_\star$, approximate peak of the GW power spectrum.

    $$z_\star = k_{\star} R_* \simeq 2 c_s k_p R_*$$
    :giombi_2024_cs:`\ ` eq. 3.15.
    With the default $k_p R_* = 2\pi$ of the analytical $\tilde{P}_v$, $z_\star = 4 \pi c_s$.

    :param cs: $c_s$, speed of sound
    :param kp: $k_p R_*$, peak of $\tilde{P}_v$
    :return: $z_\star$
    """
    return 2 * cs * kp
