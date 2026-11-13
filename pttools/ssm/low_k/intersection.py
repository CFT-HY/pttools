r"""Intersections of the low-, intermediate- and high-frequency approximations, and the peak scales.

There are five different scales, which are named as follows.

- ``kp_sw``, ``zp_sw``: $k_p = \frac{2\pi}{R_*}$ and $k_p R_* = 2\pi$, the sound wave scale of
  :giombi_2024_cs:`\ ` and :giombi_2026:`\ ` (:py:data:`pttools.ssm.low_k.const.ZP_SW`)
- ``xp_v``: $x$ at the maximum of $x \tilde{P}_v(x)$ of a given velocity spectral density (:py:func:`xp_v`).
  For the analytical $\tilde{P}_v$ this is $k_p R_*$.
- ``kp_gw``, ``zp_gw``: $k_\star$ and $k_\star R_*$, the peak of the GW power spectrum, which the articles
  estimate as $k_\star \simeq 2 c_s k_p$ (:py:func:`zp_gw_approx`)
- ``z_cross``: $z_\times$, where $\tilde{P}_\text{gw}^\text{low} = \tilde{P}_\text{gw}^\text{int}$
- ``z_int_high``: $z_{\text{int,high}}$, where $\tilde{P}_\text{gw}^\text{int} = \tilde{P}_\text{gw}^\text{high}$
  (:py:func:`z_int_high`). This is below the peak of the GW power spectrum.
"""

import numpy as np

from pttools.speedup import njit
from pttools.ssm.barotropic import Upsilon
from pttools.ssm.barotropic import tau_star as tau_star_func
from pttools.ssm.low_k.const import IV_ANALYTICAL, ZP_SW
from pttools.ssm.low_k.integration import power_spectrum_integration_int, spec_den_gw_low
from pttools.ssm.low_k.kernel import kernel_int_bracket, kernel_low
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr

__all__ = [
    "xp_v",
    "z_cross",
    "z_cross_approx",
    "z_cross_junction",
    "z_int_high",
    "z_int_high_approx",
    "zp_gw_approx",
]


@njit(cache=True)
def xp_v(x: th.FloatArr1D, P_tilde_v: th.FloatArr1D) -> float:
    r"""$x_{p,v}$, $x$ at the maximum of $x \tilde{P}_v(x)$.

    $$x_{p,v} = \underset{x}{\operatorname{arg\,max}} \; x \tilde{P}_v(x)$$
    This is a dimensionless wavenumber $x = pR_*$, and not the value of $x \tilde{P}_v(x)$ at the maximum.
    It is the wavenumber of the sound waves that have the most weight in the resonance integral of
    $\tilde{P}_\text{gw}^\text{high}$, which can be written as an integral of
    $x \tilde{P}_v(x) \, y \tilde{P}_v(y)$ with $z = c_s (x + y)$ times a scale-invariant geometric factor
    (:giombi_2024_cs:`\ ` sec. 3.3, the footnote of eq. 3.15).
    It is thus the generalisation of the sound wave scale $k_p R_*$ to an arbitrary $\tilde{P}_v$.
    It is not $x$ at the maximum of $\tilde{P}_v(x)$ or of the velocity power spectrum
    $\propto x^3 \tilde{P}_v(x)$, which are at $0.89 k_p R_*$ and $1.31 k_p R_*$ for the analytical $\tilde{P}_v$.
    For the analytical $\tilde{P}_v$ (:py:func:`pttools.ssm.low_k.analytical.Pv_analytical`)
    this is exactly the sound wave scale $k_p R_* = 2\pi$ (:py:data:`pttools.ssm.low_k.const.ZP_SW`).
    For the $\tilde{P}_v$ of the Sound Shell Model it is $x_{p,v} \approx 1.1 - 1.5$ instead.
    The value is the point of the $x$ array with the largest $x \tilde{P}_v(x)$, without interpolation.
    If this is the first or the last point, the $x$ array does not cover the maximum, and the result is nan.
    This is the case e.g. for the lookup array of a spectrum, the $y$ array of which ends below $y \approx 1$.

    :param x: $x = pR_*$, which should cover the maximum
    :param P_tilde_v: $\tilde{P}_v(x)$
    :return: $x_{p,v}$, or nan if the $x$ array does not cover the maximum
    """
    i = np.argmax(x * P_tilde_v)
    if i == 0 or i == x.size - 1:
        return np.nan
    return x[i]


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
def z_int_high(z: th.FloatArr1D, Pgw_int: th.FloatArr1D, Pgw_high: th.FloatArr1D) -> float:
    r"""$z_{\text{int,high}}$, the crossing of the intermediate- and high-frequency regimes.

    This is the first point of the $z$ array where
    $\tilde{P}_\text{gw}^\text{high} > \tilde{P}_\text{gw}^\text{int}$.
    It therefore depends on the points of the $z$ array.
    This is not the peak of the GW power spectrum $k_\star R_*$ (:py:func:`zp_gw_approx`), but below it,
    as $\tilde{P}_\text{gw}^\text{high}$ exceeds $\tilde{P}_\text{gw}^\text{int} \propto z^{-2}$
    already on the rising side of the peak.

    :param z: $z = kR_*$
    :param Pgw_int: $\tilde{P}_\text{gw}^\text{int}$
    :param Pgw_high: $\tilde{P}_\text{gw}^\text{high}$
    :return: $z_{\text{int,high}}$, or nan if $\tilde{P}_\text{gw}^\text{high} \leq \tilde{P}_\text{gw}^\text{int}$
        at all the points, or if the arrays are empty
    """
    above = Pgw_high > Pgw_int
    if not above.size:
        return np.nan
    # argmax gives the first True, or 0 if there is none
    i = np.argmax(above)
    if not above[i]:
        return np.nan
    return z[i]


@njit(cache=True)
def z_int_high_approx(cs: float, xp_v: float = ZP_SW) -> float:
    r"""Approximation for $z_{\text{int,high}}$, the crossing of the intermediate- and high-frequency regimes.

    $$z_{\text{int,high}} \approx c_s x_{p,v},$$
    where $x_{p,v}$ is $x$ at the maximum of $x \tilde{P}_v(x)$ (:py:func:`xp_v`).
    This is an empirical relation: in the tested cases the crossing is at $(0.9 - 1.3) c_s x_{p,v}$ both for the
    analytical $\tilde{P}_v$, for which $x_{p,v} = k_p R_* = 2\pi$, and for the $\tilde{P}_v$ of the Sound Shell
    Model, for which $x_{p,v} \approx 1.1 - 1.5$.
    It depends weakly on $r_*$ and the source duration, as $\tilde{P}_\text{gw}^\text{int} \propto \tau_*^{-1}$
    and $\tilde{P}_\text{gw}^\text{high} \propto \Upsilon_{1+2\nu}$.
    For the analytical $\tilde{P}_v$ this is half of the estimate of the peak of the GW power spectrum
    $k_\star R_* \simeq 2 c_s k_p R_*$ (:py:func:`zp_gw_approx`).

    :param cs: $c_s$, speed of sound
    :param xp_v: $x_{p,v}$, $x$ at the maximum of $x \tilde{P}_v(x)$.
        The default is that of the analytical $\tilde{P}_v$.
    :return: approximate $z_{\text{int,high}}$
    """
    return cs * xp_v


@njit(cache=True)
def zp_gw_approx(cs: float, zp_sw: float = ZP_SW) -> float:
    r"""Approximation for $k_\star R_*$, the peak of the GW power spectrum.

    $$k_\star R_* \simeq 2 c_s k_p R_*$$
    :giombi_2024_cs:`\ ` eq. 3.15, :giombi_2026:`\ ` below eq. 3.11.
    With $k_p R_* = 2\pi$ (:py:data:`pttools.ssm.low_k.const.ZP_SW`) this is $4 \pi c_s$.
    In the code $k_\star$ is called ``kp_gw`` and $k_\star R_*$ ``zp_gw``,
    as the subscript $\star$ could be confused with the subscript $*$ of e.g. $R_*$ and $\tau_*$ (``tau_star``),
    which denotes the start of the acoustic phase.
    The estimate comes from the resonance $k = c_s (p + q)$ of the GW and the sound waves with
    $p = q = k_p$, and is valid only for the analytical $\tilde{P}_v$
    (:py:func:`pttools.ssm.low_k.analytical.Pv_analytical`),
    for which the maximum of the GW power spectrum is $23 - 30 \%$ above it.
    For the Sound Shell Model the peak of the GW power spectrum is affected by the thickness of the sound shell,
    and this is not a valid estimate even when $k_p R_*$ is replaced by $x$ at the maximum of
    $x \tilde{P}_v(x)$.
    This is also not the crossing of the intermediate- and high-frequency regimes (:py:func:`z_int_high`).

    :param cs: $c_s$, speed of sound
    :param zp_sw: $k_p R_*$, the sound wave scale
    :return: approximate $k_\star R_*$
    """
    return 2 * cs * zp_sw
