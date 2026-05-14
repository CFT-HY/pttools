"""Power spectrum integration functions"""

import numpy as np
from scipy.integrate import simpson

from pttools.speedup.functions import resample_log
from pttools.ssm.const import CS0
from pttools.ssm.low_k.kernel import kernel_int_bracket, kernel_low, kernel_low_bag
import pttools.type_hints as th

DEFAULT_KERNEL_NX: int = 1000


def Iv(x: th.FloatArr1D, P_tilde_v: th.FloatArr1D) -> float:
    r"""Source contribution $\mathcal{I}_v$
    $$\mathcal{I}_v \equiv \frac{1}{2\pi^2} \int_0^\infty dx x^2 \tilde{P}_v^2(x)$$
    :giombi_2024_cs:`\ ` eq. 3.7
    :giombi_2026:`\ ` eq. 3.4
    """
    return simpson(x ** 2 * P_tilde_v ** 2, x=x) / (2 * np.pi**2)


def Iv_resampled(x: th.FloatArr1D, P_tilde_v: th.FloatArr1D, nx: int = DEFAULT_KERNEL_NX) -> float:
    r"""Source contribution $\mathcal{I}_v$ with resampling"""
    x2 = resample_log(x=x, nx=nx)
    return Iv(x=x2, P_tilde_v=np.interp(x2, x, P_tilde_v))


def Jv(x: th.FloatArr1D, P_tilde_v: th.FloatArr1D) -> float:
    r"""$\mathcal{J}_v$
    $$\mathcal{J}_v \equiv \frac{1}{2\pi^2} \int_0^\infty dx \tilde{P}_v^2(x)$$
    :giombi_2026:`\ ` eq. 3.4
    """
    return simpson(P_tilde_v **2, x=x) / (2 * np.pi**2)


def Jv_resampled(x: th.FloatArr1D, P_tilde_v: th.FloatArr1D, nx: int = DEFAULT_KERNEL_NX) -> float:
    r"""$\mathcal{J}_v$ with resampling"""
    x2 = resample_log(x=x, nx=nx)
    return Jv(x=x2, P_tilde_v=np.interp(x2, x, P_tilde_v))


def power_spectrum_integration_low(
        x_data: th.FloatArr1D,
        Pv_data: th.FloatArr1D,
        z: th.FloatArr1D,
        cs: float,
        nu: float,
        tau_star: float,
        tau_end: float) -> th.FloatArr1D:
    r"""
    Calculate the low-frequency approximation (kR_* << 1) of the gravitational wave power spectrum.
    One dimensional integration over sound wave momentum.

    $$\tilde{P}_\text{gw}^\text{low} = \frac{16 \tau_*}{15 \pi^2}
    \int_0^\infty dx x^2 \tilde{P}_v^2(x) \Delta_\text{low}(x, \tau_*, \tau_\text{end})$$
    :giombi_2024_cs:`\ ` eq. 3.5
    :giombi_2026:`\ ` eq. 3.3

    :param x_data: momentum values (pR_*)
    :param Pv_data: power spectrum values at the given momenta
    :param z: array of gravitational wave momentum values (kR_*)
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    :return: gravitational wave power spectrum values at the given momentum
    """
    factor = 16 * tau_star / (15 * np.pi**2)  # Prefactor of eq. 3.5
    # Momentum values for integration (trapezoidal rule or simpson rule)
    x = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 1000)

    if cs >= CS0 - 1e-10:
        delta = kernel_low_bag(tau_star=tau_star, tau_end=tau_end)
        integrand = x**2 * np.interp(x, x_data, Pv_data)**2 * delta
        return factor * simpson(integrand, x=x)

    # z is multiplied to the final result instead of being included in the kernel,
    # so that only one integration is needed.
    delta = kernel_low(z=1., nu=nu, tau_star=tau_star, tau_end=tau_end)
    integrand = x ** 2 * np.interp(x, x_data, Pv_data) ** 2 * delta
    return factor * simpson(integrand, x=x) * z ** (-2 * nu)


def power_spectrum_integration_int(
        z: th.FloatOrArr,
        cs: th.FloatOrArr,
        tau_star: th.FloatOrArr,
        Iv: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Calculate the intermediate-frequency approximation (1 << k eta_* << kp eta_*) of the gravitational wave power spectrum.
    One dimensional integration over sound wave momentum.
    Note that this approximation does not depend on tau_end, as it assumes several gravitational wave oscillations
    during the acoustic sourcing (eta_end - eta_* >> eta_*)

    $$\tilde{P}_\text{gw}^\text{int}(kR_*) \approx_{\Delta \eta_\text{v} \gg \eta_*}
    \frac{4}{3 c_s^4} \left( 3 - 2c_s^2 - \frac{3}{c_s}(1 - c_s^2) \arctanh(c_s) \right)
    \frac{\mathcal{I}_v}{\tau_* z^2}$$
    :giombi_2024_cs:`\ ` eq. 3.11

    :param z: array of gravitational wave momentum values (kR_*)
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param Iv: Source contribution $I_v$ (resampled)
    :return: gravitational wave power spectrum values at the given momentum
    """
    return 4 / (3 * cs**4) * kernel_int_bracket(cs=cs) * Iv / (tau_star * z**2)


def power_spectrum_integration_high(
        x_data: th.FloatArr1D,
        Pv_data: th.FloatArr1D,
        z: th.FloatArr1D,
        cs: float = CS0) -> th.FloatArr1D:
    r"""Previously known as _peak

    :param x_data: array of momentum values (pR_*)
    :param Pv_data: array of power spectrum values at the given momentum
    :param z: array of gravitational wave momentum values (kR_*)
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :return: gravitational wave power spectrum values at the given momentum
    """
    # Todo: Replace this function with the use of the base PTtools machinery.
    Pgw = np.zeros_like(z)
    for i in range(len(Pgw)):
        factor = 1 / (4 * np.pi * z[i] * cs) * (1 - cs ** 2) ** 2 / cs ** 4
        xm = 0.5 * z[i] * (1 - cs) / cs
        xp = 0.5 * z[i] * (1 + cs) / cs
        x = np.logspace(np.log10(xm), np.log10(xp), 1000)
        integrand = (x - xp) ** 2 * (x - xm) ** 2 / x / (xp + xm - x) * np.interp(x, x_data, Pv_data) * np.interp(
            (xp + xm - x), x_data, Pv_data)
        Pgw[i] = factor * simpson(integrand, x=x)

    return Pgw
