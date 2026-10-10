"""Power spectrum integration functions."""

import numpy as np
from scipy.integrate import simpson

from pttools.speedup import njit, resample_log
from pttools.ssm.const import CS0
from pttools.ssm.low_k.kernel import kernel_int_bracket, kernel_low
from pttools.ssm.spec_den_gw import spec_den_gw
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr

DEFAULT_KERNEL_NX: int = 1000


@njit
def Iv(x: th.FloatArr1D, P_tilde_v: th.FloatArr1D) -> float:
    r"""$\mathcal{I}_v$, source contribution.

    $$\mathcal{I}_v \equiv \frac{1}{2\pi^2} \int_0^\infty dx x^2 \tilde{P}_v^2(x)$$
    :giombi_2024_cs:`\ ` eq. 3.7
    :giombi_2026:`\ ` eq. 3.4.

    :param x: $x = pR_*$, which should cover the support of $\tilde{P}_v$
    :param P_tilde_v: $\tilde{P}_v(x)$
    :return: $\mathcal{I}_v$
    """
    return simpson(x ** 2 * P_tilde_v ** 2, x=x) / (2 * np.pi**2)


@njit
def Iv_resampled(x: th.FloatArr1D, P_tilde_v: th.FloatArr1D, nx: int = DEFAULT_KERNEL_NX) -> float:
    r"""Source contribution $\mathcal{I}_v$ with resampling.

    $$\mathcal{I}_v \equiv \frac{1}{2\pi^2} \int_{x_{\text{min}}}^{x_{\text{max}}} dx x^2 \tilde{P}_v^2(x)$$
    computed with $n_x$ logarithmically spaced points between the minimum and the maximum of the given $x$,
    where $\tilde{P}_v$ is interpolated linearly.

    :param x: $x = pR_*$, which should cover the support of $\tilde{P}_v$
    :param P_tilde_v: $\tilde{P}_v(x)$
    :param nx: $n_x$, number of points for the integration
    :return: $\mathcal{I}_v$
    """
    x2 = resample_log(x=x, nx=nx)
    return Iv(x=x2, P_tilde_v=np.interp(x2, x, P_tilde_v))


@njit
def Jv(x: th.FloatArr1D, P_tilde_v: th.FloatArr1D) -> float:
    r"""$\mathcal{J}_v$, source integral of $\tilde{P}_v^2$ without the $x^2$ factor of $\mathcal{I}_v$.

    $$\mathcal{J}_v \equiv \frac{1}{2\pi^2} \int_0^\infty dx \tilde{P}_v^2(x)$$
    :giombi_2026:`\ ` eq. 3.4.

    :param x: $x = pR_*$, which should cover the support of $\tilde{P}_v$
    :param P_tilde_v: $\tilde{P}_v(x)$
    :return: $\mathcal{J}_v$
    """
    return simpson(P_tilde_v ** 2, x=x) / (2 * np.pi**2)


@njit
def Jv_resampled(x: th.FloatArr1D, P_tilde_v: th.FloatArr1D, nx: int = DEFAULT_KERNEL_NX) -> float:
    r"""$\mathcal{J}_v$ with resampling.

    $$\mathcal{J}_v \equiv \frac{1}{2\pi^2} \int_{x_{\text{min}}}^{x_{\text{max}}} dx \tilde{P}_v^2(x)$$
    computed with $n_x$ logarithmically spaced points between the minimum and the maximum of the given $x$,
    where $\tilde{P}_v$ is interpolated linearly.

    :param x: $x = pR_*$, which should cover the support of $\tilde{P}_v$
    :param P_tilde_v: $\tilde{P}_v(x)$
    :param nx: $n_x$, number of points for the integration
    :return: $\mathcal{J}_v$
    """
    x2 = resample_log(x=x, nx=nx)
    return Jv(x=x2, P_tilde_v=np.interp(x2, x, P_tilde_v))


@njit
def spec_den_gw_low[T: FloatOrArr](
        z: T,
        cs: float,
        nu: float,
        tau_star: float,
        tau_end: float,
        Iv: float,
        cs_friction: bool = False) -> T:
    r"""$\tilde{P}_\text{gw}^\text{low}$, low-frequency GW spectral density for a given $\mathcal{I}_v$.

    As the kernel $\Delta_\text{low}$ does not depend on $x$,
    the integral of :giombi_2024_cs:`\ ` eq. 3.5 and :giombi_2026:`\ ` eq. 3.3 is
    $\int_0^\infty dx x^2 \tilde{P}_v^2(x) = 2 \pi^2 \mathcal{I}_v$, and
    $$\tilde{P}_\text{gw}^\text{low} = \frac{16 \tau_*}{15 \pi^2} 2 \pi^2 \mathcal{I}_v \Delta_\text{low}
    = \frac{32}{15} \tau_* \mathcal{I}_v \Delta_\text{low}(z, \tau_*, \tau_\text{end}),$$
    where $\Delta_\text{low}$ is given by :py:func:`pttools.ssm.low_k.kernel.kernel_low`.

    :param z: $z = kR_*$
    :param cs: $c_s$, speed of sound
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{R_*}$
    :param tau_end: $\tau_\text{end} = \frac{\eta_\text{end}}{R_*}$
    :param Iv: $\mathcal{I}_v$, source contribution
    :param cs_friction: whether to compute the acoustic friction from $c_s$ instead of $\nu$,
        see :py:func:`pttools.ssm.low_k.kernel.kernel_low`. Disabled by default.
    :return: $\tilde{P}_\text{gw}^\text{low}$
    """
    if cs_friction:
        kernel = kernel_low(z=z, nu=nu, tau_star=tau_star, tau_end=tau_end, cs=cs)
    else:
        kernel = kernel_low(z=z, nu=nu, tau_star=tau_star, tau_end=tau_end)
    return 32 * tau_star / 15 * Iv * kernel  # pyrefly: ignore[bad-return]


@njit
def power_spectrum_integration_low(
        x_data: th.FloatArr1D,
        Pv_data: th.FloatArr1D,
        z: th.FloatArr1D,
        cs: float,
        nu: float,
        tau_star: float,
        tau_end: float,
        cs_friction: bool = False) -> th.FloatArr1D:
    r"""
    Calculate the low-frequency approximation (kR_* << 1) of the gravitational wave power spectrum.

    One-dimensional integration over sound wave momentum.

    $$\tilde{P}_\text{gw}^\text{low} = \frac{16 \tau_*}{15 \pi^2}
    \int_0^\infty dx x^2 \tilde{P}_v^2(x) \Delta_\text{low}(x, \tau_*, \tau_\text{end})$$
    :giombi_2024_cs:`\ ` eq. 3.5
    :giombi_2026:`\ ` eq. 3.3
    This is computed with :py:func:`spec_den_gw_low` and :py:func:`Iv_resampled`.

    :param x_data: momentum values (pR_*), which should cover the support of Pv_data
    :param Pv_data: power spectrum values at the given momenta
    :param z: array of gravitational wave momentum values (kR_*)
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    :param cs_friction: whether to compute the acoustic friction from $c_s$ instead of $\nu$,
        see :py:func:`pttools.ssm.low_k.kernel.kernel_low`. Disabled by default.
    :return: gravitational wave power spectrum values at the given momentum
    """
    return spec_den_gw_low(
        z=z, cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end,
        Iv=Iv_resampled(x=x_data, P_tilde_v=Pv_data), cs_friction=cs_friction
    )


@njit
def power_spectrum_integration_int[T: FloatOrArr](
        z: T,
        cs: T | float,
        tau_star: T | float,
        Iv: T | float) -> T:
    r"""
    Calculate the intermediate-frequency approximation of the gravitational wave power spectrum.

    This approximation is valid for 1 << k eta_* << kp eta_*.
    One dimensional integration over sound wave momentum.
    Note that this approximation does not depend on tau_end, as it assumes several gravitational wave oscillations
    during the acoustic sourcing (eta_end - eta_* >> eta_*).

    $$\tilde{P}_\text{gw}^\text{int}(kR_*) \approx_{\Delta \eta_\text{v} \gg \eta_*}
    \frac{4}{3 c_s^4} \left( 3 - 2c_s^2 - \frac{3}{c_s}(1 - c_s^2) \text{arctanh}(c_s) \right)
    \frac{\mathcal{I}_v}{\tau_* z^2}$$
    :giombi_2024_cs:`\ ` eq. 3.11

    :param z: array of gravitational wave momentum values (kR_*)
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param Iv: Source contribution $I_v$ (resampled)
    :return: gravitational wave power spectrum values at the given momentum
    """
    return 4 / (3 * cs**4) * kernel_int_bracket(cs=cs) * Iv / (tau_star * z**2)  # pyrefly: ignore[bad-return]


@njit
def power_spectrum_integration_high(
        x_data: th.FloatArr1D,
        Pv_data: th.FloatArr1D,
        z: th.FloatArr1D,
        cs: float = CS0,
        nx: int = DEFAULT_KERNEL_NX,
        parallel: bool = True) -> th.FloatArr1D:
    r"""$\tilde{P}_\text{gw}^\text{high}$, high-frequency approximation of the GW spectral density.

    $$\tilde{P}_\text{gw}^\text{high}(z) = \frac{1}{4\pi z c_s} \left( \frac{1 - c_s^2}{c_s^2} \right)^2
    \int_{x_{-}}^{x_{+}} \frac{dx}{x} \frac{(x - x_{+})^2 (x - x_{-})^2}{x_{+} + x_{-} - x}
    \tilde{P}_v(x) \tilde{P}_v(x_{+} + x_{-} - x)$$
    :giombi_2024_cs:`\ ` eq. 3.13 without the source lifetime factor $\Upsilon_{1+2\nu}$.
    This is the SSM spectral density of :py:func:`pttools.ssm.spec_den_gw.spec_den_gw`
    with the source lifetime factor set to unity.
    Previously known as _peak.

    :param x_data: array of momentum values (pR_*), which should cover $[x_{-}(z_{\text{min}}), x_{+}(z_{\text{max}})]$
    :param Pv_data: array of power spectrum values at the given momentum
    :param z: array of gravitational wave momentum values (kR_*)
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nx: number of points for the integration over $x$
    :param parallel: whether to run with multiple threads
    :return: gravitational wave power spectrum values at the given momentum
    """
    return spec_den_gw(
        z_lookup=x_data, P_tilde_v_lookup=Pv_data, y=z, cs=cs, nx_P_tilde_gw=nx, parallel=parallel
    )[0]
