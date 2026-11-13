r"""Analytical models for a simplified $P_v(p)$."""

import numpy as np
from scipy.integrate import quad

from pttools.bubble.thermo import nu_gdh2024
from pttools.speedup import njit
from pttools.ssm.barotropic import Upsilon
from pttools.ssm.barotropic import tau_star as tau_star_func
from pttools.ssm.low_k.const import IV_ANALYTICAL, ZP_SW
from pttools.ssm.low_k.integration import power_spectrum_integration_int, spec_den_gw_low
from pttools.ssm.low_k.intersection import z_cross, z_int_high_approx
from pttools.ssm.low_k.join import join_regimes
from pttools.ssm.rho import rho_pm, x_minus, x_plus
from pttools.type_hints import FloatOrArr

__all__ = [
    "Pgw_approx",
    "Pv_analytical",
    "gw_spectral_density_approx_high",
    "gw_spectral_density_approx_high_integrand",
    "gw_spectral_density_approx_int",
    "gw_spectral_density_approx_low",
]


@njit(nogil=True)
def gw_spectral_density_approx_low[T: FloatOrArr](z: T, HLf: float, cs: float, tau_end: float) -> T:
    r"""$\tilde{P}_\text{gw}^\text{low}$ for the analytical $\tilde{P}_v$.

    $$\tilde{P}_\text{gw}^\text{low} = (H L_f)^{-1 + 2\nu} \left( \frac{1 + \nu}{2} \right)^{-2\nu} z^{-2\nu}
    (1 + \nu) \frac{\Gamma^2 \left( \frac{1}{2} + \nu \right)}{2\pi} \frac{16}{15} \mathcal{I}_v
    \Upsilon_{2\nu}^2 \left( \frac{\tau_*}{\tau_\text{end}} \right)
    = \frac{32}{15} \tau_* \mathcal{I}_v \Delta_\text{low},$$
    :giombi_2026:`\ ` eq. 3.5a divided by the prefactor of eq. 2.40,
    with $\mathcal{I}_v = \frac{1}{32\pi^2}$, $\tau_* = \frac{1 + \nu}{H L_f}$
    and $\nu = \frac{1 - 3c_s^2}{1 + 3c_s^2}$.
    For $c_s = \frac{1}{\sqrt{3}}$ this is
    $\frac{8}{15} \tau_* \mathcal{I}_v \ln^2 \left( \frac{\tau_\text{end}}{\tau_*} \right)$
    (:giombi_2024_cs:`\ ` eq. 3.6 with $\nu \rightarrow 0$).
    Not relevant for the SSM, as it uses analytical approximation.

    :param z: $z = kR_*$
    :param HLf: $H L_f = r_*$
    :param cs: $c_s$, speed of sound
    :param tau_end: $\tau_\text{end} = \frac{\eta_\text{end}}{R_*}$
    :return: $\tilde{P}_\text{gw}^\text{low}$
    """
    nu = nu_gdh2024(cs**2)
    return spec_den_gw_low(
        z=z, cs=cs, nu=nu, tau_star=tau_star_func(r_star=HLf, nu=nu), tau_end=tau_end, Iv=IV_ANALYTICAL)


@njit(nogil=True)
def gw_spectral_density_approx_int[T: FloatOrArr](z: T, HLf: float, cs: float) -> T:
    r"""GW spectral density approximation for intermediate frequencies.

    $$\tilde{P}_\text{gw}^\text{int} =
    \frac{4}{3 c_s^4} \left( 3 - 2c_s^2 - \frac{3}{c_s}(1 - c_s^2) \text{arctanh}(c_s) \right)
    \frac{\mathcal{I}_v}{\tau_* z^2}$$
    :giombi_2024_cs:`\ ` eq. 3.11,
    with $\mathcal{I}_v = \frac{1}{32\pi^2}$, $\tau_* = \frac{1 + \nu}{H L_f}$
    and $\nu = \frac{1 - 3c_s^2}{1 + 3c_s^2}$.

    :param z: $z = kR_*$
    :param HLf: $H L_f = r_*$
    :param cs: $c_s$, speed of sound
    :return: $\tilde{P}_\text{gw}^\text{int}$
    """
    tau_star = tau_star_func(r_star=HLf, nu=nu_gdh2024(cs**2))
    return power_spectrum_integration_int(z=z, cs=cs, tau_star=tau_star, Iv=IV_ANALYTICAL)


@njit(nogil=True)
def gw_spectral_density_approx_high_integrand(x: float, z: float, cs: float, upsilon: float) -> float:
    r"""Integrand of :py:func:`gw_spectral_density_approx_high`.

    $$\frac{\rho(z, x, y)}{4\pi c_s z^3} \tilde{P}_v(x) \tilde{P}_v(y) \Upsilon_{1+2\nu},
    \quad y = x_{+} + x_{-} - x,$$
    where $\tilde{P}_v$ is the analytical ansatz :py:func:`Pv_analytical` with $k_p R_* = 2\pi$ and $\bar{U}_f = 1$.
    As $\rho(z, x, y) = z^2 \left( \frac{1 - c_s^2}{c_s^2} \right)^2 \frac{(x - x_{+})^2 (x - x_{-})^2}{x y}$
    (:giombi_2024_cs:`\ ` eq. B.21),
    this is the integrand of :giombi_2024_cs:`\ ` eq. 3.13 including the prefactor.

    :param x: $x = pR_*$
    :param z: $z = kR_*$
    :param cs: $c_s$, speed of sound
    :param upsilon: $\Upsilon_{1+2\nu}$, source lifetime factor
    :return: the integrand
    """
    y = x_plus(z=z, cs=cs) + x_minus(z=z, cs=cs) - x
    return rho_pm(z=z, x=x, cs=cs) / (4 * np.pi * cs * z ** 3) \
        * Pv_analytical(k=x, kp_sw=ZP_SW, ubarf2=1.) \
        * Pv_analytical(k=y, kp_sw=ZP_SW, ubarf2=1.) \
        * upsilon


def gw_spectral_density_approx_high(z: float, HLf: float, cs: float, tau_end: float) -> float:
    r"""GW spectral density approximation for high frequencies.

    $$\tilde{P}_\text{gw}^\text{high} = \frac{1}{4\pi z c_s} \left( \frac{1 - c_s^2}{c_s^2} \right)^2
    \Upsilon_{1+2\nu} \left( \frac{\tau_*}{\tau_\text{end}} \right)
    \int_{x_{-}}^{x_{+}} \frac{dx}{x} \frac{(x - x_{+})^2 (x - x_{-})^2}{x_{+} + x_{-} - x}
    \tilde{P}_v(x) \tilde{P}_v(x_{+} + x_{-} - x)$$
    :giombi_2024_cs:`\ ` eq. 3.13,
    with the analytical $\tilde{P}_v$, $\tau_* = \frac{1 + \nu}{H L_f}$ and $\nu = \frac{1 - 3c_s^2}{1 + 3c_s^2}$.
    The integral is computed with :external:py:func:`scipy.integrate.quad`, and therefore this cannot be jitted.

    :param z: $z = kR_*$
    :param HLf: $H L_f = r_*$
    :param cs: $c_s$, speed of sound
    :param tau_end: $\tau_\text{end} = \frac{\eta_\text{end}}{R_*}$
    :return: $\tilde{P}_\text{gw}^\text{high}$
    """
    nu = nu_gdh2024(cs**2)
    tau_star = tau_star_func(r_star=HLf, nu=nu)
    upsilon = Upsilon(r=tau_star / tau_end, l=1 + 2 * nu)
    return quad(
        gw_spectral_density_approx_high_integrand, x_minus(z=z, cs=cs), x_plus(z=z, cs=cs), args=(z, cs, upsilon)
    )[0]


def Pgw_approx(z: float, HLf: float, cs: float, tau_star: float, tau_end: float) -> float:
    r"""GW power spectrum combining low, intermediate, and high frequency approximations.

    $$\tilde{P}_\text{gw} =
    \tilde{P}_\text{gw}^\text{low} \frac{1}{2} \operatorname{erfc} \left( 2 \pi \tau_* (z - z_{\times}) \right)
    + \frac{1}{2} \left[ 1 + \operatorname{erf} \left( 2 \pi \tau_* (z - z_{\times}) \right) \right]
    \tilde{P}_\text{gw}^\text{int}
    \frac{1}{2} \operatorname{erfc} \left( 2 \pi \tau_* \left( z - z_{\text{int,high}} \right) \right)
    + \tilde{P}_\text{gw}^\text{high}$$
    This is :giombi_2024_cs:`\ ` eq. 4.9 with the analytical $\tilde{P}_v$,
    except that the intermediate regime is switched off at the approximate crossing of the intermediate- and
    high-frequency regimes $z_{\text{int,high}} \approx c_s k_p R_*$
    (:py:func:`pttools.ssm.low_k.intersection.z_int_high_approx`)
    instead of the estimate for the peak of the GW power spectrum $k_\star R_* \simeq 2 c_s k_p R_*$
    (:py:func:`pttools.ssm.low_k.intersection.zp_gw_approx`), which is twice as large.
    Here $k_p R_* = 2\pi$ is the sound wave scale (:py:data:`pttools.ssm.low_k.const.ZP_SW`),
    and $z_\times$ is given by :py:func:`pttools.ssm.low_k.intersection.z_cross`.
    This cannot be jitted, as :py:func:`gw_spectral_density_approx_high` cannot.

    :param z: $z = kR_*$
    :param HLf: $H L_f = r_*$
    :param cs: $c_s$, speed of sound
    :param tau_star: $\tau_* = \frac{\eta_*}{R_*}$
    :param tau_end: $\tau_\text{end} = \frac{\eta_\text{end}}{R_*}$
    :return: $\tilde{P}_\text{gw}$
    """
    P_low = gw_spectral_density_approx_low(z, HLf, cs, tau_end)
    P_int = gw_spectral_density_approx_int(z, HLf, cs)
    P_high = gw_spectral_density_approx_high(z, HLf, cs, tau_end)
    return join_regimes(
        z=z, Pgw_low=P_low, Pgw_int=P_int, Pgw_high=P_high, tau_star=tau_star,
        z_cross=z_cross(HLf, cs, nu_gdh2024(cs**2), tau_star, tau_end),
        z_int_off=z_int_high_approx(cs=cs, xp_v=ZP_SW), z_high_on=-np.inf
    )


@njit(cache=True)
def Pv_analytical[T: FloatOrArr](k: T, kp_sw: T | float, ubarf2: T | float) -> T:
    r"""Analytical ansatz for $P_v(k)$.

    $$P_v(p) = 3 \pi \frac{\bar{U}_f^2}{k_p^3} \frac{(p/k_p)^2}{1 + (p/k_p)^6}$$
    :giombi_2024_cs:`\ ` eq. 3.1
    :giombi_2026:`\ ` eq. 3.1.

    :param k: $p$, wavenumber
    :param kp_sw: $k_p$, the sound wave scale, which is $p$ at the maximum of $p P_v(p)$.
        It has the same units as $k$, so for the dimensionless $\tilde{P}_v(x)$ with $k = x = pR_*$ it is
        $k_p R_* = 2\pi$ (:py:data:`pttools.ssm.low_k.const.ZP_SW`).
        This is not the peak of the GW power spectrum $k_\star$ (``kp_gw``).
    :param ubarf2: $\bar{U}_f^2$
    :return: $P_v(p)$
    """
    # The equation numbers in the articles happen to be the same.
    k_rel = k / kp_sw
    return 3 * np.pi * ubarf2 / kp_sw**3 * k_rel **2 / (1 + k_rel ** 6)  # pyrefly: ignore[bad-return]
