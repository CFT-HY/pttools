"""Functions for joining the GW power spectrum regimes together."""

import logging
import math

import numba
import numpy as np

from pttools.speedup import logspace, njit, vectorize
from pttools.ssm.low_k import integration, intersection
from pttools.ssm.rho import x_minus, x_plus
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr

logger: logging.Logger = logging.getLogger(__name__)


@vectorize(cache=True)
def switch_off[T: FloatOrArr](z: T, z0: float, tau_star: float) -> T:
    r"""Smooth switch from 1 to 0 at $z_0$.

    $$\frac{1}{2} \operatorname{erfc} \left( 2 \pi \tau_* (z - z_0) \right)
    = \frac{1}{2} \operatorname{erfc} \left( 2 \pi \eta_* (k - k_0) \right)$$
    :giombi_2024_cs:`\ ` eq. 4.9

    :param z: $z = kR_*$
    :param z0: $z_0 = k_0 R_*$, the location of the switch
    :param tau_star: $\tau_* = \frac{\eta_*}{R_*}$, which sets the width of the switch
    :return: value of the switch
    """
    return 0.5 * math.erfc(2 * np.pi * tau_star * (z - z0))  # pyrefly: ignore[bad-return, bad-argument-type]


@vectorize(cache=True)
def switch_on[T: FloatOrArr](z: T, z0: float, tau_star: float) -> T:
    r"""Smooth switch from 0 to 1 at $z_0$.

    $$\frac{1}{2} \left[ 1 + \operatorname{erf} \left( 2 \pi \tau_* (z - z_0) \right) \right]
    = \frac{1}{2} \left[ 1 + \operatorname{erf} \left( 2 \pi \eta_* (k - k_0) \right) \right]$$
    :giombi_2024_cs:`\ ` eq. 4.9

    :param z: $z = kR_*$
    :param z0: $z_0 = k_0 R_*$, the location of the switch
    :param tau_star: $\tau_* = \frac{\eta_*}{R_*}$, which sets the width of the switch
    :return: value of the switch
    """
    return 0.5 * (1 + math.erf(2 * np.pi * tau_star * (z - z0)))  # pyrefly: ignore[bad-return, bad-argument-type]


@njit(cache=True)
def join_regimes[T: FloatOrArr](
        z: T,
        Pgw_low: T,
        Pgw_int: T,
        Pgw_high: T,
        tau_star: float,
        z_cross: float,
        z_int_off: float,
        z_high_on: float) -> T:
    r"""Join the low-, intermediate- and high-frequency regimes of the GW spectral density with smooth switches.

    $$\tilde{P}_\text{gw} =
    \tilde{P}_\text{gw}^\text{low} \frac{1}{2} \operatorname{erfc} \left( 2 \pi \tau_* (z - z_{\times}) \right)
    + \frac{1}{2} \left[ 1 + \operatorname{erf} \left( 2 \pi \tau_* (z - z_{\times}) \right) \right]
    \tilde{P}_\text{gw}^\text{int}
    \frac{1}{2} \operatorname{erfc} \left( 2 \pi \tau_* (z - z_{\text{int,off}}) \right)
    + \frac{1}{2} \left[ 1 + \operatorname{erf} \left( 2 \pi \tau_* (z - z_{\text{high,on}}) \right) \right]
    \tilde{P}_\text{gw}^\text{high}$$
    This is :giombi_2024_cs:`\ ` eq. 4.9 with $\tau_* z = \eta_* k$, $z_\text{int,off} = z_{\star}$
    and $z_\text{high,on} = -\infty$, i.e. without a switch for the high-frequency regime.

    :param z: $z = kR_*$
    :param Pgw_low: $\tilde{P}_\text{gw}^\text{low}$
    :param Pgw_int: $\tilde{P}_\text{gw}^\text{int}$
    :param Pgw_high: $\tilde{P}_\text{gw}^\text{high}$
    :param tau_star: $\tau_* = \frac{\eta_*}{R_*}$
    :param z_cross: $z_\times$, where the low-frequency regime is switched off and the intermediate one on
    :param z_int_off: $z_\text{int,off}$, where the intermediate-frequency regime is switched off
    :param z_high_on: $z_\text{high,on}$, where the high-frequency regime is switched on.
        Use $-\infty$ to have no switch.
    :return: $\tilde{P}_\text{gw}$
    """
    return (
        switch_off(z, z_cross, tau_star) * Pgw_low  # pyrefly: ignore[bad-return]
        + switch_on(z, z_cross, tau_star) * Pgw_int * switch_off(z, z_int_off, tau_star)
        + switch_on(z, z_high_on, tau_star) * Pgw_high
    )


@njit
def gw_junction(
        z: th.FloatArr1D,
        Pgw_low: th.FloatArr1D,
        Pgw_int: th.FloatArr1D,
        Pgw_high: th.FloatArr1D,
        cs: float,
        nu: float,
        tau_star: float,
        tau_end: float,
        cs_friction: bool = False) -> th.FloatArr1D:
    r"""
    Create the junction of the gravitational wave power spectrum between different regimes.

    The junction is created starting from the profiles in each regime.

    $$\tilde{P}_\text{gw}(kR_*) =
    \tilde{P}_\text{gw}^\text{low} (kR_*) \frac{1}{2} \operatorname{erfc} \left( 2 \pi \eta_* (k - k_{\times}) \right)
    + \frac{1}{2} \left[ 1 + \operatorname{erf} \left( 2 \pi \eta_* (k - k_{\times}) \right) \right]
    \tilde{P}_\text{gw}^\text{int} (kR_*)
    \frac{1}{2} \operatorname{erfc} \left( 2 \pi \eta_* (k - k_{\star}) \right)
    + \frac{1}{2} \left[ 1 + \operatorname{erf} \left( 2 \pi \eta_* (k - k_{\star}) \right) \right]
    \tilde{P}_\text{gw}^\text{high} (kR_*)$$
    This is :giombi_2024_cs:`\ ` eq. 4.9 with a switch also for the high-frequency regime.
    Here $k_\star$ is the first $k$ where $\tilde{P}_\text{gw}^\text{high} > \tilde{P}_\text{gw}^\text{int}$,
    or :py:func:`pttools.ssm.low_k.intersection.z_star_approx` if there is no such $k$,
    and $k_\times$ is given by :py:func:`pttools.ssm.low_k.intersection.z_cross_junction`.
    The warning is logged in object mode, as logging is not supported in Numba-jitted code.

    :param z: gravitational wave momentum values (kR_*)
    :param Pgw_low: array of gravitational wave power spectrum values in the low-frequency regime
    :param Pgw_int: array of gravitational wave power spectrum values in the intermediate-frequency regime
    :param Pgw_high: array of gravitational wave power spectrum values in the high-frequency regime
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    :param cs_friction: whether to compute the acoustic friction from $c_s$ instead of $\nu$,
        see :py:func:`pttools.ssm.low_k.kernel.kernel_low`. Disabled by default.
    :return: gravitational wave power spectrum values at the given momentum
    """
    difference = Pgw_high - Pgw_int
    index = np.where(difference > 0)[0]
    if index.size:
        z_star = z[index[0]]
    else:
        z_star = intersection.z_star_approx(cs=cs)
        with numba.objmode:
            logger.warning("Using fallback z_star=%s for low-k junction.", z_star)
    z_cross = intersection.z_cross_junction(
        cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end, cs_friction=cs_friction)
    return join_regimes(
        z=z, Pgw_low=Pgw_low, Pgw_int=Pgw_int, Pgw_high=Pgw_high,
        tau_star=tau_star, z_cross=z_cross, z_int_off=z_star, z_high_on=z_star
    )


@njit
def pow_gw_approximation(
        z: th.FloatArr1D,
        spec_den_v: th.FloatArr1D,
        cs: float,
        nu: float,
        tau_star: float,
        tau_end: float,
        eps: float = 1e-8,
        cs_friction: bool = False) -> th.FloatArr1D:
    r"""
    Spectral density of gravitational waves from the SSM with low- and intermediate-frequency approximations.

    This is computed with the sound shell model plus analytic approximation
    in the low-frequency and intermediate-frequency regimes.
    Multiply by z**3/2/np.pi**2 * HR* Ht  to get the final power spectrum.

    $$\tilde{P}_\text{gw} = \frac{4}{3} \tilde{P}_\text{gw}^\text{junction}\left(
    \tilde{P}_\text{gw}^\text{low}, \tilde{P}_\text{gw}^\text{int}, \tilde{P}_\text{gw}^\text{high} \right),$$
    where the regimes are joined with :py:func:`gw_junction`,
    and $\tilde{P}_v$ is sampled at $n_z$ logarithmically spaced points $x \in [x_{-}(z_{\text{min}}) - \epsilon,
    x_{+}(z_{\text{max}}) + \epsilon]$, where $n_z$ is the size of $z$.
    The origin of the factor $\frac{4}{3}$ is not documented.

    :param z: gravitational wave momentum values (kR_*)
    :param spec_den_v: spectral density of the velocity field at the momenta $x$ described above
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    :param eps: $\epsilon$, a small correction the integration x-range
    :param cs_friction: whether to compute the acoustic friction from $c_s$ instead of $\nu$,
        see :py:func:`pttools.ssm.low_k.kernel.kernel_low`. Disabled by default.
    :return: gravitational wave power spectrum values at the given momentum z = kR_*
    """
    # Todo: The eps of 1e-8 seems to be needed for max(z) <= 100. Why?
    # nx = len(z) can be too few for velocity PS convolutions
    xmax = x_plus(z=z.max(), cs=cs) + eps
    xmin = x_minus(z=z.min(), cs=cs) - eps
    x: th.FloatArr1D = logspace(np.log10(xmin), np.log10(xmax), z.size)  # x = pR_*

    Pgw_high = 4/3 * integration.power_spectrum_integration_high(x, spec_den_v, z, cs)
    Pgw_low = 4/3 * integration.power_spectrum_integration_low(
        x, spec_den_v, z, cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end, cs_friction=cs_friction)
    Pgw_int = 4/3 * integration.power_spectrum_integration_int(
        z, cs=cs, tau_star=tau_star, Iv=integration.Iv_resampled(x=x, P_tilde_v=spec_den_v))
    return gw_junction(
        z, Pgw_low, Pgw_int, Pgw_high, cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end,
        cs_friction=cs_friction)
