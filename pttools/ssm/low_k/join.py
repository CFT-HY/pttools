"""Functions for joining the GW power spectrum regimes together"""

import logging

import numpy as np
from scipy.special import erf, erfc

from pttools.ssm.low_k import integration, intersection
import pttools.type_hints as th

logger = logging.getLogger(__name__)


def gw_junction(
        z: th.FloatArr1D,
        Pgw_low: th.FloatArr1D,
        Pgw_int: th.FloatArr1D,
        Pgw_high: th.FloatArr1D,
        cs: float,
        nu: float,
        tau_star: float,
        tau_end: float,
        r_star: float):
    r"""
    Create the junction of the gravitational wave power spectrum between different regimes
    starting from the profiles in each regime.

    $\tilde{P}_\text{gw}(kR_*) =
    \tilde{P}_\text{gw}^\text{low} (kR_*) \frac{1}{2} \erfc \left( 2 \pi \eta_* (k - k_{\times}) \right) +
    \frac{1}{2} \left[ 1 + \erf \left( 2 \pi \eta_* (k - k_{\times}) \right) \right] \tilde{P}_\text{gw}^\text{int}
    \frac{1}{2} \erfc \left( 2 \pi \eta_* (k - k_{\star}) \right) +
    \tilde{P}_\text{gw}^\text{high} (kR_*)
    This is a part of :giombi_2024_cs:`\ ` eq. 4.9

    :param z: gravitational wave momentum values (kR_*)
    :param Pgw_low: array of gravitational wave power spectrum values in the low-frequency regime
    :param Pgw_int: array of gravitational wave power spectrum values in the intermediate-frequency regime
    :param Pgw_high: array of gravitational wave power spectrum values in the high-frequency regime
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    :param r_star: $H L_f = r_*$
    :return: gravitational wave power spectrum values at the given momentum
    """
    # if not (z.size and Pgw_low.size and Pgw_int.size and Pgw_high.size):
    #     raise ValueError(
    #         "Input arrays must not be empty. Got sizes: "
    #         f"z={z.size}, Pgw_low={Pgw_low.size}, Pgw_int={Pgw_int.size}, Pgw_high={Pgw_high.size}"
    #     )

    difference = Pgw_high - Pgw_int
    index = np.where(difference > 0)[0]
    if index.size:
        z_star = z[index[0]]
    else:
        z_star = 4 * cs * np.pi * (1 + nu) / r_star
        logger.warning("Using fallback z_star=%s for low-k junction.", z_star)
    z_cross = intersection.z_cross_junction(cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)

    term_low = 0.5 * erfc(2 * np.pi * tau_star * (z - z_cross)) * Pgw_low
    term_int = 0.5 * (1 + erf(2 * np.pi * tau_star * (z - z_cross))) * Pgw_int * \
               0.5 * erfc(2 * np.pi * tau_star * (z - z_star))
    term_high = 0.5 * (1 + erf(2 * np.pi * tau_star * (z - z_star))) * Pgw_high

    return term_low + term_int + term_high


def pow_gw_approximation(
        z: th.FloatArr1D,
        spec_den_v: th.FloatArr1D,
        cs: float,
        tau_star: float,
        tau_end: float,
        eps: float = 1e-8) -> th.FloatArr1D:
    r"""
    Spectral density of gravitational waves computed with the sound shell model plus analytic approximation
    in the low-frequency and intermediate-frequency regimes.
    Multiply by z**3/2/np.pi**2 * HR* Ht  to get the final power spectrum.

    :param z: gravitational wave momentum values (kR_*)
    :param spec_den_v: spectral density of the velocity field at the given momenta
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    :param eps: $\epsilon$, a small correction the integration x-range
    :return: gravitational wave power spectrum values at the given momentum z = kR_*
    """
    # Todo: The eps of 1e-8 seems to be needed for max(z) <= 100. Why?
    # nx = len(z) can be too few for velocity PS convolutions
    xmax = z.max() * (0.5 * (1. + cs) / cs) + eps
    xmin = z.min() * (0.5 * (1. - cs) / cs) - eps
    x: th.FloatArr1D = np.logspace(np.log10(xmin), np.log10(xmax), z.size)  # x = pR_*

    Pgw_high = 4/3 * integration.power_spectrum_integration_high(x, spec_den_v, z, cs)
    Pgw_low = 4/3 * integration.power_spectrum_integration_low(x, spec_den_v, z, cs=cs, tau_star=tau_star, tau_end=tau_end)
    Pgw_int = 4/3 * integration.power_spectrum_integration_int(x, spec_den_v, z, cs=cs, tau_star=tau_star)
    return gw_junction(z, Pgw_low, Pgw_int, Pgw_high, cs=cs, tau_star=tau_star, tau_end=tau_end)
