import numpy as np
from scipy.special import erf, erfc

from pttools.ssmtools.spectrum_bag import spec_den_v_bag
from pttools.ssmtools.low_k import integration, intersection
from pttools.ssmtools.low_k.utils import parse_params_gw


def Pgw_junction(z, Pgw_low, Pgw_int, Pgw_high, params_gw):
    """
    Create the junction of the gravitaional wave power spectrum between different regimes
    starting from the profiles in each regime.
    Parameters:
        - z: array of gravitational wave momentum values (kR_*)
        - Pgw_low: array of gravitational wave power spectrum values in the low-frequency regime
        - Pgw_int: array of gravitational wave power spectrum values in the intermediate-frequency regime
        - Pgw_high: array of gravitational wave power spectrum values in the high-frequency regime

    Input parameters for gravitational wave power spectrum:
        cs = params_gw[0]       scalar  (required) [0 < cs < 1/sqrt(3)]
        tau_star = params_gw[1] scalar  (required) [tau_star = eta_star/Lf]
        tau_end = params_gw[2]  scalar  (required) [tau_end = eta_end/Lf]
    Returns:
        - Pgw: array of gravitational wave power spectrum values at the given momentum
    """

    cs, tau_star, tau_end = parse_params_gw(params_gw)  # unpack parameters for gravitational wave power spectrum
    nu = (1 - 3 * cs ** 2) / (1 + 3 * cs ** 2)
    # z_star = 4*cs*np.pi * (1+nu) / HLf
    difference = Pgw_high - Pgw_int
    index = np.where(difference > 0)[0]
    z_star = z[index[0]]  # if len(index) > 0 else z_star
    z_cross = intersection.cross_z_junction(params_gw)
    # print(z_star)

    term_low = 0.5 * erfc(2 * np.pi * tau_star * (z - z_cross)) * Pgw_low
    term_int = 0.5 * (1 + erf(2 * np.pi * tau_star * (z - z_cross))) * Pgw_int * 0.5 * erfc(
        2 * np.pi * tau_star * (z - z_star))
    term_high = 0.5 * (1 + erf(2 * np.pi * tau_star * (z - z_star))) * Pgw_high

    return term_low + term_int + term_high


def Pgw_approximation(z, params_v, params_gw):
    """
    Spectral density of gravitaional waves computed with the sound shell model plus analytic approximation
    in the low-frequency and intermediate-frequency regimes.
    Multiply by z**3/2/np.pi**2 * HR* Ht  to get the final power spectrum.

    Input parameters for velocity spectral density:
        vw = params_v[0]       scalar  (required) [0 < vw < 1]
        alpha = params_v[1]    scalar  (required) [0 < alpha_n < alpha_n_max(v_w)]
        nuc_type = params_v[2] string  (optional) [exponential* | simultaneous]
        nuc_args = params_v[3] tuple   (optional) default (1,)

    Input parameters for gravitational wave power spectrum:
        cs = params_gw[0]       scalar  (required) [0 < cs < 1/sqrt(3)]
        tau_star = params_gw[1] scalar  (required) [tau_star = eta_star/Lf]
        tau_end = params_gw[2]  scalar  (required) [tau_end = eta_end/Lf]

    Returns:
        - Pgw: array of gravitational wave power spectrum values at the given momentum z = kR_*
    """

    cs, tau_star, tau_end = parse_params_gw(params_gw)  # unpack parameters for gravitational wave power spectrum

    eps = 1e-8  # Seems to be needed for max(z) <= 100. Why?
    #    nx = len(z) - this can be too few for velocity PS convolutions
    npt = len(z)  # number of points for the logspace in the power spectrum integration
    xmax = max(z) * (0.5 * (1. + cs) / cs) + eps
    xmin = min(z) * (0.5 * (1. - cs) / cs) - eps

    x = np.logspace(np.log10(xmin), np.log10(xmax), npt)  # x = pR_*

    velocity_spectral_density = spec_den_v_bag(x, params_v)  # Pv from sound shell model
    Pgw_high = 4 / 3 * integration.power_spectrum_integration_high(x, velocity_spectral_density, z, cs)
    Pgw_low = 4 / 3 * integration.power_spectrum_integration_low(x, velocity_spectral_density, z, params_gw)
    Pgw_int = 4 / 3 * integration.power_spectrum_integration_int(x, velocity_spectral_density, z, params_gw)
    # spectal_densities = np.array([Pgw_low, Pgw_int, Pgw_peak], dtype = object)

    Pgw_approx = Pgw_junction(z, Pgw_low, Pgw_int, Pgw_high, params_gw)

    return Pgw_approx
