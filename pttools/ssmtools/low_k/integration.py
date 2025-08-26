import numpy as np
from scipy.integrate import simpson
from scipy.special import gamma

from pttools.ssmtools.low_k.utils import parse_params_gw


def power_spectrum_integration_low(x_data, Pv_data, z, params_gw):
    """
    Calculate the low-frequency approximation (kR_* << 1) of the gravitational wave power spectrum.
    One dimensional integration over sound wave momentum.
    Parameters:
        - x_data: array of momentum values (pR_*)
        - Pv_data: array of power spectrum values at the given momentum
        - z: array of gravitational wave momentum values (kR_*)

    Input parameters for gravitational wave power spectrum:
        cs = params_gw[0]       scalar  (required) [0 < cs < 1/sqrt(3)]
        tau_star = params_gw[1] scalar  (required) [tau_star = eta_star/Lf]
        tau_end = params_gw[2]  scalar  (required) [tau_end = eta_end/Lf]
    Returns:
        - Pgw: array of gravitational wave power spectrum values at the given momentum
    """
    cs, tau_star, tau_end = parse_params_gw(params_gw)  # unpack parameters for gravitational wave power spectrum

    nu = (1 - 3 * cs ** 2) / (1 + 3 * cs ** 2)  # conformal parameter
    Pgw = np.zeros_like(z)  # initialize an empty array for the gravitational wave power spectrum
    factor = 16 * tau_star / 15 / np.pi ** 2
    xm = min(x_data)  # left extremum of momentum integration
    xp = max(x_data)  # right extremum of momentum integration
    x = np.logspace(np.log10(xm), np.log10(xp),
                    1000)  # momentum values for integration (trapezoidal rule or simpson rule)

    # compute the gravitational wave power spectrum for each value of z
    if (cs >= np.sqrt(1 / 3) - 1e-10):  # if cs is close to 1/sqrt(3), use the radiation dominated kernel
        # print('cs^2 = 1/3')
        Delta_radiation = 0.25 * np.log(tau_end / tau_star) ** 2  # kernel function for radiation dominated era
        for i in range(len(Pgw)):
            integrand = x ** 2 * np.interp(x, x_data, Pv_data) ** 2 * Delta_radiation
            Pgw[i] = factor * simpson(integrand, x=x)

    else:  # if cs is not close to 1/sqrt(3), use the kernel function with cs^2 \neq 1/3
        # print('cs^2 != 1/3')
        for i in range(len(Pgw)):
            Delta = (0.5 * z[i] * tau_star) ** (-2 * nu) * gamma(0.5 + nu) ** 2 / (4 * np.pi) * (
                    1 - (tau_star / tau_end) ** (2 * nu)) ** 2 / (2 * nu) ** 2  # kernel function with cs^2 \neq 1/3
            integrand = x ** 2 * np.interp(x, x_data,
                                           Pv_data) ** 2 * Delta  # integrand for the gravitational wave power spectrum
            Pgw[i] = factor * simpson(integrand, x=x)

    return Pgw


def power_spectrum_integration_int(x_data, Pv_data, z, params_gw):
    """
    Calculate the intermediate-frequency approximation (1 << k eta_* << kp eta_*) of the gravitational wave power spectrum.
    One dimensional integration over sound wave momentum.
    Note that this approximation does not depend on tau_end, as it assumes several gravitational wave oscillations
    during the acoustic sourcing (eta_end - eta_* >> eta_*)
    Parameters:
        - x_data: array of momentum values (pR_*)
        - Pv_data: array of power spectrum values at the given momentum
        - z: array of gravitational wave momentum values (kR_*)

    Input parameters for gravitational wave power spectrum:
        cs = params_gw[0]       scalar  (required) [0 < cs < 1/sqrt(3)]
        tau_star = params_gw[1] scalar  (required) [tau_star = eta_star/Lf]
        tau_end = params_gw[2]  scalar  (required) [tau_end = eta_end/Lf]
    Returns:
        - Pgw: array of gravitational wave power spectrum values at the given momentum
    """
    cs, tau_star, _ = parse_params_gw(params_gw)  # unpack parameters for gravitational wave power spectrum
    # nu = (1- 3*cs**2)/(1+ 3*cs**2)
    Pgw = np.zeros_like(z)  # initialize an empty array for the gravitational wave power spectrum
    xm = min(x_data)  # left extremum of momentum integration
    xp = max(x_data)  # right extremum of momentum integration
    x = np.logspace(np.log10(xm), np.log10(xp),
                    1000)  # momentum values for integration (trapezoidal rule or simpson rule)

    # compute the gravitational wave power spectrum for each value of z
    for i in range(len(Pgw)):
        factor = 4 / 3 / cs ** 4 * (3 - 2 * cs ** 2 - 3 / cs * (1 - cs ** 2) * np.arctanh(cs)) / tau_star / z[i] ** 2
        integrand = x ** 2 * np.interp(x, x_data, Pv_data) ** 2 / 2 / np.pi ** 2
        Pgw[i] = factor * simpson(integrand, x=x)

    return Pgw


def power_spectrum_integration_high(x_data, Pv_data, z, cs):
    """Previously known as _peak"""
    Pgw = np.zeros_like(z)
    cs = np.sqrt(1 / 3)
    for i in range(len(Pgw)):
        # print(i)
        factor = 1 / (4 * np.pi * z[i] * cs) * (1 - cs ** 2) ** 2 / cs ** 4
        xm = 0.5 * z[i] * (1 - cs) / cs
        xp = 0.5 * z[i] * (1 + cs) / cs
        x = np.logspace(np.log10(xm), np.log10(xp), 1000)
        integrand = (x - xp) ** 2 * (x - xm) ** 2 / x / (xp + xm - x) * np.interp(x,
                                                                                  x_data, Pv_data) * np.interp(
            (xp + xm - x), x_data, Pv_data)
        Pgw[i] = factor * simpson(integrand, x=x)

    return Pgw
