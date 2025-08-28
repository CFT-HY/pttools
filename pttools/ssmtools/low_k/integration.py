import numpy as np
from scipy.integrate import simpson
from scipy.special import gamma

from pttools.ssmtools import const


def power_spectrum_integration_low(
        x_data: np.ndarray[tuple[int], np.float64],
        Pv_data: np.ndarray[tuple[int], np.float64],
        z: np.ndarray[tuple[int], np.float64],
        cs: float,
        nu: float,
        tau_star: float,
        tau_end: float) -> np.ndarray[tuple[int], np.float64]:
    r"""
    Calculate the low-frequency approximation (kR_* << 1) of the gravitational wave power spectrum.
    One dimensional integration over sound wave momentum.

    :param x_data: momentum values (pR_*)
    :param Pv_data: power spectrum values at the given momenta
    :param z: array of gravitational wave momentum values (kR_*)
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param nu: $\nu_\text{gdh2024}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :param tau_end: $\tau_{end} = \frac{\eta_{end}}{L_f}$
    :return: gravitational wave power spectrum values at the given momentum
    """
    Pgw = np.zeros_like(z)  # initialize an empty array for the gravitational wave power spectrum
    factor = 16 * tau_star / 15 / np.pi**2
    # Momentum values for integration (trapezoidal rule or simpson rule)
    x = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 1000)

    # Compute the gravitational wave power spectrum for each value of z
    if cs >= const.CS0 - 1e-10:  # if cs is close to 1/sqrt(3), use the radiation dominated kernel
        delta_radiation = 0.25 * np.log(tau_end / tau_star)**2  # kernel function for radiation dominated era
        for i in range(len(Pgw)):
            integrand = x**2 * np.interp(x, x_data, Pv_data)**2 * delta_radiation
            Pgw[i] = factor * simpson(integrand, x=x)

    else:  # if cs is not close to 1/sqrt(3), use the kernel function with cs^2 \neq 1/3
        for i in range(len(Pgw)):
            Delta = (0.5 * z[i] * tau_star) ** (-2 * nu) * gamma(0.5 + nu) ** 2 / (4 * np.pi) * (
                    1 - (tau_star / tau_end) ** (2 * nu)) ** 2 / (2 * nu) ** 2  # kernel function with cs^2 \neq 1/3
            integrand = x ** 2 * np.interp(x, x_data, Pv_data) ** 2 * Delta  # integrand for the gravitational wave power spectrum
            Pgw[i] = factor * simpson(integrand, x=x)

    return Pgw


def power_spectrum_integration_int(
        x_data: np.ndarray[tuple[int], np.float64],
        Pv_data: np.ndarray[tuple[int], np.float64],
        z: np.ndarray[tuple[int], np.float64],
        cs: float,
        tau_star: float) -> np.ndarray[tuple[int], np.float64]:
    r"""
    Calculate the intermediate-frequency approximation (1 << k eta_* << kp eta_*) of the gravitational wave power spectrum.
    One dimensional integration over sound wave momentum.
    Note that this approximation does not depend on tau_end, as it assumes several gravitational wave oscillations
    during the acoustic sourcing (eta_end - eta_* >> eta_*)

    :param x_data: array of momentum values (pR_*)
    :param Pv_data: array of power spectrum values at the given momentum
    :param z: array of gravitational wave momentum values (kR_*)
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :param tau_star: $\tau_* = \frac{\eta_*}{L_f}$
    :return: gravitational wave power spectrum values at the given momentum
    """
    # nu = (1- 3*cs**2)/(1+ 3*cs**2)
    Pgw = np.zeros_like(z)  # initialize an empty array for the gravitational wave power spectrum
    # momentum values for integration (trapezoidal rule or simpson rule)
    x = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 1000)

    # compute the gravitational wave power spectrum for each value of z
    for i in range(len(Pgw)):
        factor = 4 / 3 / cs ** 4 * (3 - 2 * cs ** 2 - 3 / cs * (1 - cs ** 2) * np.arctanh(cs)) / tau_star / z[i] ** 2
        integrand = x ** 2 * np.interp(x, x_data, Pv_data) ** 2 / 2 / np.pi ** 2
        Pgw[i] = factor * simpson(integrand, x=x)

    return Pgw


def power_spectrum_integration_high(
        x_data: np.ndarray[tuple[int], np.float64],
        Pv_data: np.ndarray[tuple[int], np.float64],
        z: np.ndarray[tuple[int], np.float64],
        cs: float = const.CS0) -> np.ndarray[tuple[int], np.float64]:
    r"""Previously known as _peak

    :param x_data: array of momentum values (pR_*)
    :param Pv_data: array of power spectrum values at the given momentum
    :param z: array of gravitational wave momentum values (kR_*)
    :param cs: sound speed, $0 < c_s < \frac{1}{\sqrt{3}}$
    :return: gravitational wave power spectrum values at the given momentum
    """
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
