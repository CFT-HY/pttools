import numpy as np

import pttools.type_hints as th
from pttools.omgw0 import const


def signal_to_noise_ratio(
        f: np.ndarray,
        signal: np.ndarray,
        noise: np.ndarray,
        obs_time: float = const.LISA_OBS_TIME) -> th.FloatOrArr:
    r"""Signal-to-noise ratio
    $$\rho = \sqrt{T_{\text{obs}} \int_{f_\text{min}}^{f_\text{max}} df \frac{
    h^2 \Omega_{\text{gw},0}^2}{
    h^2 \Omega_{\text{n}}^2}}
    :gowling_2021:`\ ` eq. 3.12

    :param f: frequencies
    :param signal: signal array
    :param noise: noise array
    :obs_time: observation time
    :return: signal-to-noise ratio $\rho$
    """
    return np.sqrt(obs_time * np.trapezoid(signal**2 / noise**2, f))


def ft(L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArr:
    r"""Transfer frequency
    $$f_t = \frac{c}{2\pi L}$$
    """
    return const.c / (2*np.pi*L)


# def N_AE(f: th.FloatOrArr, L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArrNumba:
#     r"""A and E channels of LISA instrument noise
#     $$N_A = N_E = ...$$
#     :gowling_2021:`\ ` eq. 3.4
#     """
#     f_frac = f/ft
#     cos_f_frac = np.cos(f_frac)
#     W = 1 - np.exp(-2j*f_frac)
#     return ((4 + 2*cos_f_frac)*P_oms(f, L) + 8*(1 + cos_f_frac + cos_f_frac**2 * P_acc(f, L))) * np.abs(W)**2


def omega_eb(f: th.FloatOrArr, f_ref_eb: float = 25, omega_ref_eb: float = 8.9e-10) -> th.FloatOrArr:
    return omega_ref_eb * (f/f_ref_eb)**(2/3)


def omega_gb(f: th.FloatOrArr) -> th.FloatOrArr:
    return 4*np.pi**2 / (3*const.H0**2) * f**3 * S_gb(f)


def omega_ins(f: th.FloatOrArr) -> th.FloatOrArr:
    r"""LISA instrument noise
    $$\Omega_\text{ins} = \left( \frac{4 \pi^2}{3 H_0^2} f^3 S_A(f)$$
    """
    return (4*np.pi**2 / (3*const.H0**2)) * f**3 * S_AE(f)


def omega_noise(f: th.FloatOrArr) -> th.FloatOrArr:
    return omega_ins(f) + omega_eb(f) + omega_gb(f)


def P_acc(f: th.FloatOrArr, L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArr:
    r"""
    LISA single test mass acceleration noise
    :gowling_2021:`\ ` eq. 3.3
    """
    return (3e-15 / ((2*np.pi*f)**2 * L))**2 * (1 + (0.4e-3/f)**2)


def P_oms(f: th.FloatOrArr, L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArr:
    r"""
    LISA optical metrology noise
    $$P_\text{oms}(f) = \left( \frac{1.5 \cdot 10^{-11} \text{m}}{L} \right)^2 \text{Hz}^{-1}$$
    :gowling_2021:`\ ` eq. 3.2
    """
    # Todo: Why is the frequency not used?
    return (1.5e-11 / L)**2


def S_AE(f: th.FloatOrArr, L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArr:
    r"""Approximate noise power spectral density for the LISA A and E channels
    $$S_A = S_E$
    """
    return 40/3 * (P_oms(f, L) + 4*P_acc(f, L)) * (1 + (f/(4*ft(L)/3))**2)


# TODO: fix the units for A
def S_gb(
        f: th.FloatOrArr,
        A: float = 9e-38,
        f_ref_gb: float = 1,
        fk: float = 1.13e-3,
        a: float = 0.138,
        b: float = -221,
        c: float = 521,
        d: float = 1680) -> th.FloatOrArr:
    r"""Noise power spectral density for galactic binaries"""
    return A * (1e-3 / f)**(-7/3) * np.exp(-(f/f_ref_gb)**a - b*f*np.sin(c*f)) * (1 + np.tanh(d*(fk - f)))
