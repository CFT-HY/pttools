import numpy as np

import pttools.type_hints as th
from pttools.omgw0 import const


def signal_to_noise_ratio(
        f: np.ndarray,
        signal: np.ndarray,
        noise: np.ndarray,
        obs_time: float = const.LISA_OBS_TIME) -> th.FloatOrArr:
    r"""Signal-to-noise ratio
    $$\rho = \sqrt{T_{\text{obs}} \int_{{f}_\text{min}}^{{f}_\text{max}} df \frac{
    h^2 \Omega_{\text{gw},0}^2}{
    h^2 \Omega_{\text{n}}^2}}
    :gowling_2021:`\ ` eq. 3.12,
    :smith_2019:`\ ` p. 1

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
    :gowling_2021:`\ ` p. 12
    """
    return const.c / (2*np.pi*L)

FT_LISA = ft()
_ft_func = ft


def N_AE(f: th.FloatOrArr, ft: th.FloatOrArr = FT_LISA, L: th.FloatOrArr = const.LISA_ARM_LENGTH, W_abs2: th.FloatOrArr = None) -> th.FloatOrArrNumba:
    r"""A and E channels of LISA instrument noise
    $$N_A = N_E = ...$$
    :gowling_2021:`\ ` eq. 3.4
    """
    cos_f_frac = np.cos(f/ft)
    if W_abs2 is None:
        W_abs2 = np.abs(W(f, ft))**2
    return ((4 + 2*cos_f_frac)*P_oms(L) + 8*(1 + cos_f_frac + cos_f_frac**2) * P_acc(f, L)) * W_abs2


def omega(f: th.FloatOrArr, S: th.FloatOrArr) -> th.FloatOrArr:
    r"""Convert an effective noise power spectral density (aka. sensitivity) $S$
    to a fractional GW energy density power spectrum $\Omega$
    $$\Omega = \frac{4 \pi^2}{3 H_0^2} f^3 S(f)$$
    :gowling_2021:`\ ` eq. 3.8,
    :gowling_2023:`\ ` eq. 3.8,
    :smith_2019:`\ ` p. 1
    """
    return 4*np.pi**2 / (3*const.H0_HZ**2) * f**3 * S


def omega_eb(f: th.FloatOrArr, f_ref_eb: float = 25, omega_ref_eb: float = 8.9e-10) -> th.FloatOrArr:
    return omega_ref_eb * (f/f_ref_eb)**(2/3)


def omega_gb(f: th.FloatOrArr) -> th.FloatOrArr:
    return omega(f, S_gb(f))


def omega_ins(f: th.FloatOrArr) -> th.FloatOrArr:
    r"""LISA instrument noise
    $$\Omega_\text{ins} = \left( \frac{4 \pi^2}{3 H_0^2} f^3 S_A(f)$$
    """
    return omega(f=f, S=S_AE(f))


def omega_noise(f: th.FloatOrArr) -> th.FloatOrArr:
    r"""$$\Omega_\text{noise} = \Omega_\text{ins} + \Omega_\text{eb} + \Omega_\text{gb}$$"""
    return omega_ins(f) + omega_eb(f) + omega_gb(f)


def P_acc(f: th.FloatOrArr, L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArr:
    r"""
    LISA single test mass acceleration noise
    :gowling_2021:`\ ` eq. 3.3
    """
    return (3e-15 / ((2*np.pi*f)**2 * L))**2 * (1 + (0.4e-3/f)**2)


def P_oms(L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArr:
    r"""
    LISA optical metrology noise
    $$P_\text{oms}(f) = \left( \frac{1.5 \cdot 10^{-11} \text{m}}{L} \right)^2 \text{Hz}^{-1}$$
    :gowling_2021:`\ ` eq. 3.2
    This is white noise and therefore independent of the frequency.
    Note that there is a typo on :gowling_2021:`\ ` p. 12:
    the correct $L = 2.5 \cdot 10^9 \text{m}$.
    """
    return (1.5e-11 / L)**2


def R_AE(f: th.FloatOrArr, ft: th.FloatOrArr = FT_LISA, W_abs2: th.FloatOrArr = None) -> th.FloatOrArr:
    r"""Gravitational wave response function for the A and E channels
    :gowling_2021:`\ ` eq. 3.6
    """
    if W_abs2 is None:
        W_abs2 = np.abs(W(f, ft))**2
    return 9/20 * W_abs2 / (1 + (3*f/(4*ft))**2)


def S(N: th.FloatOrArr, R: th.FloatOrArr) -> th.FloatOrArr:
    r"""Noise power spectral density
    $$S = \frac{N}{\mathcal{R}}$$
    :gowling_2021:`\ ` eq. 3.1
    """
    return N / R


def S_AE(f: th.FloatOrArr, ft: th.FloatOrArr = FT_LISA, L: th.FloatOrArr = const.LISA_ARM_LENGTH, both_channels: bool = True) -> th.FloatOrArr:
    r"""Noise power spectral density for the LISA A and E channels
    $$S_A = S_E = \frac{N_A}{\mathcal{R}_A}$$
    :gowling_2021:`\ ` eq. 3.7
    """
    # The W_abs2 cancels and can therefore be set to unity
    ret = S(N=N_AE(f=f, ft=ft, L=L, W_abs2=1), R=R_AE(f=f, ft=ft, W_abs2=1))
    if both_channels:
        return 1/np.sqrt(2) * ret
    return ret


def S_AE_approx(f: th.FloatOrArr, L: th.FloatOrArr = const.LISA_ARM_LENGTH, both_channels: bool = True) -> th.FloatOrArr:
    r"""Approximate noise power spectral density for the LISA A and E channels
    $$S_A = S_E = \frac{N_A}{\mathcal{R}_A}
    \approx \frac{40}{3} ({P}_\text{oms} + {4P}_\text{acc}) \left( 1 + \frac{3f}{4f_t} \right^2$$
    :gowling_2021:`\ ` eq. 3.7
    """
    ret = 40/3 * (P_oms(L) + 4*P_acc(f, L)) * (1 + (3*f/(4*ft(L)))**2)
    if both_channels:
        return 1/np.sqrt(2) * ret
    return ret


def S_gb(
        f: th.FloatOrArr,
        A: float = 9e-35,  # 1/mHz -> 10³
        f_ref_gb: float = 1,
        fk: float = 1.13e-3,
        a: float = 0.138,
        b: float = -221,
        c: float = 521,
        d: float = 1680) -> th.FloatOrArr:
    r"""Noise power spectral density for galactic binaries"""
    return A * (1e-3 / f)**(-7/3) * np.exp(-(f/f_ref_gb)**a - b*f*np.sin(c*f)) * (1 + np.tanh(d*(fk - f)))


def W(f: th.FloatOrArr, ft: th.FloatOrArr) -> th.FloatOrArr:
    r"""Round trip modulation
    $$W(f,f_t) = 1 - e^{-2i \frac{f}{f_t}}$$
    :gowling_2021:`\ ` p. 12
    """
    return 1 - np.exp(-2j * f / ft)
