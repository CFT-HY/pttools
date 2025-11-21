"""Utilities for calculating the noise of gravitational wave detectors, especially LISA"""

import typing as tp

import numpy as np

import pttools.type_hints as th
from pttools.omgw0 import const


def signal_to_noise_ratio(
        f: np.ndarray[tuple[int], np.float64],
        signal: np.ndarray[tuple[int], np.float64],
        noise: np.ndarray[tuple[int], np.float64],
        f_noise: tp.Union[np.ndarray[tuple[int], np.float64], None] = None,
        obs_time: float = const.LISA_OBS_TIME,
        f_min: float | None = None,
        f_max: float | None = None) -> float:
    r"""Signal-to-noise ratio
    $$\rho = \sqrt{T_{\text{obs}} \int_{{f}_\text{min}}^{{f}_\text{max}} df \frac{
    h^2 \Omega_{\text{signal}}^2}{
    h^2 \Omega_{\text{noise}}^2}}$$
    :caprini_2020:`\ ` eq. 33
    :smith_2019:`\ ` eq. 60

    The equation :gowling_2023:`\ ` eq. 3.9 has an additional factor of 2,
    which is canceled out by another factor of 2 in eq. 3.8.

    :param f: frequencies (Hz)
    :param signal: $\Omega_\text{signal}$
    :param noise: $\Omega_\text{noise}$
    :param f_noise: frequencies for the noise (assumed to be the same as for the signal, if not provided)
    :param obs_time: observation time (s)
    :param f_min: minimum frequency to be considered (Hz)
    :param f_max: maximum frequency to be considered (Hz)
    :return: signal-to-noise ratio SNR, aka. $\rho$
    """
    if f_noise is None:
        if not (f_min is None and f_max is None):
            i_f_min = 0 if f_min is None else np.argmax(f >= f_min)
            i_f_max = -1 if f_max is None else np.argmax(f >= f_max)
            f = f[i_f_min:i_f_max]
            noise = noise[i_f_min:i_f_max]
            signal = signal[i_f_min:i_f_max]
    else:
        if f_min is None:
            f_min = max(f[0], f_noise[0])
        if f_max is None:
            f_max = min(f[-1], f_noise[-1])
        i_f_min = np.argmax(f_noise >= f_min)
        i_f_max = np.argmax(f_noise >= f_max)
        f_gw = f

        f = f_noise[i_f_min:i_f_max]
        noise = noise[i_f_min:i_f_max]
        signal = 10.**np.interp(np.log10(f), np.log10(f_gw), np.log10(signal))

    return np.sqrt(obs_time * np.trapezoid(signal**2 / noise**2, f))


def ft(L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArr:
    r"""Transfer frequency
    $$f_t = \frac{c}{2\pi L}$$
    :gowling_2021:`\ ` p. 12
    """
    return const.c / (2*np.pi*L)

#: Default LISA transfer frequency $f_t$
FT_LISA: float = ft()
#: $f_2$ from :lisa_sci_req:`\ ` eq. 3
F2_LISA: float = 4/3 * FT_LISA


def N_acc(L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArr:
    r"""LISA acceleration noise
    $$N_\text{acc} = \frac{3 \cdot 10^{-15} \frac{\text{m}}{\text{s^2}}}{L}
    \approx 1.44 \cdot 10^{-48} \frac{1}{\text{s}^4 \text{Hz}}$$
    :gowling_2021:`\ ` eq. 3.3
    :gowling_2023:`\ ` p. 6

    $$4 {N}_\text{acc} \approx 5.76 \cdot 10^{-48} \frac{1}{\text{s}^4 \text{Hz}}$$
    :smith_2019:`\ ` eq. 53
    :lisa_sci_req:`\ ` eq. 3
    """
    return (3e-15 / L)**2


def N_AE(
        f: th.FloatOrArr,
        ft: th.FloatOrArr = FT_LISA,
        L: th.FloatOrArr = const.LISA_ARM_LENGTH,
        W_abs2: th.FloatOrArr | None = None) -> th.FloatOrArr:
    r"""A and E channels of LISA instrument noise
    $$N_A = N_E = \left(\left(
    4 + 2 \cos \left( \frac{f}{f_t} \right)\right) {P}_\text{oms} +
    8 \left( 1 + \cos \left( \frac{f}{f_t} \right) + \cos^2 \left( \frac{f}{f_t} \right) \right) {P}_\text{acc}
    \right) \lvert W \rvert^2$$
    :gowling_2021:`\ ` eq. 3.4
    :smith_2019:`\ ` eq. 57
    """
    cos_f_frac = np.cos(f/ft)
    if W_abs2 is None:
        W_abs2 = np.abs(W(f, ft))**2
    return ((4 + 2*cos_f_frac)*P_oms(L) + 8*(1 + cos_f_frac + cos_f_frac**2) * P_acc(f, L)) * W_abs2


def omega(f: th.FloatOrArr, S: th.FloatOrArr) -> th.FloatOrArr:
    r"""Convert an effective noise power spectral density (aka. sensitivity) $S$
    to a fractional GW energy density power spectrum $\Omega$
    $$\Omega = \frac{4 \pi^2}{3 H_0^2} f^3 S(f)$$
    :lisa_conventions:`\ ` eq. 167,
    :gowling_2021:`\ ` eq. 3.8,
    :gowling_2023:`\ ` eq. 3.8,
    :smith_2019:`\ ` eq. 59
    :maggiore_1999:`\ ` eq. 18

    However, there is a factor of 2 instead of a factor of 4 in
    :caprini_2020:`\ ` eq. 34
    """
    return 4*np.pi**2 / (3*const.H0_HZ**2) * f**3 * S


def omega_eb(f: th.FloatOrArr, f_ref_eb: float = 25, omega_ref_eb: float = 8.9e-10) -> th.FloatOrArr:
    r"""
    Energy density of extragalactic compact binaries
    $$\Omega_\text{eb}(f) = \Omega_\text{ref,eb} \left( \frac{f}{{f}_\text{ref,eb} \right)^\frac{2}{3}$$
    :gowling_2021:`\ ` eq. 3.9
    """
    return omega_ref_eb * (f/f_ref_eb)**(2/3)


def omega_gb(f: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Energy density of unresolved galactic compact binaries
    $$\Omega_\text{gb} = \left( \frac{4 \pi^2}{3 H_0^2} \right) f^3 {S}_\text{gb}(f)$$
    :gowling_2021:`\ ` eq. 3.11
    """
    return omega(f, S_gb(f))


def omega_ins(f: th.FloatOrArr) -> th.FloatOrArr:
    r"""LISA instrument noise
    $$\Omega_\text{ins} = \frac{4 \pi^2}{3 H_0^2} f^3 S_A(f)$$
    """
    return omega(f=f, S=S_AE(f))


def omega_noise(f: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Total energy density of noise
    $$\Omega_\text{noise} = \Omega_\text{ins} + \Omega_\text{eb} + \Omega_\text{gb}$$
    :gowling_2021:`\ ` eq. 3.13
    """
    return omega_ins(f) + omega_eb(f) + omega_gb(f)


def P_acc(f: th.FloatOrArr, L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArr:
    r"""
    LISA single test mass acceleration noise, $P_\text{acc}$
    :gowling_2021:`\ ` eq. 3.3
    :gowling_2023:`\ ` eq. 3.5
    :smith_2019:`\ ` eq. 52
    """
    return S_I(f, L) / (4 * (2*np.pi*f)**4)


def P_oms(L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArr:
    r"""
    LISA optical metrology noise $P_\text{oms}$, aka. $S_II$ or $S_s$
    $$P_\text{oms}(f) = \left( \frac{1.5 \cdot 10^{-11} \text{m}}{L} \right)^2 \text{Hz}^{-1}$$
    :gowling_2021:`\ ` eq. 3.2
    :lisa_sci_req:`\ ` eq. 3
    :smith_2019:`\ ` eq. 52, 54
    This is white noise and therefore independent of the frequency.
    Note that there is a typo on :gowling_2021:`\ ` p. 12:
    the correct $L = 2.5 \cdot 10^9 \text{m}$.
    For this $L$, $P_oms = 3.59 \cdot 10^{-41} Hz^{-1}$.
    """
    return (1.5e-11 / L)**2


def R_AE(f: th.FloatOrArr, ft: th.FloatOrArr = FT_LISA, W_abs2: th.FloatOrArr | None = None) -> th.FloatOrArr:
    r"""Gravitational wave response function for the A and E channels
    $$\mathcal{R}_A^\text{Fit} = \mathcal{R}_E^\text{Fit} \approx \frac{9}{20} \lvert W \rvert^2
    \left(1 + \left( \frac{3f}{4f_t} \right)^2 \right)^{-1}$$
    :gowling_2021:`\ ` eq. 3.6
    """
    if W_abs2 is None:
        W_abs2 = np.abs(W(f, ft))**2
    return 9/20 * W_abs2 / (1 + (3*f/(4*ft))**2)


def R_LISA(f: th.FloatOrArr, f2: th.FloatOrArr = F2_LISA) -> th.FloatOrArr:
    r"""Auxiliary function from LISA science requirements
    :lisa_sci_req:`\ ` eq. 3
    """
    return 1 + (f / f2)**2


def S(N: th.FloatOrArr, R: th.FloatOrArr) -> th.FloatOrArr:
    r"""Noise power spectral density
    $$S = \frac{N}{\mathcal{R}}$$
    :gowling_2021:`\ ` eq. 3.1
    """
    return N / R


def S_AE(
        f: th.FloatOrArr,
        ft: th.FloatOrArr = FT_LISA,
        L: th.FloatOrArr = const.LISA_ARM_LENGTH,
        both_channels: bool = True) -> th.FloatOrArr:
    r"""Noise power spectral density for the LISA A and E channels
    $$S_A = S_E = \frac{N_A}{\mathcal{R}_A}$$
    :gowling_2021:`\ ` eq. 3.7

    The factor of $\frac{1}{\sqrt{2}}$ for using both channels comes from :smith_2019:`\ ` eq. 59
    """
    # The W_abs2 cancels and can therefore be set to unity
    ret = S(N=N_AE(f=f, ft=ft, L=L, W_abs2=1), R=R_AE(f=f, ft=ft, W_abs2=1))
    if both_channels:
        return 1/np.sqrt(2) * ret
    return ret


def S_AE_approx(
        f: th.FloatOrArr,
        L: th.FloatOrArr = const.LISA_ARM_LENGTH,
        both_channels: bool = True) -> th.FloatOrArr:
    r"""Approximate noise power spectral density for the LISA A and E channels
    $$S_A = S_E = \frac{N_A}{\mathcal{R}_A}
    \approx \frac{40}{3} ({P}_\text{oms} + {4P}_\text{acc}) \left( 1 + \frac{3f}{4f_t} \right)^2$$
    :gowling_2021:`\ ` eq. 3.7
    :smith_2019:`\ ` eq. 63

    The factor of $\frac{1}{\sqrt{2}}$ for using both channels comes from :smith_2019:`\ ` eq. 59
    """
    ret = 40/3 * (P_oms(L) + 4*P_acc(f, L)) * (1 + (3*f/(4*ft(L)))**2)
    if both_channels:
        return 1/np.sqrt(2) * ret
    return ret


def S_I(f: th.FloatOrArr, L: th.FloatOrArr = const.LISA_ARM_LENGTH) -> th.FloatOrArr:
    r"""Subsidiary formula $S_I$ for acceleration noise
    :smith_2019:`\ ` eq. 53
    :lisa_sci_req:`\ ` eq. 3
    """
    return 4 * N_acc(L) * (1 + (const.F1_LISA/f)**2)


def S_gb(
        f: th.FloatOrArr,
        t: th.FloatOrArr = 4,  # years
        A: float = 1.8e-44) -> th.FloatOrArr:
    alpha = np.interp(t, GB_TIMES, GB_ALPHAS)
    beta = np.interp(t, GB_TIMES, GB_BETAS)
    kappa = np.interp(t, GB_TIMES, GB_KAPPAS)
    gamma = np.interp(t, GB_TIMES, GB_GAMMAS)
    fk = np.interp(t, GB_TIMES, GB_FKS)
    r"""Noise power spectral density for galactic binaries
    $$
    S_c(f) = A f^\frac{-7}{3} \exp \left( -f^\alpha + \beta f \sin(\kappa f) \right)
    \left( 1 + \tanh(\gamma (f_k - f) \right) \text{Hz}^{-1}
    $$
    :cornish_2017:`\ ` eq. 3
    :gowling_2021:`\ ` eq. 3.10
    """
    return A * f**(-7/3) * np.exp(-f**alpha + beta * f * np.sin(kappa * f)) * (1 + np.tanh(gamma * (fk - f)))


def W(f: th.FloatOrArr, ft: th.FloatOrArr) -> th.FloatOrArr:
    r"""Round trip modulation
    $$W(f,f_t) = 1 - e^{-2i \frac{f}{f_t}}$$
    :gowling_2021:`\ ` p. 12
    """
    return 1 - np.exp(-2j * f / ft)


#: Coefficients for the galactic binary noise, :cornish_2017:`\ ` table 1
GB_DATA: np.ndarray[tuple[int, int], np.float64] = np.array([
    [0.5, 1, 2, 4],
    [0.133, 0.171, 0.165, 0.138],
    [243, 292, 299, -221],
    [482, 1020, 611, 521],
    [917, 1680, 1340, 1680],
    [0.00258, 0.00215, 0.00173, 0.00113]
])
GB_TIMES: np.ndarray[tuple[int], np.float64] = GB_DATA[0, :]
GB_ALPHAS: np.ndarray[tuple[int], np.float64] = GB_DATA[1, :]
GB_BETAS: np.ndarray[tuple[int], np.float64] = GB_DATA[2, :]
GB_KAPPAS: np.ndarray[tuple[int], np.float64] = GB_DATA[3, :]
GB_GAMMAS: np.ndarray[tuple[int], np.float64] = GB_DATA[4, :]
GB_FKS: np.ndarray[tuple[int], np.float64] = GB_DATA[5, :]
