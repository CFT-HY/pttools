"""Utilities for calculating the noise of gravitational wave detectors, especially LISA."""

import numpy as np

from pttools.omgw0.const import H0_100_HZ, LISA_ARM_LENGTH, LISA_OBS_TIME, c
from pttools.speedup import njit
from pttools.type_hints import FloatArr1D, FloatArr2D, FloatOrArr

CACHE_H0_100_HZ: bool = True
"""
The functions in this module use Numba caching, but are dependent on the value of :py:data:pttools.omgw0.const.H0_HZ:.
If you change this value, you must clean Numba cache.
"""


@njit(cache=True)
def index_f_min(f: FloatArr1D, f_min: float | None = None) -> int:
    r"""Index of the first frequency bin of the band $[{f}_\text{min}, {f}_\text{max}]$.

    :param f: frequencies (Hz), sorted in ascending order
    :param f_min: minimum frequency to be considered (Hz), inclusive.
        If not provided, the band starts from the lowest available frequency.
    :return: index of the first frequency bin of the band
    """
    if f_min is None:
        return 0
    return int(np.argmax(f >= f_min))


@njit(cache=True)
def index_f_max(f: FloatArr1D, f_max: float | None = None) -> int:
    r"""Index one past the last frequency bin of the band $[{f}_\text{min}, {f}_\text{max}]$,
    to be used as the exclusive end of a slice.

    :param f: frequencies (Hz), sorted in ascending order
    :param f_max: maximum frequency to be considered (Hz), inclusive.
        If not provided, the band ends at the highest available frequency.
    :return: index one past the last frequency bin of the band
    """
    if f_max is None:
        return f.size
    # The bin at f_max is included in the band, and therefore the first bin above it is the end of the slice.
    if f_max >= f[-1]:
        return f.size
    return int(np.argmax(f > f_max))


@njit(cache=CACHE_H0_100_HZ, nogil=True)
def signal_to_noise_ratio(
        f: FloatArr1D,
        signal: FloatArr1D,
        noise: FloatArr1D | None = None,
        f_noise: FloatArr1D | None = None,
        obs_time: float = LISA_OBS_TIME,
        f_min: float | None = None,
        f_max: float | None = None,
        noise_eb: bool = True,
        noise_gb: bool = True,
        noise_ins: bool = True) -> tuple[float, FloatArr1D, FloatArr1D]:
    r"""Signal-to-noise ratio
    $$\rho = \sqrt{T_{\text{obs}} \int_{{f}_\text{min}}^{{f}_\text{max}} df \left( \frac{
    h^2 \Omega_{\text{signal}}}{
    h^2 \Omega_{\text{noise}}} \right)^2}$$
    :caprini_2020:`\ ` eq. 33
    :smith_2019:`\ ` eq. 60.

    The equation :gowling_2021:`\ ` eq. 3.12 has unusual powers for $h$,
    but those cancel out, giving the same results.

    The equation :gowling_2023:`\ ` eq. 3.9 has an additional factor of 2,
    which is canceled out by another factor of 2 in eq. 3.8.

    :param f: frequencies (Hz)
    :param signal: $\Omega_\text{signal} h^2$
    :param noise: $\Omega_\text{noise} h^2$
    :param f_noise: frequencies for the noise (assumed to be the same as for the signal, if not provided)
    :param obs_time: observation time (s)
    :param f_min: minimum frequency to be considered (Hz), inclusive.
        If not provided, the integration starts from the lowest available frequency.
    :param f_max: maximum frequency to be considered (Hz), inclusive.
        If not provided, the integration ends at the highest available frequency.
    :param noise_eb: whether to generate extragalactic compact binary noise when noise is not provided
    :param noise_gb: whether to generate galactic compact binary noise when noise is not provided
    :param noise_ins: whether to generate instrument noise when noise is not provided
    :return: signal-to-noise ratio SNR, aka. $\rho$
    """
    if f_noise is None:
        i_f_min = index_f_min(f, f_min)
        i_f_max = index_f_max(f, f_max)
        f2 = f[i_f_min:i_f_max]

        noise2 = omega_noise_h2(f=f2, eb=noise_eb, gb=noise_gb, ins=noise_ins) \
            if noise is None else noise[i_f_min:i_f_max]
        signal = signal[i_f_min:i_f_max]
    else:
        f_min2 = max(f[0], f_noise[0]) if f_min is None else f_min
        f_max2 = min(f[-1], f_noise[-1]) if f_max is None else f_max
        i_f_min = index_f_min(f_noise, f_min2)
        i_f_max = index_f_max(f_noise, f_max2)

        f2 = f_noise[i_f_min:i_f_max]
        noise2 = omega_noise_h2(f=f2, eb=noise_eb, gb=noise_gb, ins=noise_ins) \
            if noise is None else noise[i_f_min:i_f_max]
        # The NumPy stubs do not know that the output of np.interp() is an array when the input is an array.
        signal = 10.**np.interp(np.log10(f2), np.log10(f), np.log10(signal))  # pyrefly: ignore[bad-assignment]

    # The NumPy stubs do not know that np.trapezoid() returns a scalar for a 1D array.
    snr: float = np.sqrt(obs_time * np.trapezoid(signal**2 / noise2**2, f2))  # pyrefly: ignore[bad-assignment]
    return snr, f2, noise2


@njit(cache=True)
def ft[T: FloatOrArr](L: T = LISA_ARM_LENGTH) -> T:
    r"""Transfer frequency
    $$f_t = \frac{c}{2\pi L}$$
    :gowling_2021:`\ ` p. 12.
    """
    # typing.cast() is not used below, since Numba cannot compile it.
    return c / (2*np.pi*L)  # pyrefly: ignore[bad-return]

#: Default LISA transfer frequency $f_t$
FT_LISA: float = ft()
#: :lisa_sci_req:`\ ` eq. 3 (Hz)
F1_LISA: float = 4e-4
#: $f_2$ from :lisa_sci_req:`\ ` eq. 3
F2_LISA: float = 4/3 * FT_LISA


@njit(cache=True)
def N_acc[T: FloatOrArr](L: T = LISA_ARM_LENGTH) -> T:
    r"""LISA acceleration noise
    $${N}_\text{acc} = \frac{3 \cdot 10^{-15}}{L} \frac{\text{m}}{\text{s}^2}
    \approx 1.44 \cdot 10^{-48} \frac{1}{\text{s}^4 \text{Hz}}$$
    :gowling_2021:`\ ` eq. 3.3
    :gowling_2023:`\ ` p. 6.

    $$4 {N}_\text{acc} \approx 5.76 \cdot 10^{-48} \frac{1}{{\text{s}}^4 \text{Hz}}$$
    :smith_2019:`\ ` eq. 53
    :lisa_sci_req:`\ ` eq. 3
    """
    return (3e-15 / L)**2  # pyrefly: ignore[bad-return]


@njit(cache=CACHE_H0_100_HZ)
def N_AE[T: FloatOrArr](
        f: T,
        ft: T | float = FT_LISA,
        L: T | float = LISA_ARM_LENGTH,
        W_abs2: T | float | None = None) -> T:
    r"""A and E channels of LISA instrument noise
    $$N_A = N_E = \left(\left(
    4 + 2 \cos \left( \frac{f}{f_t} \right)\right) {P}_\text{oms} +
    8 \left( 1 + \cos \left( \frac{f}{f_t} \right) + \cos^2 \left( \frac{f}{f_t} \right) \right) {P}_\text{acc}
    \right) \lvert W \rvert^2$$
    :gowling_2021:`\ ` eq. 3.4
    :smith_2019:`\ ` eq. 57.
    """
    cos_f_frac = np.cos(f/ft)
    if W_abs2 is None:
        W_abs2 = np.abs(W(f, ft))**2  # pyrefly: ignore[bad-assignment]
    return ((4 + 2*cos_f_frac)*P_oms(L) + 8*(1 + cos_f_frac + cos_f_frac**2) * P_acc(f, L)) * W_abs2


@njit(cache=CACHE_H0_100_HZ)
def omega_h2[T: FloatOrArr](f: T, S: T | float) -> T:
    r"""Convert an effective noise power spectral density (aka. sensitivity) $S$
    to a fractional GW energy density power spectrum $\Omega$.

    $$\Omega h^2 = \frac{4 \pi^2}{3 H_{100}^2} f^3 S(f)$$
    This is adapted from
    $$\Omega = \frac{4 \pi^2}{3 H_0^2} f^3 S(f)$$
    :lisa_conventions:`\ ` eq. 167,
    :gowling_2021:`\ ` eq. 3.8,
    :gowling_2023:`\ ` eq. 3.8,
    :smith_2019:`\ ` eq. 59
    :maggiore_1999:`\ ` eq. 18.

    However, there is a factor of 2 instead of a factor of 4 in
    :caprini_2020:`\ ` eq. 34
    """
    return 4*np.pi**2 / (3 * H0_100_HZ**2) * f**3 * S  # pyrefly: ignore[bad-return]


#: $\Omega_\text{ref,eb}
#: `abbott_2019`:`\ ` p. 4
OMEGA_REF_EB: float = 8.9e-10
#: $\Omega_\text{ref,eb} h^2$
#: `abbott_2019`:`\ `, using the value of $H_0 = 67.9 \frac{\text{km}}{\text{s Mpc}}$ from the article.
OMEGA_REF_EB_H2: float = OMEGA_REF_EB * 0.679**2


@njit(cache=True)
def omega_eb_h2[T: FloatOrArr](f: T, f_ref_eb: float = 25, omega_ref_eb_h2: float = OMEGA_REF_EB_H2) -> T:
    r"""
    Energy density of extragalactic compact binaries
    $$\Omega_\text{eb}(f) = \Omega_\text{ref,eb} \left( \frac{f}{{f}_\text{ref,eb}} \right)^\frac{2}{3}$$
    :gowling_2021:`\ ` eq. 3.9.
    """
    return omega_ref_eb_h2 * (f/f_ref_eb)**(2/3)  # pyrefly: ignore[bad-return]


@njit(cache=CACHE_H0_100_HZ)
def omega_gb_h2[T: FloatOrArr](f: T) -> T:
    r"""
    Energy density of unresolved galactic compact binaries
    $$\Omega_\text{gb} = \left( \frac{4 \pi^2}{3 H_{100}^2} \right) f^3 {S}_\text{gb}(f)$$
    :gowling_2021:`\ ` eq. 3.11.
    """
    return omega_h2(f=f, S=S_gb(f))


@njit(cache=CACHE_H0_100_HZ)
def omega_ins_h2[T: FloatOrArr](f: T) -> T:
    r"""LISA instrument noise
    $$\Omega_\text{ins} = \frac{4 \pi^2}{3 H_{100}^2} f^3 S_A(f)$$.
    """
    return omega_h2(f=f, S=S_AE(f))


@njit(cache=CACHE_H0_100_HZ)
def omega_noise_h2[T: FloatOrArr](f: T, eb: bool = True, gb: bool = True, ins: bool = True) -> T:
    r"""
    Total energy density of LISA noise
    $$\Omega_\text{noise} h^2 = \left( \Omega_\text{ins} + \Omega_\text{eb} + \Omega_\text{gb} \right) h^2$$
    :gowling_2021:`\ ` eq. 3.13.
    """
    om = np.zeros_like(f)
    if ins:
        om += omega_ins_h2(f)
    if eb:
        om += omega_eb_h2(f)
    if gb:
        om += omega_gb_h2(f)
    return om  # pyrefly: ignore[bad-return]


@njit(cache=True)
def P_acc[T: FloatOrArr](f: T, L: T | float = LISA_ARM_LENGTH) -> T:
    r"""
    LISA single test mass acceleration noise, $P_\text{acc}$
    :gowling_2021:`\ ` eq. 3.3
    :gowling_2023:`\ ` eq. 3.5
    :smith_2019:`\ ` eq. 52.
    """
    return S_I(f, L) / (4 * (2 * np.pi * f)**4)  # pyrefly: ignore[bad-return]


@njit(cache=True)
def P_oms[T: FloatOrArr](L: T = LISA_ARM_LENGTH) -> T:
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
    return (1.5e-11 / L)**2  # pyrefly: ignore[bad-return]


@njit(cache=True)
def R_AE[T: FloatOrArr](f: T, ft: T | float = FT_LISA, W_abs2: T | float | None = None) -> T:
    r"""Gravitational wave response function for the A and E channels
    $$\mathcal{R}_A^\text{Fit} = \mathcal{R}_E^\text{Fit} \approx \frac{9}{20} \lvert W \rvert^2
    \left(1 + \left( \frac{3f}{4f_t} \right)^2 \right)^{-1}$$
    :gowling_2021:`\ ` eq. 3.6.
    """
    # A separate variable is used, as assigning to W_abs2 would not narrow away its None type.
    w_abs2: FloatOrArr = np.abs(W(f, ft))**2 if W_abs2 is None else W_abs2
    return 9/20 * w_abs2 / (1 + (3*f/(4*ft))**2)  # pyrefly: ignore[bad-return]


@njit(cache=True)
def R_LISA[T: FloatOrArr](f: T, f2: T | float = F2_LISA) -> T:
    r"""Auxiliary function from LISA science requirements
    :lisa_sci_req:`\ ` eq. 3.
    """
    return 1 + (f / f2)**2  # pyrefly: ignore[bad-return]


@njit(cache=True)
def S[T: FloatOrArr](N: T, R: T | float) -> T:
    r"""Noise power spectral density
    $$S = \frac{N}{\mathcal{R}}$$
    :gowling_2021:`\ ` eq. 3.1.
    """
    return N / R  # pyrefly: ignore[bad-return]


@njit(cache=True)
def S_AE[T: FloatOrArr](
        f: T,
        ft: T | float = FT_LISA,
        L: T | float = LISA_ARM_LENGTH,
        both_channels: bool = True) -> T:
    r"""Noise power spectral density for the LISA A and E channels
    $$S_A = S_E = \frac{N_A}{\mathcal{R}_A}$$
    :gowling_2021:`\ ` eq. 3.7.

    The factor of $\frac{1}{\sqrt{2}}$ for using both channels comes from :smith_2019:`\ ` eq. 59
    """
    # The W_abs2 cancels and can therefore be set to unity
    ret = S(N=N_AE(f=f, ft=ft, L=L, W_abs2=1), R=R_AE(f=f, ft=ft, W_abs2=1))
    if both_channels:
        return 1/np.sqrt(2) * ret
    return ret  # pyrefly: ignore[bad-return]


@njit(cache=True)
def S_AE_approx[T: FloatOrArr](
        f: T,
        L: T | float = LISA_ARM_LENGTH,
        both_channels: bool = True) -> T:
    r"""Approximate noise power spectral density for the LISA A and E channels
    $$S_A = S_E = \frac{N_A}{\mathcal{R}_A}
    \approx \frac{40}{3} ({P}_\text{oms} + {4P}_\text{acc}) \left( 1 + \frac{3f}{4f_t} \right)^2$$
    :gowling_2021:`\ ` eq. 3.7
    :smith_2019:`\ ` eq. 63.

    The factor of $\frac{1}{\sqrt{2}}$ for using both channels comes from :smith_2019:`\ ` eq. 59
    """
    ret = 40/3 * (P_oms(L) + 4*P_acc(f, L)) * (1 + (3*f/(4*ft(L)))**2)
    if both_channels:
        return 1/np.sqrt(2) * ret
    return ret  # pyrefly: ignore[bad-return]


@njit(cache=True)
def S_I[T: FloatOrArr](f: T, L: T | float = LISA_ARM_LENGTH) -> T:
    r"""Subsidiary formula $S_I$ for acceleration noise
    :smith_2019:`\ ` eq. 53
    :lisa_sci_req:`\ ` eq. 3.
    """
    return 4 * N_acc(L) * (1 + (F1_LISA/f)**2)  # pyrefly: ignore[bad-return]


@njit(cache=True)
def S_gb[T: FloatOrArr](
        f: T,
        t: T | float = 4,  # years
        A: float = 1.8e-44) -> T:
    r"""Noise power spectral density for galactic binaries
    $$S_c(f) = A f^\frac{-7}{3} \exp \left( -f^\alpha + \beta f \sin(\kappa f) \right)
    \left( 1 + \tanh(\gamma (f_k - f) \right) \text{Hz}^{-1}$$
    :cornish_2017:`\ ` eq. 3
    :gowling_2021:`\ ` eq. 3.10.
    """
    alpha = np.interp(t, GB_TIMES, GB_ALPHAS)
    beta = np.interp(t, GB_TIMES, GB_BETAS)
    kappa = np.interp(t, GB_TIMES, GB_KAPPAS)
    gamma = np.interp(t, GB_TIMES, GB_GAMMAS)
    fk = np.interp(t, GB_TIMES, GB_FKS)
    return A * f**(-7/3) * np.exp(-f**alpha + beta * f * np.sin(kappa * f)) * (1 + np.tanh(gamma * (fk - f)))


@njit(cache=True)
def W[T: FloatOrArr](f: T, ft: T | float) -> T:
    r"""Round trip modulation
    $$W(f,f_t) = 1 - e^{-2i \frac{f}{f_t}}$$
    :gowling_2021:`\ ` p. 12.
    """
    return 1 - np.exp(-2j * f / ft)  # pyrefly: ignore[bad-return]


#: Coefficients for the galactic binary noise, :cornish_2017:`\ ` table 1
GB_DATA: FloatArr2D = np.array([
    [0.5, 1, 2, 4],
    [0.133, 0.171, 0.165, 0.138],
    [243, 292, 299, -221],
    [482, 1020, 611, 521],
    [917, 1680, 1340, 1680],
    [0.00258, 0.00215, 0.00173, 0.00113]
])
GB_TIMES: FloatArr1D = GB_DATA[0, :]
GB_ALPHAS: FloatArr1D = GB_DATA[1, :]
GB_BETAS: FloatArr1D = GB_DATA[2, :]
GB_KAPPAS: FloatArr1D = GB_DATA[3, :]
GB_GAMMAS: FloatArr1D = GB_DATA[4, :]
GB_FKS: FloatArr1D = GB_DATA[5, :]
