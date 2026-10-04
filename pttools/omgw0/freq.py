r"""Frequency conversion functions for $\Omega_{\text{gw},0}$."""

import typing as tp

import numpy as np

from pttools.omgw0.const import DEFAULT_G_STAR, DEFAULT_T_STAR, GS0, T_CMB
from pttools.omgw0.factors import H_SI, a_ratio, e_SI
from pttools.ssm.const import DEFAULT_R_STAR
from pttools.type_hints import FloatOrArr


def f[T: FloatOrArr](z: T, r_star: T | float, f_star0: T | float) -> T:
    r"""Convert the dimensionless wavenumber $z$ to frequency today by taking into account the redshift.

    $$f = \frac{z}{{r}_\ast} f_{\ast,0}$$,
    :gowling_2021:`\ ` eq. 2.12
    :gowling_2023:`\ ` eq. 2.8.

    :param z: $z$, dimensionless wavenumber
    :param r_star: $r_*$, Hubble-scaled mean bubble spacing
    :param f_star0: $f_{\ast,0}$
    :return: frequency $f$ today
    """
    return z / r_star * f_star0  # pyrefly: ignore[bad-return]


def f0[T: FloatOrArr](
        r_star: T,
        T_star: T | float = DEFAULT_T_STAR,
        ge_star: T | float = DEFAULT_G_STAR,
        gs_star: T | float | None = None) -> T:
    r"""$f_0$, factor required to take into account the redshift of the frequency scale.

    $$f_0 = \frac{f_{\ast,0}}{r_{\ast}}$$

    :param r_star: $r_\ast$, Hubble-scaled mean bubble spacing
    :param T_star: $T_\ast$, temperature at the time of GW production in GeV
    :param ge_star: $g_{e\ast}$, degrees of freedom for energy density at the time of GW production
    :param gs_star: $g_{s\ast}$, degrees of freedom for entropy at the time of GW production.
        If not given, $g_{s\ast} = g_{e\ast}$ is assumed.
    :return: $f_0$
    """
    return f_star0(T_star=T_star, ge_star=ge_star, gs_star=gs_star) / r_star  # pyrefly: ignore[bad-return]


def f_star0[T: FloatOrArr](
        T_star: T,
        ge_star: T | float = DEFAULT_G_STAR,
        gs_star: T | float | None = None,
        gs0: float = GS0,
        T0: float = T_CMB) -> T:
    r"""$f_{\ast,0}$, conversion factor from frequencies in units of $H_\ast$ at GW production to frequencies today.

    $$f_{\ast,0} = \frac{H_{\ast}}{2\pi} \frac{a_{\ast}}{a_{0}}$$
    The Hubble rate follows from the Friedmann equation,
    $$H_{\ast} = \sqrt{\frac{8 \pi G}{3 c^2} e_{\ast}}, \quad
    e_{\ast} = \frac{\pi^2}{30} g_{e\ast} \frac{(k_{\text{B}} T_{\ast})^4}{(\hbar c)^3},$$
    where $e_\ast$ is the energy density at the time of GW production,
    and the redshift from the conservation of entropy,
    $$\frac{a_{\ast}}{a_{0}} = \frac{T_{0}}{T_{\ast}} \left( \frac{g_{s0}}{g_{s\ast}} \right)^\frac{1}{3}.$$
    These are the same assumptions as in :py:func:`pttools.omgw0.factors.F_gw0_h2`.
    The frequency today is then $f = \frac{z}{r_{\ast}} f_{\ast,0}$, see :py:func:`f`.
    The Hubble rate brings in $g_{e\ast}$ and the redshift $g_{s\ast}$,
    whereas the degrees of freedom for pressure do not appear.

    When $g_{s\ast} = g_{e\ast}$, this reduces to the form used in the literature,
    $$f_{\ast,0} = f_{\ast,0,\text{ref}}
    \left( \frac{T_{\ast}}{100 \text{ GeV}} \right)
    \left( \frac{g_{e\ast}}{100} \right)^\frac{1}{6},$$
    where $f_{\ast,0,\text{ref}}$ is $f_{\ast,0}$ at $T_\ast = 100 \text{ GeV}$ and $g_\ast = 100$.
    With $g_{s0} = 3.9298$ and $T_0 = 2.72548 \text{K}$,
    $f_{\ast,0,\text{ref}} = 2.625 \cdot 10^{-6} \text{Hz}$.
    The articles use the rounded value $f_{\ast,0,\text{ref}} = 2.6 \cdot 10^{-6} \text{Hz}$,
    and the definition of $g_\ast$ varies between them:

    - :croon_2024:`\ ` eq. 38, where eq. 32 defines $g_\ast \equiv g_{e\ast}$.
      They use $f_{\ast,0,\text{ref}} = 2.7 \cdot 10^{-6} \text{Hz}$ and have an additional factor $\frac{1}{B}$.
    - :gowling_2021:`\ ` eq. 2.13 and :gowling_2023:`\ ` eq. 2.9,
      where p. 7 defines $g_\ast \equiv g_{p\ast}$.
    - :caprini_2020:`\ ` eq. 31, where p. 11 states that "$g_\ast$ is the effective number of relativistic
      degrees of freedom after the PT during which the GWs are produced".
    - :hindmarsh_2017:`\ ` uses the form
      $$f_{p,0} \approx 26 \frac{1}{H_n R_{\ast}} \frac{z_p}{10} \frac{T_n}{100 \text{ GeV}}
      \left( \frac{h_{\ast}}{100} \right)^\frac{1}{6} \mu\text{Hz},$$
      where $h_\ast \equiv g_{s\ast}$.

    :param T_star: $T_\ast$, temperature at the time of GW production in GeV
    :param ge_star: $g_{e\ast}$, degrees of freedom for energy density at the time of GW production
    :param gs_star: $g_{s\ast}$, degrees of freedom for entropy at the time of GW production.
        If not given, $g_{s\ast} = g_{e\ast}$ is assumed.
    :param gs0: $g_{s0}$, degrees of freedom for entropy today
    :param T0: $T_0$, temperature of the CMB today in K
    :return: $f_{\ast,0}$ in Hz
    """
    if gs_star is None:
        gs_star = ge_star
    return tp.cast(T,
        H_SI(e=e_SI(T=T_star, ge=ge_star)) / (2 * np.pi) * a_ratio(T_star=T_star, T0=T0, gs_star=gs_star, gs0=gs0)
    )


def z[T: FloatOrArr](
        f: T,
        T_star: T | float = DEFAULT_T_STAR,
        r_star: T | float = DEFAULT_R_STAR,
        ge_star: T | float = DEFAULT_G_STAR,
        gs_star: T | float | None = None) -> T:
    r"""Convert from frequencies $f$ back to wavenumbers $z$.

    $$z(f) = \frac{f}{f_{\ast,0}} {r}_\ast$$
    Inverted from :gowling_2021:`\ ` eq. 2.12

    :param f: frequencies $f$ today
    :param T_star: $T_\ast$, temperature at the time of GW production in GeV
    :param r_star: $r_\ast$, Hubble-scaled mean bubble spacing
    :param ge_star: $g_{e\ast}$, degrees of freedom for energy density at the time of GW production
    :param gs_star: $g_{s\ast}$, degrees of freedom for entropy at the time of GW production.
        If not given, $g_{s\ast} = g_{e\ast}$ is assumed.
    :return: wavenumbers $z$
    """
    return f / f_star0(T_star=T_star, ge_star=ge_star, gs_star=gs_star) * r_star  # pyrefly: ignore[bad-return]
