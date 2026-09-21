r"""Frequency conversion functions for $\Omega_{\text{gw},0}$."""

from pttools.omgw0.const import DEFAULT_G_STAR, DEFAULT_T_STAR, F_STAR0_REF
from pttools.ssm.const import DEFAULT_R_STAR
import pttools.type_hints as th


def f(z: th.FloatOrArr, r_star: th.FloatOrArr, f_star0: th.FloatOrArr) -> th.FloatOrArr:
    r"""Convert the dimensionless wavenumber $z$ to frequency today by taking into account the redshift.
    $$f = \frac{z}{{r}_\ast} f_{\ast,0}$$,
    :gowling_2021:`\ ` eq. 2.12
    :gowling_2023:`\ ` eq. 2.8.

    :param z: $z$, dimensionless wavenumber
    :param r_star: $r_*$, Hubble-scaled mean bubble spacing
    :param f_star0: $f_{\ast,0}$
    :return: frequency $f$ today
    """
    return z / r_star * f_star0


def f0(
        r_star: th.FloatOrArr,
        T_star: th.FloatOrArr = DEFAULT_T_STAR,
        g_star: th.FloatOrArr = DEFAULT_G_STAR) -> th.FloatOrArr:
    r"""$f_0$, factor required to take into account the redshift of the frequency scale.

    $$f_0 = \frac{f_{\ast,0}}{r_{\ast}}$$

    :param r_star: $r_\ast$
    :param T_star: $T_\ast$
    :param g_star: $g_\ast$
    :return: $f_0$
    """
    return f_star0(T_star, g_star) / r_star


def f_star0(
        T_star: th.FloatOrArr,
        g_star: th.FloatOrArr = DEFAULT_G_STAR,
        f_star0_ref: float = F_STAR0_REF) -> th.FloatOrArr:
    r"""
    $f_{\ast,0}$, conversion factor between the frequencies at the time of the GW formation and frequencies today.

    $$f_{\ast,0} = f_{\ast,0,\text{ref}}
    \left( \frac{T_n}{100 \text{GeV}} \right)
    \left( \frac{{g}_\ast}{100} \right)^{\frac{1}{6}} \text{Hz}$$,
    :croon_2024:`\ `, eq. 38,
    :caprini_2020:`\ ` eq. 31,
    :gowling_2021:`\ ` eq. 2.13,
    :gowling_2023:`\ ` eq. 2.9.

    :param T_star: $T_\ast$, temperature at the time of GW production
    :param g_star: $g_\ast$, degrees of freedom at the time the GWs were produced.
        The default value is from the article.
    :param f_star0_ref: The constant $f_{\ast,0,\text{ref}}$ in the front of the formula
    :return: $f_{\ast,0}$
    """
    return f_star0_ref * (T_star / 100) * (g_star / 100)**(1 / 6)


def z(
        f: th.FloatOrArr,
        T_star: th.FloatOrArr = DEFAULT_T_STAR,
        r_star: th.FloatOrArr = DEFAULT_R_STAR,
        g_star: th.FloatOrArr = DEFAULT_G_STAR) -> th.FloatOrArr:
    r"""Convert from frequencies $f$ back to wavenumbers $z$.

    $$z(f) = \frac{f}{f_{\ast,0}} {r}_\ast$$
    Inverted from :gowling_2021:`\ ` eq. 2.12

    :param f: frequencies $f$ today
    :param T_star: Temperature $T_*$ at the time of GW production
    :param g_star: Degrees of freedom at the time the GWs were produced. The default value is from the article.
    :param r_star: Hubble-scaled mean bubble spacing
    :return: wavenumbers $z$
    """
    return f / f_star0(T_star=T_star, g_star=g_star) * r_star
