r"""Frequency conversion functions for $\Omega_{\text{gw},0}$"""

from pttools.omgw0 import const
import pttools.type_hints as th


def f(z: th.FloatOrArr, r_star: th.FloatOrArr, f_star0: th.FloatOrArr) -> th.FloatOrArr:
    r"""Convert the dimensionless wavenumber $z$ to frequency today by taking into account the redshift.
    $$f = \frac{z}{r_*} f_{*,0}$$,
    :gowling_2021:`\ ` eq. 2.12

    :param z: dimensionless wavenumber $z$
    :param r_star: Hubble-scaled mean bubble spacing
    :return: frequency $f$ today
    """
    return z/r_star * f_star0


def f0(rs: th.FloatOrArr, T_n: th.FloatOrArr = const.T_DEFAULT, g_star: float = 100) -> th.FloatOrArr:
    r"""Factor required to take into account the redshift of the frequency scale"""
    return f_star0(T_n, g_star) / rs


def f_star0(Tn: th.FloatOrArr, g_star: th.FloatOrArr = 100) -> th.FloatOrArr:
    r"""
    Conversion factor between the frequencies at the time of the nucleation and frequencies today.
    $$f_{*,0} = 2.6 \cdot 10^{-6} \text{Hz}
    \left( \frac{T_n}{100 \text{GeV}} \right)
    \left( \frac{g_*}{100} \right)^{\frac{1}{6}}$$,
    :gowling_2021:`\ ` eq. 2.13

    :param Tn: Nucleation temperature
    :param g_star: Degrees of freedom at the time the GWs were produced. The default value is from the article.
    :return: $f_{*,0}$
    """
    return const.fs0_ref * (Tn / 100) * (g_star / 100)**(1 / 6)
