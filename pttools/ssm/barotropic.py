"""Formulae for a barotropic equation of state"""

from pttools.bubble.const import DEFAULT_NU_GDH2024
from pttools.ssm.const import DEFAULT_A_STAR_A_R_RATIO, DEFAULT_N_SH, DEFAULT_R_STAR
from pttools.type_hints import FloatOrArr


def dilution_of_e(
        a_star_a_r_ratio: FloatOrArr = DEFAULT_A_STAR_A_R_RATIO,
        nu: FloatOrArr = DEFAULT_NU_GDH2024) -> FloatOrArr:
    r"""Dilution of the background energy density $\bar{e}$
    $$\left( \frac{a_*}{a_r} \right)^\frac{2 \nu}{1 + \nu} = \left( \frac{a_*}{a} \right)^4 \frac{\bar{e_*}}{\bar{e}}$$
    :giombi_2024_cs:`\ ` eq. 2.18

    The FLRW scale factor $a$ is defined as
    $$a(\eta) \d\eta = dt$$
    :giombi_2024_cs:`\ ` p. 3

    :param a_star_a_r_ratio: $\frac{a_*}{a_r}$
    :param nu: $\nu_\text{gdh2024}$
    :return: Dilution of the background energy density $\bar{e}$
    """
    return a_star_a_r_ratio ** (2 * nu / (1 + nu))


def eta_ratio(
        ubarf: FloatOrArr,
        r_star: FloatOrArr = DEFAULT_R_STAR,
        N_sh: FloatOrArr = DEFAULT_N_SH,
        nu: FloatOrArr = DEFAULT_NU_GDH2024) -> FloatOrArr:
    r"""Ratio of conformal times $\frac{\Delta \eta_\text{v}}{\eta_*}$ for a barotropic EoS
    $$\frac{\Delta \eta_\text{v}}{\eta_*}
    = \frac{N_{\text{sh}} \eta_\text{sh}}{\eta_*}
    = \frac{N_{\text{sh}} r_*}{(1 + \nu_\text{gdh2024}) \bar{U}_f},$$
    where we have used
    $\eta_\text{sh} \approx \frac{R_*}{\bar{U}_f}$,
    $r_* \equiv H_* R_*$
    and
    $H_* \eta_* = 1 + \nu_\text{gdh2024}$.
    """
    return N_sh * r_star / ((1 + nu) * ubarf)


def H_eta[T: FloatOrArr](nu: T = DEFAULT_NU_GDH2024) -> T:
    r"""$H \eta$ for a barotropic EoS
    $$H \eta = \frac{\dot{a}}{a} = 1 + \nu_\text{gdh2024} = \frac{2}{1 + 3 \omega}$$

    This comes from the scale factor for barotropic EoS
    $$a(\eta) = a(\eta_*) \left( \frac{\eta}{\eta_*} \right)^\frac{2}{1+3\omega}$$
    :giombi_2024_cs:`\ ` p. 5
    and that
    $$\frac{2}{1+3\omega} = 1 + \nu_\text{gdh2024}$$
    """
    return 1 + nu


def l[T: FloatOrArr](nu: T = DEFAULT_NU_GDH2024) -> T:
    r"""$\ell(\nu)
    $$\ell(\nu) = 1 + 2\nu$$
    :giombi_2026:`\ ` p. 25
    """
    return 1 + 2 * nu


def source_lifetime_factor(
        ubarf: FloatOrArr,
        r_star: FloatOrArr = DEFAULT_R_STAR,
        N_sh: FloatOrArr = DEFAULT_N_SH,
        nu: FloatOrArr = DEFAULT_NU_GDH2024):
    r"""
    Source lifetime factor $\Upsilon_\ell$
    $$\Upsilon_\ell \equiv
    \frac{1}{\ell(\nu)} \left(1 - \left( \frac{\eta_*}{\eta_\text{end}} \right)^{\ell(\nu)} \right)
    = \frac{1}{\ell(\nu)} \left(1 - \left(1 + \frac{\Delta \eta_\text{v}}{\eta_*} \right)^{-\ell(\nu)} \right)$$
    :giombi_2026:`\ ` eq. 3.6,
    :giombi_2024_cs:`\ ` eq. 3.13

    This is an updated version of
    :maki_msc:`\ ` eq. 3.79
    """
    return -Upsilon(r=1 + eta_ratio(ubarf=ubarf, r_star=r_star, N_sh=N_sh, nu=nu), l=-l(nu))


def Upsilon(r: FloatOrArr, l: FloatOrArr) -> FloatOrArr:
    r"""$\Upsilon_\ell$ for arbitrary $\ell$
    $$\Upsilon_\ell (r) = \frac{1}{\ell} \left( 1 - r^\ell \right)$$
    :giombi_2026:`\ ` eq. 3.6
    """
    return (1 - r**l) / l
