"""Formulae for a barotropic equation of state."""

from pttools.bubble.const import DEFAULT_NU_GDH2024
from pttools.ssm.const import DEFAULT_A_STAR_A_R_RATIO, DEFAULT_N_SH, DEFAULT_R_STAR
from pttools.type_hints import FloatOrArr


def dilution_of_e[T: FloatOrArr](
        a_star_a_r_ratio: T = DEFAULT_A_STAR_A_R_RATIO,
        nu: T | float = DEFAULT_NU_GDH2024) -> T:
    r"""Dilution of the background energy density $\bar{e}$
    $$\left( \frac{a_*}{a_r} \right)^\frac{2 \nu}{1 + \nu} = \left( \frac{a_*}{a} \right)^4 \frac{\bar{e_*}}{\bar{e}}$$
    :giombi_2024_cs:`\ ` eq. 2.18.

    The FLRW scale factor $a$ is defined as
    $$a(\eta) \d\eta = dt$$
    :giombi_2024_cs:`\ ` p. 3

    :param a_star_a_r_ratio: $\frac{a_*}{a_r}$
    :param nu: $\nu_\text{gdh2024}$
    :return: Dilution of the background energy density $\bar{e}$
    """
    return a_star_a_r_ratio ** (2 * nu / (1 + nu))  # pyrefly: ignore[bad-return]


def eta_ratio[T: FloatOrArr](
        ubarf: T,
        r_star: T | float = DEFAULT_R_STAR,
        N_sh: T | float = DEFAULT_N_SH,
        nu: T | float = DEFAULT_NU_GDH2024) -> T:
    r"""Source duration in units of the conformal time at the start of the acoustic phase,
    $\frac{\Delta \eta_\text{v}}{\eta_*}$, for a barotropic EoS.

    $$\frac{\Delta \eta_\text{v}}{\eta_*}
    = \frac{N_{\text{sh}} \eta_\text{sh}}{\eta_*}
    = \frac{N_{\text{sh}} \mathcal{H}_* \eta_\text{sh}}{\mathcal{H}_* \eta_*}
    = \frac{N_{\text{sh}} r_*}{(1 + \nu_\text{gdh2024}) \bar{U}_f},$$
    where $\Delta \eta_\text{v} \equiv \eta_\text{end} - \eta_* = N_{\text{sh}} \eta_\text{sh}$
    (:giombi_2026:`\ ` eq. 4.1, :giombi_2024_cs:`\ ` eq. 4.1),
    and we have used
    $\eta_\text{sh} \approx \frac{R_*}{\bar{U}_f}$ (see :py:func:`pttools.ssm.scaling.H_star_eta_sh`),
    $r_* \equiv \mathcal{H}_* R_*$
    and
    $\mathcal{H}_* \eta_* = 1 + \nu_\text{gdh2024}$ (see :py:func:`H_eta`).
    Here $\mathcal{H} \equiv a'/a$ is the conformal Hubble rate, $\eta$ is conformal time
    and $R_*$ is the comoving mean bubble spacing.

    Please note that the source duration $\Delta \eta_\text{v}$ is not the effective source lifetime
    $\eta_\text{v} = \eta_* \Upsilon_\ell$ of :py:func:`pttools.ssm.scaling.H_star_eta_v`.

    :param ubarf: $\bar{U}_f$, enthalpy-weighted RMS fluid velocity
    :param r_star: $r_* \equiv \mathcal{H}_* R_*$, Hubble-scaled mean bubble spacing
    :param N_sh: $N_{\text{sh}}$, number of shock formation times
    :param nu: $\nu_\text{gdh2024}$
    :return: $\frac{\Delta \eta_\text{v}}{\eta_*}$
    """
    return N_sh * r_star / ((1 + nu) * ubarf)  # pyrefly: ignore[bad-return]


def H_eta[T: FloatOrArr](nu: T = DEFAULT_NU_GDH2024) -> T:  # type: ignore[assignment]
    r"""$\mathcal{H} \eta$, conformal Hubble rate times conformal time, for a barotropic EoS.

    $$\mathcal{H} \eta = \frac{a'}{a} \eta = 1 + \nu_\text{gdh2024} = \frac{2}{1 + 3 \omega},$$
    where $a' \equiv \frac{da}{d\eta}$.

    This comes from the scale factor for a barotropic EoS
    $$a(\eta) = a(\eta_*) \left( \frac{\eta}{\eta_*} \right)^{1 + \nu_\text{gdh2024}}
    = a(\eta_*) \left( \frac{\eta}{\eta_*} \right)^\frac{2}{1+3\omega}$$
    :giombi_2026:`\ ` eq. 2.17,
    :giombi_2024_cs:`\ ` eq. 2.15,
    which gives
    $$\mathcal{H} = \frac{1 + \nu_\text{gdh2024}}{\eta}.$$
    In a radiation-dominated Universe $\nu_\text{gdh2024} = 0$ and $\mathcal{H} \eta = 1$.

    Please note that this is not $H \eta$ with the physical Hubble rate $H = \frac{\dot{a}}{a} = \frac{\mathcal{H}}{a}$,
    which would depend on the normalisation of the scale factor.
    The corresponding physical relation is $H_* \tau = \mathcal{H}_* \eta$ with $\tau = a_* \eta$
    the physical time at $\eta_*$, so $H_* (a_* \eta_*) = 1 + \nu_\text{gdh2024}$,
    which is the "physical Hubble time" of :hindmarsh_2015:`\ ` appendix A.
    See the notation section of :py:mod:`pttools.ssm.scaling`.

    :param nu: $\nu_\text{gdh2024}$
    :return: $\mathcal{H} \eta$
    """
    # typing.cast() is not used below, since Numba cannot compile it.
    return 1 + nu  # pyrefly: ignore[bad-return]


def l[T: FloatOrArr](nu: T = DEFAULT_NU_GDH2024) -> T:  # noqa: E743  # type: ignore[assignment]
    r"""$\ell(\nu)
    $$\ell(\nu) = 1 + 2\nu$$
    :giombi_2026:`\ ` p. 25.
    """
    return 1 + 2 * nu  # pyrefly: ignore[bad-return]


def source_lifetime_factor[T: FloatOrArr](
        ubarf: T,
        r_star: T | float = DEFAULT_R_STAR,
        N_sh: T | float = DEFAULT_N_SH,
        nu: T | float = DEFAULT_NU_GDH2024) -> T:
    r"""
    Source lifetime factor $\Upsilon_\ell$.

    $$\Upsilon_\ell \equiv
    \frac{1}{\ell(\nu)} \left(1 - \left( \frac{\eta_*}{\eta_\text{end}} \right)^{\ell(\nu)} \right)
    = \frac{1}{\ell(\nu)} \left(1 - \left(1 + \frac{\Delta \eta_\text{v}}{\eta_*} \right)^{-\ell(\nu)} \right)$$
    :giombi_2026:`\ ` eqs. 3.6, 3.9a (there as $\Upsilon(\eta_* / \eta_\text{end}, 1 + 2\nu)$),
    :giombi_2024_cs:`\ ` eq. 3.13.
    It enters the GW power spectrum as $\mathcal{H}_* \eta_\text{v} = \mathcal{H}_* \eta_* \Upsilon_\ell$,
    see :py:func:`pttools.ssm.scaling.H_star_eta_v`.

    This is an updated version of
    :maki_msc:`\ ` eq. 3.79
    """
    return -Upsilon(r=1 + eta_ratio(ubarf=ubarf, r_star=r_star, N_sh=N_sh, nu=nu), l=-l(nu))  # pyrefly: ignore[bad-return]


def Upsilon[T: FloatOrArr](r: T, l: T | float) -> T:  # noqa: E741
    r"""$\Upsilon_\ell$ for arbitrary $\ell$
    $$\Upsilon_\ell (r) = \frac{1}{\ell} \left( 1 - r^\ell \right)$$
    :giombi_2026:`\ ` eq. 3.6.
    """
    return (1 - r**l) / l  # pyrefly: ignore[bad-return]
