"""Formulae for a barotropic equation of state."""

import numpy as np

from pttools.bubble.const import DEFAULT_NU_GDH2024
from pttools.speedup import njit
from pttools.ssm.const import DEFAULT_A_STAR_A_R_RATIO, DEFAULT_N_SH, DEFAULT_R_STAR
from pttools.type_hints import FloatOrArr


def dilution_of_e[T: FloatOrArr](
        a_star_a_r_ratio: T = DEFAULT_A_STAR_A_R_RATIO,
        nu: T | float = DEFAULT_NU_GDH2024) -> T:
    r"""Dilution of the background energy density $\bar{e}$.

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
    r"""$\frac{\Delta \eta_\text{v}}{\eta_*}$, source duration in units of the conformal time $\eta_*$.

    Here $\eta_*$ is the conformal time at the start of the acoustic phase.
    This is for a barotropic EoS.

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


def expansion_factor(eta_ratio: float, nu: float = DEFAULT_NU_GDH2024) -> float:
    r"""$\frac{\eta_\text{v}}{\Delta \eta_\text{v}}$, reduction of the GW power by the expansion of the Universe.

    $$\frac{\eta_\text{v}}{\Delta \eta_\text{v}} = \frac{\Upsilon_\ell}{\Delta \eta_\text{v} / \eta_*}$$
    This is the ratio of the effective lifetime $\eta_\text{v} = \eta_* \Upsilon_\ell$
    (see :py:func:`pttools.ssm.scaling.H_star_eta_v` and :py:func:`source_lifetime_factor`)
    to the source duration $\Delta \eta_\text{v}$.
    Therefore, it's the GW power that a stationary source produces in the expanding Universe
    during $\Delta \eta_\text{v}$, relative to the GW power that it would produce in flat space
    during the same time.
    For a short source $\Delta \eta_\text{v} \ll \eta_*$, it approaches 1,
    and for a long source $\Delta \eta_\text{v} \gg \eta_*$, it approaches
    $\frac{\eta_*}{\ell(\nu) \Delta \eta_\text{v}}$.
    In a radiation-dominated Universe ($\nu = 0$, $\ell = 1$), it's
    $$\frac{\eta_\text{v}}{\Delta \eta_\text{v}} = \frac{1}{1 + \Delta \eta_\text{v} / \eta_*}.$$

    :param eta_ratio: $\frac{\Delta \eta_\text{v}}{\eta_*}$, source duration in units of $\eta_*$,
        see :py:func:`eta_ratio`
    :param nu: $\nu_\text{gdh2024}$
    :return: $\frac{\eta_\text{v}}{\Delta \eta_\text{v}}$
    """
    if eta_ratio == 0:
        return 1.
    return float(-Upsilon(r=1 + eta_ratio, l=-l(nu)) / eta_ratio)


@njit(cache=True)
def H_eta[T: FloatOrArr](nu: T = DEFAULT_NU_GDH2024) -> T:
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
    return 1 + nu  # pyrefly: ignore[bad-return]


def l[T: FloatOrArr](nu: T = DEFAULT_NU_GDH2024) -> T:  # noqa: E743
    r"""$\ell(\nu)$, the exponent $\ell$ of the source lifetime factor $\Upsilon_\ell$.

    $$\ell(\nu) = 1 + 2\nu$$
    :giombi_2026:`\ ` below eqs. 3.10 and C.23.
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
    :giombi_2026:`\ ` eqs. 3.9a, 3.10 and C.23 (there as $\Upsilon_{1+2\nu}(\eta_* / \eta_\text{end})$),
    :giombi_2024_cs:`\ ` eq. 3.13.
    It enters the GW power spectrum as $\mathcal{H}_* \eta_\text{v} = \mathcal{H}_* \eta_* \Upsilon_\ell$,
    see :py:func:`pttools.ssm.scaling.H_star_eta_v`.

    This is an updated version of
    :maki_msc:`\ ` eq. 3.79
    """
    # nu must be a scalar, as Upsilon requires a scalar l.
    return -Upsilon(  # pyrefly: ignore[bad-return]
        r=1 + eta_ratio(ubarf=ubarf, r_star=r_star, N_sh=N_sh, nu=nu),
        l=-l(nu)  # pyrefly: ignore[bad-argument-type]
    )


@njit(cache=True)
def Upsilon[T: FloatOrArr](r: T, l: float) -> T:  # noqa: E741
    r"""$\Upsilon_\ell$ for arbitrary $\ell$.

    $$\Upsilon_\ell (r) = \frac{1}{\ell} \left( 1 - r^\ell \right)$$
    :giombi_2026:`\ ` eqs. 3.6 and C.23.
    This is computed as $\Upsilon_\ell (r) = -\frac{1}{\ell} \left( e^{\ell \ln r} - 1 \right)$
    with ``numpy.expm1``, which is accurate also for small $|\ell|$,
    and with the limit $\Upsilon_0 (r) = -\ln r$ for $\ell = 0$.

    :param r: $r$, e.g. $\frac{\eta_*}{\eta_\text{end}}$
    :param l: $\ell$, a scalar
    :return: $\Upsilon_\ell (r)$
    """
    log_r = np.log(r)
    if l == 0:
        return -log_r  # pyrefly: ignore[bad-return]
    return -np.expm1(l * log_r) / l  # pyrefly: ignore[bad-return]


@njit(cache=True)
def tau_star[T: FloatOrArr](r_star: T, nu: T | float = DEFAULT_NU_GDH2024) -> T:
    r"""$\tau_*$, dimensionless conformal time when the anisotropic stress turns on.

    $$\tau_* \equiv \frac{\eta_*}{R_*} = \frac{\mathcal{H}_* \eta_*}{\mathcal{H}_* R_*}
    = \frac{1 + \nu_\text{gdh2024}}{r_*}$$
    :giombi_2026:`\ ` sec. 2.5 and below eq. 3.5,
    :giombi_2024_cs:`\ ` p. 10,
    where $\mathcal{H}_* \eta_*$ is given by :py:func:`H_eta`.

    :param r_star: $r_* \equiv \mathcal{H}_* R_*$, Hubble-scaled mean bubble spacing
    :param nu: $\nu_\text{gdh2024}$
    :return: $\tau_*$
    """
    return H_eta(nu) / r_star  # pyrefly: ignore[bad-return]
