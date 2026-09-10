"""Scaling factors."""

import typing as tp

import numpy as np

from pttools.bubble.const import DEFAULT_NU_GDH2024
from pttools.bubble.energy_budget import ubarf_approx_K
from pttools.ssm.barotropic import H_eta, source_lifetime_factor
from pttools.ssm.const import DEFAULT_N_SH
from pttools.type_hints import FloatArr1D, FloatOrArr


def H_star_tau_nl(r_star: FloatOrArr, ubarf: FloatOrArr) -> FloatOrArr:
    r"""Hubble-scaled timescale of non-linearities $H \tau_\text{nl}$.

    $$H_* \tau_\text{nl} = \frac{r_*}{\bar{U}_f}$$.
    :gw_pt_ssm:`\ ` p. 6, 13
    :notes:`\ ` p. 48
    :giombi_2024_cs:`\ ` p. 2.

    This is also known as the Hubble-scaled shock appearance timescale $H_* \tau_\text{sh}$.
    :hindmarsh_2017:`\ ` eq. 22.

    Some sources use the notation $v_{\text{rms}} \equiv \bar{U}_f$.

    Please note that $\tau_\text{nl}$ and $\tau_\text{v}$ are different quantities.
    If $H \tau_\text{nl} \gg 1$, then $H \tau_\text{v} \rightarrow 1$.
    :gw_pt_ssm:`\ ` p. 13
    """
    return r_star / ubarf


def H_star_tau_nl_approx(r_star: FloatOrArr, K: FloatOrArr) -> FloatOrArr:
    r"""Approximation of Hubble-scaled timescale of non-linearities $H \tau_\text{nl}$.

    $$H_* \tau_\text{nl} = \frac{r_*}{\bar{U}_f} \approx \frac{r_*}{\sqrt{K}}$$
    :hindmarsh_2017:`\ ` eq. 22,
    :caprini_2020:`\ ` p. 17,
    :ajmi_2022:`\ ` p. 9.
    """
    return r_star / ubarf_approx_K(K)


def H_star_tau_v(source_lifetime_factor: FloatOrArr, nu: FloatOrArr = DEFAULT_NU_GDH2024) -> FloatOrArr:
    r"""$H_* \tau_\text{v}$, Hubble-scaled effective lifetime of the source.

    $$\mathcal{H} \tau_v = \mathcal{H}_* \eta_* \Upsilon_\ell$$
    :ajmi_2022:`\ ` eq. 80
    :gowling_2021:`\ ` eq. 2.7.
    """
    return H_eta(nu) * source_lifetime_factor


def H_star_tau_v_old[T: FloatOrArr](H_star_tau_nl: T) -> T:
    r"""$H_* \tau_\text{v}$, Hubble-scaled effective lifetime of the source, old approximation.

    $$H_* \tau_v \approx 1 - \frac{1}{\sqrt{1 + 2x}}$$,
    where $x = H_* \tau_\text{nl}$.
    :ajmi_2022:`\ ` eq. 80,
    :gowling_2021:`\ ` eq. 2.7.
    This is an approximation, and the source lifetime factor should be used instead.

    :param H_star_tau_nl: $H_* \tau_\text{nl}$
    """
    return tp.cast(T, 1 - (1 + 2 * H_star_tau_nl) ** (-0.5))


def J(r_star: FloatOrArr, H_star_tau_v: FloatOrArr) -> FloatOrArr:
    r"""Combined lifetime factor $J$.

    $$J \equiv r_* H_* \tau_v$$
    """
    return r_star * H_star_tau_v


def J_full(
        r_star: FloatOrArr,
        ubarf: FloatOrArr,
        N_sh: FloatOrArr = DEFAULT_N_SH,
        nu: FloatOrArr = DEFAULT_NU_GDH2024) -> FloatOrArr:
    r"""Combined lifetime factor $J$.

    This function calls the sub-functions directly.
    $$J \equiv r_* H_* \tau_\text{v}
    = r_* \mathcal{H}_* \eta_* \Upsilon_\ell
    = r_* (1 + \nu) \Upsilon_\ell
    = r_* (1 + \nu) \frac{1}{\ell(\nu)} \left(1 - \left( \frac{\eta_*}{\eta_\text{end}} \right)^{\ell(\nu)} \right)
    = r_* (1 + \nu) \frac{1}{\ell(\nu)} \left(1 - \left(1 + \frac{\Delta \eta_\text{v}}{\eta_*} \right)^{-\ell(\nu)} \right)
    $$
    See
    :py:func:`pttools.ssm.scaling.H_star_tau_v`,
    :py:func:`pttools.ssm.barotropic.source_lifetime_factor` and
    :py:func:`pttools.ssm.barotropic.eta_ratio`.
    """
    return J(
        r_star=r_star,
        H_star_tau_v=H_star_tau_v(
            source_lifetime_factor=source_lifetime_factor(
                ubarf=ubarf, r_star=r_star, N_sh=N_sh, nu=nu
            ),
            nu=nu
        )
    )


def J_old(r_star: FloatOrArr, K: FloatOrArr) -> FloatOrArr:
    r"""Combined lifetime factor $J$, old approximation.

    $$J \equiv r_* H_* \tau_\text{v} \approx r_* \left(1 - \frac{1}{\sqrt{1 + 2x}}$$,
    where $x = \frac{r_*}{\sqrt{K}}$
    :gowling_2021:`\ ` eq. 2.8,
    :ajmi_2022:`\ ` eq. 81.
    """
    return J(
        r_star=r_star,
        H_star_tau_v=H_star_tau_v_old(
            H_star_tau_nl=H_star_tau_nl_approx(r_star=r_star, K=K)
        )
    )


def omega_tilde_gw(x: FloatArr1D, spec_den_gw: FloatArr1D) -> float:
    r"""GW production efficiency parameter $\Omega_\text{gw}$.

    $$\tilde{\Omega}_\text{gw} = \frac{1}{2 \pi^2} \int_0^\infty dx x^2 \tilde{P}_\text{gw}(x)$$
    This parameter quantifies the efficiency with which shear stress is converted to gravitational waves.
    :hindmarsh_2017:`\ ` eq. 24.
    This parameter is approximately independent of the length scale and RMS velocity of the fluid flow.
    :hindmarsh_2015:`\ `
    """
    return 1 / (2 * np.pi) * np.trapezoid(x**2 * spec_den_gw, x)  # type: ignore[return-value]
