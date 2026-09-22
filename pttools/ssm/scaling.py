r"""Scaling factors of the gravitational wave power spectrum.

**Notation.**
This module follows the conformal notation of :giombi_2026:`\ ` and :giombi_2024_cs:`\ `:
$\eta$ is conformal time, $\mathcal{H} \equiv a'/a$ is the conformal Hubble rate,
$R_*$ is the comoving mean bubble spacing and $r_* \equiv \mathcal{H}_* R_*$.
As Python identifiers cannot contain $\mathcal{H}$, the prefix ``H_star_`` in the names of this module
denotes the conformal Hubble rate $\mathcal{H}_* \equiv \mathcal{H}(\eta_*)$ at the start of the acoustic phase.

The earlier articles :hindmarsh_2015:`\ `, :hindmarsh_2017:`\ `, :gw_pt_ssm:`\ `, :gowling_2021:`\ ` and :ajmi_2022:`\ `
work in Minkowski space with the physical Hubble rate $H_* = \mathcal{H}_* / a_*$ and physical times $\tau$
and lengths $L_f = R_*$ measured at $\eta_*$.
Since a physical time or length at $\eta_*$ is $a_*$ times the conformal or comoving one,
the dimensionless combinations agree:
$H_* \tau = \mathcal{H}_* \eta$ and $H_* R_*^\text{phys} = \mathcal{H}_* R_*^\text{com}$.
Therefore, the values computed here are the same as those of the earlier notation,
and the correspondences are
$\mathcal{H}_* \eta_\text{sh} \leftrightarrow H_* \tau_\text{nl} = H_* \tau_\text{sh}$ and
$\mathcal{H}_* \eta_\text{v} \leftrightarrow H_* \tau_\text{v}$.
Please note that $\mathcal{H}_* \eta_* = 1 + \nu$, whereas the physical Hubble rate times the cosmic time
is $H_* t_* = \frac{2}{3(1+\omega)}$, so $H$ and $\mathcal{H}$ must not be interchanged without also
changing the time variable.

Not to be confused with the above,
:giombi_2024_cs:`\ ` and :giombi_2026:`\ ` also use a dimensionless conformal time $\tau \equiv \eta / R_*$,
which is what :py:attr:`pttools.ssm.spectrum.SSMSpectrum.tau_star` and
:py:attr:`pttools.ssm.spectrum.SSMSpectrum.tau_end` denote.
"""

import numpy as np

from pttools.bubble.const import DEFAULT_ADIABATIC_INDEX, DEFAULT_NU_GDH2024
from pttools.bubble.thermo import ubarf2_from_K
from pttools.ssm.barotropic import H_eta, source_lifetime_factor
from pttools.ssm.const import DEFAULT_N_SH
from pttools.type_hints import FloatArr1D, FloatOrArr


def H_star_eta_sh[T: FloatOrArr](r_star: T, ubarf: T | float) -> T:
    r"""$\mathcal{H}_* \eta_\text{sh}$, Hubble-scaled shock formation time.

    $$\mathcal{H}_* \eta_\text{sh} = \frac{\mathcal{H}_* R_*}{\bar{U}_f} = \frac{r_*}{\bar{U}_f}$$
    This definition is a choice.

    Shocks and other non-linearities appear on the timescale $\eta_\text{sh} \approx R_* / \bar{U}_f$,
    where $R_*$ is the comoving mean bubble spacing and $\bar{U}_f$ the enthalpy-weighted RMS fluid velocity.
    :giombi_2026:`\ ` sec. 1 (as $\eta_\text{sh} = R_*/v_{\text{rms}}$),
    :giombi_2024_cs:`\ ` sec. 1 (as $\tau_\text{nl} = R_*/v_{\text{rms}}$).

    In the simplified model of the velocity field of their numerical results,
    Giombi et al. instead define $\eta_\text{sh} \equiv \xi_* / v_{\text{rms}}$
    using the integral scale $\xi_* = R_* / (4 \pi \sqrt{3})$
    (:giombi_2026:`\ ` eq. 4.2, :giombi_2024_cs:`\ ` eq. 4.2),
    which is a factor $4 \pi \sqrt{3} \approx 21.8$ shorter.
    PTtools uses the definition of Hindmarsh et al. given above.

    Earlier notation: this is the Hubble-scaled non-linearity timescale $H_* \tau_\text{nl}$
    (:gw_pt_ssm:`\ ` p. 6, 13, :notes:`\ ` p. 48),
    also known as the shock appearance timescale $H_* \tau_\text{sh}$
    (:hindmarsh_2017:`\ ` eq. 22, :ajmi_2022:`\ ` p. 9),
    with $\tau_\text{sh} = a_* \eta_\text{sh}$ the physical timescale at $\eta_*$
    and $H_* = \mathcal{H}_*/a_*$ the physical Hubble rate.
    Some sources use the notation $v_{\text{rms}} \equiv \bar{U}_f$.

    Please note that $\eta_\text{sh}$ and $\eta_\text{v}$ are different quantities.
    If $\mathcal{H}_* \eta_\text{sh} \gg 1$, then
    $\mathcal{H}_* \eta_\text{v} \rightarrow \mathcal{H}_* \eta_* / \ell(\nu)$,
    which is 1 in a radiation-dominated Universe.
    :gw_pt_ssm:`\ ` p. 13

    :param r_star: $r_* \equiv \mathcal{H}_* R_*$, Hubble-scaled mean bubble spacing
    :param ubarf: $\bar{U}_f$, enthalpy-weighted RMS fluid velocity
    :return: $\mathcal{H}_* \eta_\text{sh}$
    """
    return r_star / ubarf  # pyrefly: ignore[bad-return]


def H_star_eta_sh_full[T: FloatOrArr](
        r_star: T,
        K: T | float,
        adiabatic_index: T | float = DEFAULT_ADIABATIC_INDEX) -> T:
    r"""$\mathcal{H}_* \eta_\text{sh}$, Hubble-scaled shock formation time.

    $$\mathcal{H}_* \eta_\text{sh} = \frac{r_*}{\bar{U}_f} = r_* \sqrt{\frac{\Gamma}{K}}$$
    :hindmarsh_2017:`\ ` eq. 22,
    :caprini_2020:`\ ` p. 17.
    See :py:func:`H_star_eta_sh`.

    Earlier notation:
    $$H_* \tau_\text{nl} \approx H_* \tau_\text{sh}$$.

    Some sources define
    $$H_* \tau_\text{sh} = \frac{r_*}{\sqrt{K}}$$.
    :ajmi_2022:`\ ` p. 9.
    This lacks the factor $\Gamma$ from within the square root.

    :param r_star: $r_* \equiv \mathcal{H}_* R_*$, Hubble-scaled mean bubble spacing
    :param K: $K$, kinetic energy fraction
    :param adiabatic_index: $\Gamma$, mean adiabatic index
    :return: $\mathcal{H}_* \eta_\text{sh}$
    """
    return r_star / np.sqrt(ubarf2_from_K(K, adiabatic_index=adiabatic_index))  # pyrefly: ignore[bad-return]


def H_star_eta_v[T: FloatOrArr](source_lifetime_factor: T, nu: T | float = DEFAULT_NU_GDH2024) -> T:
    r"""$\mathcal{H}_* \eta_\text{v}$, Hubble-scaled effective lifetime of the source.

    $$\mathcal{H}_* \eta_\text{v} \equiv \mathcal{H}_* \eta_* \Upsilon_\ell = (1 + \nu) \Upsilon_\ell$$
    The effective lifetime $\eta_\text{v} \equiv \eta_* \Upsilon_\ell$ is the conformal time for which
    a stationary source would have to act in a non-expanding Universe
    to produce the same gravitational wave power as the actual source,
    which is on for $\Delta \eta_\text{v} = \eta_\text{end} - \eta_*$ in the expanding Universe.
    Here $\Upsilon_\ell = \Upsilon(\eta_* / \eta_\text{end}, \ell(\nu))$ with $\ell(\nu) = 1 + 2\nu$
    is the source lifetime factor of
    :giombi_2026:`\ ` eqs. 3.6, 3.9a and
    :giombi_2024_cs:`\ ` eq. 3.13,
    and $\mathcal{H}_* \eta_* = 1 + \nu$ for a barotropic equation of state
    (:py:func:`pttools.ssm.barotropic.H_eta`).

    Limits: for a short source $\Delta \eta_\text{v} \ll \eta_*$, $\eta_\text{v} \rightarrow \Delta \eta_\text{v}$,
    and for a long source $\Delta \eta_\text{v} \gg \eta_*$, $\eta_\text{v} \rightarrow \eta_* / \ell(\nu)$,
    that is, one conformal Hubble time in a radiation-dominated Universe ($\nu = 0$).

    Earlier notation: this is $H_* \tau_\text{v}$ of
    :hindmarsh_2015:`\ ` eq. 42 and appendix A,
    :gw_pt_ssm:`\ ` eq. 3.48,
    :gowling_2021:`\ ` eq. 2.7 and
    :ajmi_2022:`\ ` eq. 80,
    with $\tau_\text{v} = a_* \eta_\text{v}$ the effective lifetime in physical units at $\eta_*$
    and $H_* = \mathcal{H}_* / a_*$ the physical Hubble rate,
    so that $H_* \tau_\text{v} = \mathcal{H}_* \eta_\text{v}$.
    Please note that $\eta_\text{v}$ is not the source duration $\Delta \eta_\text{v}$
    (see :py:func:`pttools.ssm.barotropic.eta_ratio`).

    :param source_lifetime_factor: $\Upsilon_\ell$, source lifetime factor
    :param nu: $\nu_\text{gdh2024}$
    :return: $\mathcal{H}_* \eta_\text{v}$
    """
    return H_eta(nu) * source_lifetime_factor  # pyrefly: ignore[bad-return]


def H_star_eta_v_old[T: FloatOrArr](H_star_eta_sh: T) -> T:
    r"""$\mathcal{H}_* \eta_\text{v}$, Hubble-scaled effective lifetime of the source, old approximation.

    $$\mathcal{H}_* \eta_\text{v} \approx 1 - \frac{1}{\sqrt{1 + 2x}},$$
    where $x = \mathcal{H}_* \eta_\text{sh}$.
    :guo_2020:`\ `,
    :notes:`\ ` eq. 8.19,
    :gowling_2021:`\ ` eq. 2.7,
    :ajmi_2022:`\ ` eq. 80,
    :hakkinen_msc:`\ ` eq. 3.30.
    In the earlier notation this is $H_* \tau_\text{v} \approx 1 - (1 + 2 H_* \tau_\text{sh})^{-1/2}$.
    Please note that :notes:`\ ` eq. 8.19 has a typo: $1 + 2 \tau_\text{nl} R_*$ should be $1 + 2 \tau_\text{nl} H_n$.

    This formula treats the source as constant for a duration $\tau_\text{sh} = R_* / \bar{U}_f$
    of cosmic time $t$ after $t_*$ in a radiation-dominated Universe ($\nu = 0$),
    where $a \propto t^{1/2}$ and $H_* = 1 / (2 t_*)$.
    As $\eta \propto t^{1/2}$, this gives
    $\eta_* / \eta_\text{end} = (1 + 2 H_* \tau_\text{sh})^{-1/2}$,
    and the formula is the source lifetime factor $\Upsilon_1 = 1 - \eta_* / \eta_\text{end}$
    of :py:func:`H_star_eta_v` with the source duration measured in cosmic time.
    :py:func:`H_star_eta_v` instead measures the source duration $\Delta \eta_\text{v} = N_{\text{sh}} \eta_\text{sh}$
    in conformal time, which is the natural time variable of sound waves in an expanding Universe
    (:hindmarsh_2015:`\ ` appendix A), and it should be used instead.
    The two agree for $x \ll 1$ and for $x \gg 1$.

    :param H_star_eta_sh: $\mathcal{H}_* \eta_\text{sh}$, Hubble-scaled shock formation time
    :return: $\mathcal{H}_* \eta_\text{v}$
    """
    return 1 - (1 + 2 * H_star_eta_sh) ** (-0.5)  # pyrefly: ignore[bad-return]


def H_star_eta_v_old2[T: FloatOrArr](H_star_eta_sh: T) -> T:
    r"""$\mathcal{H}_* \eta_\text{v}$, Hubble-scaled effective lifetime of the source, old approximation 2.

    $$\mathcal{H}_* \eta_\text{v} \approx \min(\mathcal{H}_* \eta_\text{sh}, 1)$$
    :notes:`\ ` p. 48,
    :hakkinen_msc:`\ ` eq. 3.27.
    In the earlier notation this is $H_* \tau_\text{v} \approx \min(H_* \tau_\text{nl}, 1)$.
    This is an even rougher approximation than :py:func:`H_star_eta_v_old`.

    :param H_star_eta_sh: $\mathcal{H}_* \eta_\text{sh}$, Hubble-scaled shock formation time
    :return: $\mathcal{H}_* \eta_\text{v}$
    """
    return np.minimum(H_star_eta_sh, 1.)  # pyrefly: ignore[bad-return]


def J[T: FloatOrArr](r_star: T, H_star_eta_v: T | float) -> T:
    r"""Combined lifetime factor $J$.

    $$J \equiv (\mathcal{H}_* R_*)(\mathcal{H}_* \eta_\text{v}) = r_* \mathcal{H}_* \eta_\text{v}$$
    :gowling_2021:`\ ` eq. 2.8,
    :ajmi_2022:`\ ` eq. 81
    (there as $J = (H_n R_*)(H_n \tau_\text{v})$, which has the same value).

    :param r_star: $r_* \equiv \mathcal{H}_* R_*$, Hubble-scaled mean bubble spacing
    :param H_star_eta_v: $\mathcal{H}_* \eta_\text{v}$, Hubble-scaled effective lifetime of the source
    :return: $J$
    """
    return r_star * H_star_eta_v  # pyrefly: ignore[bad-return]


def J_full[T: FloatOrArr](
        r_star: T,
        ubarf: T | float,
        N_sh: T | float = DEFAULT_N_SH,
        nu: T | float = DEFAULT_NU_GDH2024) -> T:
    r"""Combined lifetime factor $J$.

    This function calls the sub-functions directly.
    $$J \equiv r_* \mathcal{H}_* \eta_\text{v}
    = r_* \mathcal{H}_* \eta_* \Upsilon_\ell
    = r_* (1 + \nu) \Upsilon_\ell
    = r_* (1 + \nu) \frac{1}{\ell(\nu)}
    \left(1 - \left( \frac{\eta_*}{\eta_\text{end}} \right)^{\ell(\nu)} \right)
    = r_* (1 + \nu) \frac{1}{\ell(\nu)}
    \left(1 - \left( 1 + \frac{\Delta \eta_\text{v}}{\eta_*} \right)^{-\ell(\nu)} \right)$$
    See
    :py:func:`H_star_eta_v`,
    :py:func:`pttools.ssm.barotropic.source_lifetime_factor` and
    :py:func:`pttools.ssm.barotropic.eta_ratio`.

    :param r_star: $r_* \equiv \mathcal{H}_* R_*$, Hubble-scaled mean bubble spacing
    :param ubarf: $\bar{U}_f$, enthalpy-weighted RMS fluid velocity
    :param N_sh: $N_{\text{sh}}$, number of shock formation times
    :param nu: $\nu_\text{gdh2024}$
    :return: $J$
    """
    return J(
        r_star=r_star,
        H_star_eta_v=H_star_eta_v(
            source_lifetime_factor=source_lifetime_factor(
                ubarf=ubarf, r_star=r_star, N_sh=N_sh, nu=nu
            ),
            nu=nu
        )
    )


def J_old[T: FloatOrArr](r_star: T, K: T | float, adiabatic_index: T | float = DEFAULT_ADIABATIC_INDEX) -> T:
    r"""Combined lifetime factor $J$, old approximation.

    $$J \equiv r_* \mathcal{H}_* \eta_\text{v} \approx r_* \left(1 - \frac{1}{\sqrt{1 + 2x}} \right),$$
    where $x = \frac{r_*}{\sqrt{K}}$.
    :gowling_2021:`\ ` eq. 2.8,
    :ajmi_2022:`\ ` eq. 81,
    :hakkinen_msc:`\ ` eq. 3.31.
    See :py:func:`H_star_eta_v_old`.

    :param r_star: $r_* \equiv \mathcal{H}_* R_*$, Hubble-scaled mean bubble spacing
    :param K: $K$, kinetic energy fraction
    :return: $J$
    """
    return J(
        r_star=r_star,
        H_star_eta_v=H_star_eta_v_old(
            H_star_eta_sh=H_star_eta_sh_full(r_star=r_star, K=K, adiabatic_index=adiabatic_index)
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
    return 1 / (2 * np.pi**2) * np.trapezoid(x**2 * spec_den_gw, x)  # pyrefly: ignore[bad-return]
