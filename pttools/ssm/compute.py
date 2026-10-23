r"""Functions for organizing the computation of the GW spectra.

The computation is split into two stages:

- :py:func:`compute_base` computes the velocity spectra, the GW spectral density without the source lifetime factor
  $\Upsilon_\ell$ and the source integral $\mathcal{I}_v$ of the low-$k$ approximation.
  These do not depend on the Hubble-scaled mean bubble spacing $r_*$,
  and this is the computationally heavy stage due to the sine transforms of the fluid profile
  and the convolution integral of the GW spectral density.
- :py:func:`compute_gw` computes the GW spectra from the results of :py:func:`compute_base`.
  These depend on $r_*$ through the source lifetime factor, which is a mere scaling factor,
  and through the low-$k$ approximation.

Therefore, spectra that differ only by $r_*$ can reuse the results of :py:func:`compute_base`,
see :py:meth:`pttools.ssm.spectrum.SSMSpectrum.with_params`.
The inputs and the results are passed as named tuples, which are supported by Numba.
"""

from __future__ import annotations

import typing as tp

from pttools.speedup import njit
from pttools.ssm.low_k.integration import Iv_resampled, power_spectrum_integration_int, spec_den_gw_low
from pttools.ssm.low_k.join import gw_junction
from pttools.ssm.nucleation import NucType, bubble_spacing_enlargement
from pttools.ssm.spec_den_gw import gen_lookup, spec_den_gw
from pttools.ssm.spec_den_v import spec_den_v as spec_den_v_func
from pttools.ssm.ssm import A2_e_conserving, LambdaCorrection, ubarf2_from_a2
from pttools.ssm.ssm import T_tilde as T_tilde_func
from pttools.ssm.ssm import qT_lookup as qT_lookup_func
import pttools.type_hints as th
from pttools.type_hints import FloatArr1D

if th.TYPE_CHECKING_OR_DOCS:
    from pttools.bubble.solution_type import SolutionType


class SSMInput(tp.NamedTuple):
    """Input parameters of the Sound Shell Model computation."""

    # Arrays (in alphabetical order)
    #: $e$, energy density of the fluid profile
    e: FloatArr1D
    #: $T$, temperature of the fluid profile.
    #: This is not called ``T`` to avoid ambiguous cross-references with the type variable ``T`` in the docs.
    T_profile: FloatArr1D
    #: $v$, fluid velocity of the fluid profile
    v: FloatArr1D
    #: $w$, enthalpy density of the fluid profile
    w: FloatArr1D
    #: $\xi$, self-similar coordinate of the fluid profile
    xi: FloatArr1D
    #: $z = kR_*$ array
    y: FloatArr1D
    # Scalars (in alphabetical order)
    #: $\tilde{\beta}$, nucleation rate parameter, or nan if not given.
    #: This is nan instead of None, so that the compiled functions do not have to be compiled separately for None.
    beta_tilde: float
    #: $c_s$, speed of sound in the broken phase
    cs: float
    #: $a$, parameter of the bubble lifetime distribution
    lifetime_distribution_a: float
    #: $\nu_\text{gdh2024}$
    nu_gdh2024: float
    #: $\Upsilon_\ell$, source lifetime factor
    source_lifetime_factor: float
    #: $\tau_\text{end}$, dimensionless conformal time when the anisotropic stress turns off
    tau_end: float
    #: $\tau_*$, dimensionless conformal time when the anisotropic stress turns on
    tau_star: float
    #: $v_\text{sh}$, shock speed
    v_sh: float
    #: $v_\text{wall}$, wall speed
    v_wall: float
    # Accuracy parameters
    #: small offset used when generating the lookup table of $z$ values for the GW spectrum
    eps_lookup: float
    #: number of points in the $\tilde{T}$ array
    nT: int  # noqa: N815
    #: number of points in the lookup arrays
    n_z_lookup: int
    #: number of points for the integration of the GW spectral density, or 0 for the size of the lookup array
    nx_P_tilde_gw: int  # noqa: N815
    #: $\tilde{T}_\text{min}$
    T_tilde_min: float
    #: $\tilde{T}_\text{max}$
    T_tilde_max: float
    #: threshold above which the sine transform is approximated
    z_st_thresh: float
    # Other
    #: nucleation type
    nuc_type: NucType
    #: solution type of the fluid profile
    sol_type: SolutionType
    #: kinetic energy term of $\lambda$, see :py:class:`pttools.ssm.ssm.LambdaCorrection`
    lambda_correction: LambdaCorrection
    #: whether to compute the acoustic friction of the low-$k$ approximation from $c_s$ instead of $\nu$
    low_k_cs_friction: bool


class SSMBaseResults(tp.NamedTuple):
    r"""Results of :py:func:`compute_base`, which do not depend on $r_*$."""

    #: ${\lvert A \rvert}^2$ for $qT_\text{lookup}$
    A2: FloatArr1D
    #: ${\lvert A \rvert}^2$ for $qT_\text{gw,lookup}$
    A2_lookup: FloatArr1D
    #: $qT_\text{lookup}$
    qT_lookup: FloatArr1D
    #: $qT_\text{gw,lookup}$
    qT_gw_lookup: FloatArr1D  # noqa: N815
    #: $\tilde{P}_v(z)$
    spec_den_v: FloatArr1D
    #: $\tilde{P}_v$ for the lookup array $z_\text{lookup}$
    spec_den_v_lookup: FloatArr1D
    #: $\tilde{T}$
    T_tilde: FloatArr1D
    #: $\bar{U}_f^2$
    ubarf2: float
    #: $z_\text{lookup}$
    z_lookup: FloatArr1D
    #: $\tilde{P}_\text{gw,ssm} / \Upsilon_\ell$, GW spectral density without the source lifetime factor
    spec_den_gw_ssm_unscaled: FloatArr1D
    #: $\mathcal{I}_v$, source contribution of the low-$k$ approximation,
    #: see :py:func:`pttools.ssm.low_k.integration.Iv_resampled`
    Iv: float


class SSMGWResults(tp.NamedTuple):
    r"""Results of :py:func:`compute_gw`, which depend on $r_*$."""

    #: $\tilde{P}_\text{gw,ssm}$
    spec_den_gw_ssm: FloatArr1D
    #: $\tilde{P}_\text{gw,low}$
    spec_den_gw_low: FloatArr1D
    #: $\tilde{P}_\text{gw,int}$
    spec_den_gw_int: FloatArr1D
    #: $\tilde{P}_\text{gw,ext}$
    spec_den_gw_expanded: FloatArr1D
    #: nucleation $f$
    nucleation_f: float
    #: nucleation $h_x$
    hx: float
    #: bubble spacing enlargement factor $\Lambda$
    bubble_spacing_enlargement_factor: float


@njit(nogil=True)
def compute(inp: SSMInput, parallel: bool = True) -> tuple[SSMBaseResults, SSMGWResults]:
    """Compute the Sound Shell Model spectra for a fluid profile, including the low-k approximation.

    This is in one Numba-compiled function so that the GIL is released for the entire computation.

    :param inp: input parameters
    :param parallel: whether to use multiple CPU cores
    :return: results of :py:func:`compute_base` and :py:func:`compute_gw`
    """
    base_res = compute_base(inp=inp, parallel=parallel)
    gw_res = compute_gw(inp=inp, base_res=base_res)
    return base_res, gw_res


@njit(nogil=True)
def compute_gw(inp: SSMInput, base_res: SSMBaseResults) -> SSMGWResults:
    r"""Compute the GW spectra, including the low-$k$ approximation, from the results of :py:func:`compute_base`.

    The source lifetime factor $\Upsilon_\ell$ is applied to the GW spectral density as a scaling factor,
    $$\tilde{P}_\text{gw,ssm} = \Upsilon_\ell \left( \frac{\tilde{P}_\text{gw,ssm}}{\Upsilon_\ell} \right),$$
    see :py:func:`pttools.ssm.spec_den_gw.spec_den_gw`.
    This is computationally light, as the convolution integrals have been computed by :py:func:`compute_base`.

    :param inp: input parameters
    :param base_res: results of :py:func:`compute_base`
    :return: GW spectra
    """
    nucleation_f, hx, bubble_spacing_enlargement_factor = bubble_spacing_enlargement(
        xi=inp.xi, T=inp.T_profile, beta_tilde=inp.beta_tilde, v_wall=inp.v_wall, sol_type=inp.sol_type
    )
    spec_den_gw_ssm = inp.source_lifetime_factor * base_res.spec_den_gw_ssm_unscaled
    spec_den_gw_low, spec_den_gw_int, spec_den_gw_expanded = compute_low_k(
        spec_den_gw_ssm=spec_den_gw_ssm,
        y=inp.y,
        cs=inp.cs, Iv=base_res.Iv, nu_gdh2024=inp.nu_gdh2024,
        tau_end=inp.tau_end, tau_star=inp.tau_star,
        cs_friction=inp.low_k_cs_friction
    )
    return SSMGWResults(
        spec_den_gw_ssm=spec_den_gw_ssm,
        spec_den_gw_low=spec_den_gw_low,
        spec_den_gw_int=spec_den_gw_int,
        spec_den_gw_expanded=spec_den_gw_expanded,
        nucleation_f=nucleation_f,
        hx=hx,
        bubble_spacing_enlargement_factor=bubble_spacing_enlargement_factor
    )


@njit(nogil=True)
def compute_low_k(
        # Arrays (in alphabetical order)
        spec_den_gw_ssm: FloatArr1D,
        y: FloatArr1D,
        # Scalars (in alphabetical order)
        cs: float,
        Iv: float,
        nu_gdh2024: float,
        tau_end: float,
        tau_star: float,
        # Other
        cs_friction: bool = False) -> tuple[FloatArr1D, FloatArr1D, FloatArr1D]:
    r"""Compute the low-k approximation of :giombi_2024_cs:`\ `.

    The source integral $\mathcal{I}_v$ should be computed with :py:func:`pttools.ssm.low_k.integration.Iv_resampled`
    over the lookup table of $\tilde{P}_v$, which covers a wider range of momenta than $y$,
    so that $\tilde{P}_\text{gw}^\text{low}$ and $\tilde{P}_\text{gw}^\text{int}$ use the same $\mathcal{I}_v$
    and therefore cross at :py:func:`pttools.ssm.low_k.intersection.z_cross_junction`.
    Then $\tilde{P}_\text{gw}^\text{low}$ is the same as that of
    :py:func:`pttools.ssm.low_k.integration.power_spectrum_integration_low`.

    :param spec_den_gw_ssm: $\tilde{P}_\text{gw,ssm}$, including the source lifetime factor
    :param y: $z = kR_*$ array
    :param cs: $c_s$, speed of sound
    :param Iv: $\mathcal{I}_v$, source contribution
    :param nu_gdh2024: $\nu_\text{gdh2024}$
    :param tau_end: $\tau_\text{end}$
    :param tau_star: $\tau_*$
    :param cs_friction: whether to compute the acoustic friction from $c_s$ instead of $\nu$,
        see :py:func:`pttools.ssm.low_k.kernel.kernel_low`. Disabled by default.
    :return: $\tilde{P}_\text{gw}^\text{low}$, $\tilde{P}_\text{gw}^\text{int}$ and the joined spectrum
    """
    spec_den_gw_low_arr = spec_den_gw_low(
        z=y, cs=cs, nu=nu_gdh2024, tau_star=tau_star, tau_end=tau_end, Iv=Iv, cs_friction=cs_friction
    )
    spec_den_gw_int = power_spectrum_integration_int(z=y, cs=cs, tau_star=tau_star, Iv=Iv)
    spec_den_gw_expanded = gw_junction(
        z=y,
        Pgw_low=spec_den_gw_low_arr,
        Pgw_int=spec_den_gw_int,
        Pgw_high=spec_den_gw_ssm,
        cs=cs, nu=nu_gdh2024,
        tau_star=tau_star, tau_end=tau_end,
        cs_friction=cs_friction
    )
    return spec_den_gw_low_arr, spec_den_gw_int, spec_den_gw_expanded


@njit(nogil=True)
def compute_base(inp: SSMInput, parallel: bool = True) -> SSMBaseResults:
    r"""Compute the quantities of the Sound Shell Model that do not depend on $r_*$ for the given fluid profile.

    These are the velocity spectra, the GW spectral density without the source lifetime factor
    $\tilde{P}_\text{gw,ssm} / \Upsilon_\ell$ and the source integral $\mathcal{I}_v$ of the low-$k$ approximation.
    They can therefore be reused for spectra with different $r_*$.
    The bubble spacing enlargement factor does not enter here, see :py:func:`pttools.ssm.ssm.ubarf2_from_a2`.

    :param inp: input parameters
    :param parallel: whether to use multiple CPU cores
    :return: the quantities that do not depend on $r_*$
    """
    # -----
    # Shared factors
    # -----
    T_tilde = T_tilde_func(T_tilde_min=inp.T_tilde_min, T_tilde_max=inp.T_tilde_max, n=inp.nT)
    qT_lookup = qT_lookup_func(T_tilde=T_tilde, z=inp.y)
    A2 = A2_e_conserving(
        v=inp.v, w=inp.w, xi=inp.xi, e=inp.e, z=qT_lookup,
        v_wall=inp.v_wall, v_sh=inp.v_sh, cs=inp.cs, z_st_thresh=inp.z_st_thresh,
        parallel=parallel, lambda_correction=inp.lambda_correction
    )[0]
    # Todo: Think which z and A2 to use here and in Spectrum.ubarf2_custom_nucleation()
    # The bubble spacing enlargement factor does not enter here, as beta R_* = (8 pi)^(1/3) v_wall
    # also with the nucleation suppression. See ubarf2_from_a2.
    ubarf2 = ubarf2_from_a2(T_tilde=T_tilde, z=qT_lookup, A2=A2, v_wall=inp.v_wall, nuc_type=inp.nuc_type)

    # -----
    # spec_den_v generation
    # -----
    spec_den_v = spec_den_v_func(
        z=inp.y, A2_lookup=A2, qT_lookup=qT_lookup, T_tilde=T_tilde,
        a=inp.lifetime_distribution_a, nuc_type=inp.nuc_type, ubarf2=ubarf2,
        v_wall=inp.v_wall,
        parallel=parallel
    )

    # -----
    # Lookup tables for spec_den_gw
    # -----
    z_gw_lookup = gen_lookup(y=inp.y, cs=inp.cs, n_x_lookup=inp.n_z_lookup, eps=inp.eps_lookup)
    qT_gw_lookup = qT_lookup_func(T_tilde=T_tilde, z=z_gw_lookup)
    A2_gw_lookup = A2_e_conserving(
        v=inp.v, w=inp.w, xi=inp.xi, e=inp.e, z=qT_gw_lookup,
        v_wall=inp.v_wall, v_sh=inp.v_sh, cs=inp.cs, z_st_thresh=inp.z_st_thresh,
        parallel=parallel, lambda_correction=inp.lambda_correction
    )[0]
    spec_den_v_lookup = spec_den_v_func(
        z=z_gw_lookup, A2_lookup=A2_gw_lookup, qT_lookup=qT_gw_lookup, T_tilde=T_tilde,
        a=inp.lifetime_distribution_a, nuc_type=inp.nuc_type, ubarf2=ubarf2,
        v_wall=inp.v_wall,
        parallel=parallel,
    )
    # The source lifetime factor is a mere scaling factor, which is applied in compute_gw().
    spec_den_gw_ssm_unscaled, _ = spec_den_gw(
        z_lookup=z_gw_lookup,
        P_tilde_v_lookup=spec_den_v_lookup,
        y=inp.y,
        cs=inp.cs,
        nx_P_tilde_gw=inp.nx_P_tilde_gw,
        parallel=parallel
    )
    return SSMBaseResults(
        A2=A2,
        A2_lookup=A2_gw_lookup,
        qT_lookup=qT_lookup,
        qT_gw_lookup=qT_gw_lookup,
        spec_den_v=spec_den_v,
        spec_den_v_lookup=spec_den_v_lookup,
        T_tilde=T_tilde,
        ubarf2=ubarf2,
        z_lookup=z_gw_lookup,
        spec_den_gw_ssm_unscaled=spec_den_gw_ssm_unscaled,
        Iv=Iv_resampled(x=z_gw_lookup, P_tilde_v=spec_den_v_lookup)
    )
