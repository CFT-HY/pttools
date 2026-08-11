"""Functions for organizing the computation of the GW spectra"""

import numba

from pttools.speedup import NUMBA_ENABLE_CACHE
from pttools.ssm.spec_den_v import spec_den_v as spec_den_v_func
from pttools.ssm.low_k.integration import \
    power_spectrum_integration_low, power_spectrum_integration_int, Iv_resampled
from pttools.ssm.low_k.join import gw_junction
from pttools.ssm.nucleation import NucType
from pttools.ssm.spec_den_gw import gen_lookup, spec_den_gw
from pttools.ssm.ssm import \
    A2_e_conserving, \
    qT_lookup as qT_lookup_func, \
    T_tilde as T_tilde_func, \
    ubarf2_from_a2
from pttools.type_hints import FloatArr1D


# @numba.njit(nogil=True, cache=NUMBA_ENABLE_CACHE)
def compute(
        # Arrays (in alphabetical order)
        e: FloatArr1D,
        v: FloatArr1D,
        w: FloatArr1D,
        xi: FloatArr1D,
        y: FloatArr1D,
        # Scalars (in alphabetical order)
        bubble_spacing_enlargement_factor: float,
        cs: float,
        lifetime_distribution_a: float,
        nu_gdh2024: float,
        r_star: float,
        source_lifetime_factor: float,
        tau_end: float,
        tau_star: float,
        v_wall: float,
        v_sh: float,
        # Accuracy parameters
        eps_lookup: float,
        nT: int,
        n_z_lookup: int,
        nx_P_tilde_gw: int | None,
        T_tilde_min: float,
        T_tilde_max: float,
        z_st_thresh: float,
        # Other
        nuc_type: NucType,
        lambda_correction: bool = False,
        parallel: bool = True) -> tuple[
            FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D,
            FloatArr1D, float, FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D]:
    """Compute the Sound Shell Model spectra for a fluid profile, including the low-k approximation

    This is in one Numba-compiled function so that the GIL is released for the entire computation.
    """
    spec_den_v, spec_den_v_lookup, spec_den_gw_ssm, \
            A2, A2_lookup, qT_lookup, qT_gw_lookup, T_tilde, ubarf2, z_lookup = compute_ssm(
        e=e, v=v, w=w, xi=xi, y=y,
        bubble_spacing_enlargement_factor=bubble_spacing_enlargement_factor,
        cs=cs, lifetime_distribution_a=lifetime_distribution_a,
        source_lifetime_factor=source_lifetime_factor,
        v_sh=v_sh, v_wall=v_wall,
        eps_lookup=eps_lookup, nT=nT, n_z_lookup=n_z_lookup, nx_P_tilde_gw=nx_P_tilde_gw,
        T_tilde_min=T_tilde_min, T_tilde_max=T_tilde_max, z_st_thresh=z_st_thresh,
        nuc_type=nuc_type, lambda_correction=lambda_correction, parallel=parallel
    )
    spec_den_gw_low, spec_den_gw_int, spec_den_gw_expanded = compute_low_k(
        spec_den_gw_ssm=spec_den_gw_ssm,
        P_tilde_v=spec_den_v,
        P_tilde_v_lookup=spec_den_v_lookup,
        y=y, z_lookup=z_lookup,
        cs=cs, nu_gdh2024=nu_gdh2024, r_star=r_star,
        tau_end=tau_end, tau_star=tau_star
    )
    return \
        spec_den_v, spec_den_v_lookup, spec_den_gw_ssm, \
        A2, A2_lookup, qT_lookup, qT_gw_lookup, T_tilde, ubarf2, z_lookup, \
        spec_den_gw_low, spec_den_gw_int, spec_den_gw_expanded


# @numba.njit(nogil=True)
def compute_low_k(
        # Arrays (in alphabetical order)
        spec_den_gw_ssm: FloatArr1D,
        P_tilde_v: FloatArr1D,
        P_tilde_v_lookup: FloatArr1D,
        y: FloatArr1D,
        z_lookup: FloatArr1D,
        # Scalars (in alphabetical order)
        cs: float,
        nu_gdh2024: float,
        r_star: float,
        tau_end: float,
        tau_star: float) -> tuple[FloatArr1D, FloatArr1D, FloatArr1D]:
    r"""Compute the low-k approximation
    :giombi_2024_cs:`\ `
    """
    spec_den_gw_low = power_spectrum_integration_low(
        x_data=y, Pv_data=P_tilde_v,
        z=y, cs=cs, nu=nu_gdh2024,
        tau_star=tau_star, tau_end=tau_end
    )
    spec_den_gw_int = power_spectrum_integration_int(
        z=y, cs=cs, tau_star=tau_star, Iv=Iv_resampled(x=z_lookup, P_tilde_v=P_tilde_v_lookup)
    )
    spec_den_gw_expanded = gw_junction(
        z=y,
        Pgw_low=spec_den_gw_low,
        Pgw_int=spec_den_gw_int,
        Pgw_high=spec_den_gw_ssm,
        cs=cs, nu=nu_gdh2024,
        tau_star=tau_star, tau_end=tau_end,
        r_star=r_star
    )
    return spec_den_gw_low, spec_den_gw_int, spec_den_gw_expanded


@numba.njit(nogil=True, cache=NUMBA_ENABLE_CACHE)
def compute_ssm(
        # Arrays (in alphabetical order)
        e: FloatArr1D,
        v: FloatArr1D,
        w: FloatArr1D,
        xi: FloatArr1D,
        y: FloatArr1D,
        # Scalars (in alphabetical order)
        bubble_spacing_enlargement_factor: float,
        cs: float,
        lifetime_distribution_a: float,
        source_lifetime_factor: float,
        v_sh: float,
        v_wall: float,
        # Accuracy parameters
        eps_lookup: float,
        nT: int,
        n_z_lookup: int,
        nx_P_tilde_gw: int,
        T_tilde_min: float,
        T_tilde_max: float,
        z_st_thresh: float,
        # Other
        nuc_type: NucType,
        lambda_correction: bool = False,
        parallel: bool = True) -> tuple[
            FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D,
            FloatArr1D, FloatArr1D, FloatArr1D, float, FloatArr1D]:
    """Compute the Sound Shell Model spectra for the given fluid profile"""
    # -----
    # Shared factors
    # -----
    T_tilde = T_tilde_func(T_tilde_min=T_tilde_min, T_tilde_max=T_tilde_max, n=nT)
    qT_lookup = qT_lookup_func(T_tilde=T_tilde, z=y)
    A2 = A2_e_conserving(
        v=v, w=w, xi=xi, e=e, z=qT_lookup,
        v_wall=v_wall, v_sh=v_sh, cs=cs, z_st_thresh=z_st_thresh,
        parallel=parallel, lambda_correction=lambda_correction
    )[0]
    # Todo: Think which z and A2 to use here and in Spectrum.ubarf2_custom_nucleation()
    ubarf2 = ubarf2_from_a2(
        T_tilde=T_tilde, z=qT_lookup, A2=A2, v_wall=v_wall, nuc_type=nuc_type,
        bubble_spacing_enlargement_factor=bubble_spacing_enlargement_factor
    )

    # -----
    # spec_den_v generation
    # -----
    spec_den_v = spec_den_v_func(
        z=y, A2_lookup=A2, qT_lookup=qT_lookup, T_tilde=T_tilde,
        a=lifetime_distribution_a, nuc_type=nuc_type, ubarf2=ubarf2,
        v_wall=v_wall, bubble_spacing_enlargement_factor=bubble_spacing_enlargement_factor,
        parallel=parallel
    )

    # -----
    # spec_den_gw generation
    # -----
    z_gw_lookup = gen_lookup(y=y, cs=cs, n_x_lookup=n_z_lookup, eps=eps_lookup)
    qT_gw_lookup = qT_lookup_func(T_tilde=T_tilde, z=z_gw_lookup)
    A2_gw_lookup = A2_e_conserving(
        v=v, w=w, xi=xi, e=e, z=qT_gw_lookup,
        v_wall=v_wall, v_sh=v_sh, cs=cs, z_st_thresh=z_st_thresh,
        parallel=parallel, lambda_correction=lambda_correction
    )[0]
    spec_den_v_lookup = spec_den_v_func(
        z=z_gw_lookup, A2_lookup=A2_gw_lookup, qT_lookup=qT_gw_lookup, T_tilde=T_tilde,
        a=lifetime_distribution_a, nuc_type=nuc_type, ubarf2=ubarf2,
        v_wall=v_wall, bubble_spacing_enlargement_factor=bubble_spacing_enlargement_factor,
        parallel=parallel,
    )
    spec_den_gw_ssm, y = spec_den_gw(
        z_lookup=z_gw_lookup,
        P_tilde_v_lookup=spec_den_v_lookup,
        y=y,
        cs=cs,
        source_lifetime_factor=source_lifetime_factor,
        nx_P_tilde_gw=nx_P_tilde_gw,
        parallel=parallel
    )
    return spec_den_v, spec_den_v_lookup, spec_den_gw_ssm, \
        A2, A2_gw_lookup, qT_lookup, qT_gw_lookup, T_tilde, ubarf2, z_gw_lookup
