"""Functions for computing the spectral density of the velocity field"""

import logging

import numba
import numpy as np

from pttools import speedup
from pttools.ssm import const, ssm
from pttools.ssm.nucleation import NucType, beta_R_star0, nu
import pttools.type_hints as th

logger = logging.getLogger(__name__)


@numba.njit
def qT_from_z(
        z: th.FloatOrArr,
        T_tilde: th.FloatOrArr,
        beta_R: th.FloatOrArr) -> th.FloatOrArr:
    r"""$qT$
    $$qT = \frac{z \tilde{T}}{\beta R_*} = \frac{\tilde{T}q}{\beta}$$
    where $z = q L_f = q R_*$
    """
    return z * T_tilde / beta_R


@numba.njit
def _spec_den_v_core_loop(
        # Arrays
        A2_lookup: th.FloatArr1D,
        qT_lookup: th.FloatArr1D,
        T_tilde: th.FloatArr1D,
        # Scalars
        a: float,
        beta_R: float,
        factor: float,
        nuc_type: NucType,
        z_i: float) -> float:
    """spec_den_v for an individual z"""
    # The argument of A(qT)
    qT = qT_from_z(z_i, T_tilde, beta_R)
    # |A(qT)|^2
    A2 = np.interp(qT, qT_lookup, A2_lookup)
    integrand = T_tilde ** 6 * nu(T_tilde, nuc_type, a) * A2
    # Integrate over T_tilde
    integral = np.trapezoid(integrand, T_tilde)
    return factor * integral


def _spec_den_v_core(
        # Arrays
        A2_lookup: th.FloatArr1D,
        qT_lookup: th.FloatArr1D,
        z: th.FloatArr1D,
        # Scalars
        a: float,
        bubble_spacing_enlargement_factor: float,
        log10_T_tilde_min: float,
        log10_T_tilde_max: float,
        nT: int,
        nuc_type: NucType,
        ubarf2: float,
        v_wall: float) -> th.FloatArr1D:
    """Parallel core of spec_den_v"""
    # $\beta R_*$ = beta, but without dividing by R_* in its equation
    # The choice of beta is somewhat arbitrary.
    # It has been chosen to correspond to the nucleation rate (beta), and is therefore called beta as well.
    # However, please note that R_* must correspond to the value used elsewhere.
    # Todo: The power of the bubble spacing enlargement factor needs to be looked into.
    # The power of 3 coming from the internal beta_R may be scaled away, leaving only a power of 3.
    beta_R = beta_R_star0(v_wall) / bubble_spacing_enlargement_factor
    factor = 1. / (ubarf2 * beta_R ** 6)

    T_tilde = speedup.logspace(log10_T_tilde_min, log10_T_tilde_max, nT)

    sd_v = np.empty_like(z)
    for i in numba.prange(z.size):  # pylint: disable=not-an-iterable
        sd_v[i] = _spec_den_v_core_loop(
            A2_lookup=A2_lookup, qT_lookup=qT_lookup, T_tilde=T_tilde,
            a=a, beta_R=beta_R, factor=factor, nuc_type=nuc_type, z_i=z[i]
        )
    return sd_v

spec_den_v_core = numba.njit(parallel=True, nogil=True)(_spec_den_v_core)
spec_den_v_core_single = numba.njit(parallel=False, nogil=True)(_spec_den_v_core)


@numba.njit(nogil=True)
def spec_den_v(
        # Arrays
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        e: th.FloatArr1D,
        z: th.FloatArr1D,
        # Scalars and other inputs
        a: float,
        cs: float,
        nuc_type: NucType,
        ubarf2: float,
        v_sh: float,
        v_wall: float,
        bubble_spacing_enlargement_factor: float = 1.,
        # Settings
        nT: int = const.DEFAULT_N_T,
        z_st_thresh: float = const.Z_ST_THRESH,
        T_tilde_min: float = const.T_TILDE_MIN,
        T_tilde_max: float = const.T_TILDE_MAX,
        parallel: bool = True,
        lambda_correction: bool = False) -> tuple[th.FloatArr1D, th.FloatArr1D]:
    r"""Spectral density of the velocity field $\tilde{P}_v$

    $$\tilde{P}_v(q)
    = \frac{1}{\bar{U}_f^2 R_{\ast}^3} P_v(q)
    = \frac{1}{\bar{U}_f^2 (\beta R_{\ast})^6} \int d\tilde{T} \nu(\tilde{T}) \tilde{T}^6
    \left| A \left( \frac{\tilde{T}q}{\beta} \right) \right|^2
    = \frac{\Lambda_\text{nucl}^6}{\bar{U}_f^2 (\beta R_{\ast,0})^6} \int d\tilde{T} \nu(\tilde{T}) \tilde{T}^6
    \left| A \left( \frac{\tilde{T}q}{\beta} \right) \right|^2$$

    Please note that
    $$P_v(q) = L_f^3 \bar{U}_f^2 \tilde{P}_v(qL_f)$$
    :gw_pt_ssm:`\ ` eq. 3.43
    and therefore
    $$\tilde{P}_v = \frac{P_v}{L_f^3 \bar{U}_f^2}$$

    $$P_v(q) = \frac{1}{\beta^6 R_*^3} \int d\tilde{T}
    \nu(\tilde{T}) \tilde{T}^6
    \left| A \left( \frac{\tilde{T}q}{\beta} \right) \right|^2$$
    :gw_pt_ssm:`\ ` eq. 4.17

    $$\Lambda_\text{nucl} \equiv \frac{R_{\ast}}{R_{\ast,0}}$$
    :ajmi_2022:`\ ` eq. 77

    :param v: velocity profile $v$
    :param w: enthalpy profile $w$
    :param xi: $\xi$ of the profiles
    :param e: energy density profile $e$
    :param z: wavenumber range $z$
    :param v_wall: wall speed $v_\text{wall}$
    :param v_sh: shock speed $v_\text{sh}$
    :param a: a multiplier for the bubble lifetime distribution $\nu$
    :param nuc_type: nucleation type
    :param nT: number of $T$ points for the integration
    :param z_st_thresh: limit above which to use approximate sin transform
    :param T_tilde_min: minimum $\tilde{T}$ for the integration
    :param T_tilde_max: maximum $\tilde{T}$ for the integration
    :param cs: speed of sound $c_s$
    :param parallel: whether to compute the result for each $z$ in parallel
    :param lambda_correction: whether to enable a non-linear correction for $\lambda$
    :return: $\tilde{P}_{\tilde{v}}$
    """
    # z limits
    log10_z_min = np.log10(np.min(z))
    log10_z_max = np.log10(np.max(z))

    # T limits
    log10_T_min = np.log10(T_tilde_min)
    log10_T_max = np.log10(T_tilde_max)

    # try:
    # Todo: Check whether this could be replaced by logspace
    qT_lookup = 10 ** np.arange(
        log10_z_min + log10_T_min,
        log10_z_max + log10_T_max,
        step=(log10_z_max - log10_z_min) / z.size
    )
    # except ValueError as e:
    #     logger.error(
    #         "Could not compute qT_lookup with log10_z_min=%s, log10_T_min=%s, log10_z_max=%s, log10_T_max=%s, dlog10z=%s",
    #         log10_z_min, log10_T_min, log10_z_max, log10_T_max, dlog10z
    #     )
    #     raise e
    A2_lookup = ssm.a2_e_conserving(
        v=v, w=w, xi=xi, e=e, z=qT_lookup,
        v_wall=v_wall, v_sh=v_sh, cs=cs, z_st_thresh=z_st_thresh,
        parallel=parallel, lambda_correction=lambda_correction
    )[0]
    # if qT_lookup.size != A2_lookup.size:
    #     raise ValueError(f"Lookup sizes don't match: {qT_lookup.size} != {A2_lookup.size}")

    if parallel:
        ret = spec_den_v_core(
            A2_lookup=A2_lookup,
            qT_lookup=qT_lookup,
            z=z,
            a=a,
            bubble_spacing_enlargement_factor=bubble_spacing_enlargement_factor,
            log10_T_tilde_min=log10_T_min,
            log10_T_tilde_max=log10_T_max,
            nT=nT,
            nuc_type=nuc_type,
            ubarf2=ubarf2,
            v_wall=v_wall
        )
    else:
        ret = spec_den_v_core_single(
            A2_lookup=A2_lookup,
            qT_lookup=qT_lookup,
            z=z,
            a=a,
            bubble_spacing_enlargement_factor=bubble_spacing_enlargement_factor,
            log10_T_tilde_min=log10_T_min,
            log10_T_tilde_max=log10_T_max,
            nT=nT,
            nuc_type=nuc_type,
            ubarf2=ubarf2,
            v_wall=v_wall
        )
    return ret, A2_lookup
