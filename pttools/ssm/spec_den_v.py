"""Functions for computing the spectral density of the velocity field"""

import logging

import numba
import numpy as np

from pttools import speedup
from pttools.ssm import const, ssm
from pttools.ssm.nucleation import NucType, nu
import pttools.type_hints as th

logger = logging.getLogger(__name__)


@numba.njit
def qT_from_z(
        z: th.FloatOrArr,
        T_tilde: th.FloatOrArr,
        beta_R: th.FloatOrArr) -> th.FloatOrArr:
    """
    $$qT = \frac{z \tilde{T}}{\beta R_*} = \frac{\tilde{T}q}{\beta}$$
    where $z = q L_f = q R_*$
    """
    return z * T_tilde / beta_R


@numba.njit
def _spec_den_v_core_loop(
        z_i: float,
        T_tilde: th.FloatArr1D,
        beta_R: float,
        qT_lookup: th.FloatArr1D,
        A2_lookup: th.FloatArr1D,
        nuc_type: NucType,
        a: float,
        factor: float) -> float:
    """P_{\tilde{v}}(z) for an individual z"""
    # The argument of A(qT)
    qT = qT_from_z(z_i, T_tilde, beta_R)
    # |A(qT)|^2
    A2 = np.interp(qT, qT_lookup, A2_lookup)
    integrand = T_tilde ** 6 * nu(T_tilde, nuc_type, a) * A2
    # Integrate over T_tilde
    integral = np.trapezoid(integrand, T_tilde)
    return factor * integral


def _spec_den_v_core(
        a: float,
        A2_lookup: th.FloatArr1D,
        log10_T_tilde_min: float,
        log10_T_tilde_max: float,
        nuc_type: NucType,
        nT: int,
        qT_lookup: th.FloatArr1D,
        v_wall: float,
        z: th.FloatArr1D):
    """Parallel core of spec_den_v"""
    T_tilde = speedup.logspace(log10_T_tilde_min, log10_T_tilde_max, nT)
    # $\beta R_*$ = beta without dividing by R_*
    # The choice of this function is somewhat arbitrary,
    # and it just happens to be the same as beta in general (and is therefore called with the same name).
    beta_R = (8. * np.pi) ** (1. / 3.) * v_wall

    # Spectral density of v
    sd_v = np.empty_like(z)
    # The 2 comes from the fact that the spectral density of v is 2 * P_v
    factor = 2 / (beta_R ** 6)

    for i in numba.prange(z.size):  # pylint: disable=not-an-iterable
        sd_v[i] = _spec_den_v_core_loop(
            z_i=z[i], T_tilde=T_tilde, beta_R=beta_R,
            qT_lookup=qT_lookup, A2_lookup=A2_lookup,
            nuc_type=nuc_type, a=a, factor=factor
        )
    return sd_v

spec_den_v_core = numba.njit(parallel=True, nogil=True)(_spec_den_v_core)
spec_den_v_core_single = numba.njit(parallel=False, nogil=True)(_spec_den_v_core)


@numba.njit(nogil=True)
def spec_den_v(
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        e: th.FloatArr1D,
        z: th.FloatArr1D,
        v_wall: float,
        v_sh: float,
        a: float,
        nuc_type: NucType,
        nT: int = const.DEFAULT_N_T,
        z_st_thresh: float = const.Z_ST_THRESH,
        T_tilde_min: float = const.T_TILDE_MIN,
        T_tilde_max: float = const.T_TILDE_MAX,
        cs: float | None = None,
        parallel: bool = True,
        lambda_correction: bool = False):
    r"""The full spectral density of the velocity field

    This is twice the spectral density of the plane wave components of the velocity field, and therefore given by
    $$P_{\tilde{v}}
    = 2 * P_v(q)
    = 2 \frac{1}{\beta^6 R_*^3} \int d\tilde{T} \nu(\tilde{T}) \tilde{T}^6
    \left| A \left( \frac{\tilde{T}q}{\beta} \right) \right|^2$$
    :gw_pt_ssm:`\ ` eq. 4.17

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
    :return: $P_{\tilde{v}} = 2 * P_v(q)$
    """
    # z limits
    log10_z_min = np.log10(np.min(z))
    log10_z_max = np.log10(np.max(z))

    # T limits
    log10_T_min = np.log10(T_tilde_min)
    log10_T_max = np.log10(T_tilde_max)

    # try:
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
            a=a,
            A2_lookup=A2_lookup,
            log10_T_tilde_min=log10_T_min,
            log10_T_tilde_max=log10_T_max,
            nT=nT,
            nuc_type=nuc_type,
            qT_lookup=qT_lookup,
            v_wall=v_wall,
            z=z
        )
    else:
        ret = spec_den_v_core_single(
            a=a,
            A2_lookup=A2_lookup,
            log10_T_tilde_min=log10_T_min,
            log10_T_tilde_max=log10_T_max,
            nT=nT,
            nuc_type=nuc_type,
            qT_lookup=qT_lookup,
            v_wall=v_wall,
            z=z
        )
    return ret, A2_lookup
