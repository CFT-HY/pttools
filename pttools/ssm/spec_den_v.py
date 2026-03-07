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
def _qT_array(qRstar, Ttilde, b_R, v_wall):
    return qRstar * Ttilde / (b_R * v_wall)


@numba.njit
def _spec_den_v_core_loop(
        z_i: float,
        t_array: th.FloatArr1D,
        b_R: float,
        v_wall: float,
        qT_lookup: th.FloatArr1D,
        A2_lookup: th.FloatArr1D,
        nuc_type: NucType,
        a: float,
        factor: float):
    """Inner loop of spec_den_v_core"""
    qT = _qT_array(z_i, t_array, b_R, v_wall)
    A2_2d_array_z = np.interp(qT, qT_lookup, A2_lookup)
    array2 = t_array ** 6 * nu(t_array, nuc_type, a) * A2_2d_array_z
    D = np.trapezoid(array2, t_array)
    return D * factor


def _spec_den_v_core(
        a: float,
        A2_lookup: th.FloatArr1D,
        log10tmin: float,
        log10tmax: float,
        nuc_type: NucType,
        nt: int,
        qT_lookup: th.FloatArr1D,
        v_wall: float,
        z: th.FloatArr1D):
    """Numba-jitted core of spec_den_v"""
    t_array = speedup.logspace(log10tmin, log10tmax, nt)
    b_R = (8. * np.pi) ** (1. / 3.)  # $\beta R_* = b_R v_w $

    # A2_2d_array = np.zeros((nz, nt))

    # array2 = np.zeros(nt)
    sd_v = np.zeros(z.size)  # array for spectral density of v
    factor = 1. / (b_R * v_wall) ** 6
    factor = 2 * factor  # because spectral density of v is 2 * P_v

    for i in numba.prange(z.size):  # pylint: disable=not-an-iterable
        sd_v[i] = _spec_den_v_core_loop(z[i], t_array, b_R, v_wall, qT_lookup, A2_lookup, nuc_type, a, factor)

    return sd_v

spec_den_v_core = numba.njit(nogil=True)(_spec_den_v_core)
spec_den_v_core_single = numba.njit(parallel=False, nogil=True)(_spec_den_v_core)


@numba.njit
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
        nt: int = const.NPTDEFAULT[1],
        z_st_thresh: float = const.Z_ST_THRESH,
        cs: float | None = None,
        parallel: bool = True):
    r"""The full spectral density of the velocity field

    This is twice the spectral density of the plane wave components of the velocity field

    :return: $P_{\tilde{v}} = 2 * P_v(q)$ of :gw_pt_ssm:`\ ` eq. 4.17
    """
    # z limits
    log10zmin = np.log10(np.min(z))
    log10zmax = np.log10(np.max(z))
    dlog10z = (log10zmax - log10zmin) / z.size

    # t limits
    tmin = const.T_TILDE_MIN
    tmax = const.T_TILDE_MAX
    log10tmin = np.log10(tmin)
    log10tmax = np.log10(tmax)

    # try:
    qT_lookup = 10 ** np.arange(log10zmin + log10tmin, log10zmax + log10tmax, dlog10z)
    # except ValueError as e:
    #     logger.error(
    #         "Could not compute qT_lookup with log10zmin=%s, log10tmin=%s, log10zmax=%s, log10tmax=%s, dlog10z=%s",
    #         log10zmin, log10tmin, log10zmax, log10tmax, dlog10z
    #     )
    #     raise e
    A2_lookup = ssm.a2_e_conserving(
        v=v, w=w, xi=xi, e=e, z=qT_lookup,
        v_wall=v_wall, v_sh=v_sh, cs=cs, z_st_thresh=z_st_thresh, parallel=parallel
    )[0]
    # if qT_lookup.size != A2_lookup.size:
    #     raise ValueError(f"Lookup sizes don't match: {qT_lookup.size} != {A2_lookup.size}")

    if parallel:
        ret = spec_den_v_core(
            a=a,
            A2_lookup=A2_lookup,
            log10tmin=log10tmin,
            log10tmax=log10tmax,
            nt=nt,
            nuc_type=nuc_type,
            qT_lookup=qT_lookup,
            v_wall=v_wall,
            z=z
        )
    else:
        ret = spec_den_v_core_single(
            a=a,
            A2_lookup=A2_lookup,
            log10tmin=log10tmin,
            log10tmax=log10tmax,
            nt=nt,
            nuc_type=nuc_type,
            qT_lookup=qT_lookup,
            v_wall=v_wall,
            z=z
        )
    return ret, A2_lookup
