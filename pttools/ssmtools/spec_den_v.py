"""Functions for computing the spectral density of the velocity field"""

import logging

import numba
import numpy as np

from pttools import bubble
from pttools import speedup
from pttools.ssmtools import const, ssm
from pttools.ssmtools.nucleation import NucType, nu

logger = logging.getLogger(__name__)


@numba.njit
def _qT_array(qRstar, Ttilde, b_R, vw):
    return qRstar * Ttilde / (b_R * vw)


@numba.njit
def _spec_den_v_core_loop(
        z_i: float, t_array: np.ndarray, b_R: float, vw: float,
        qT_lookup: np.ndarray, A2_lookup: np.ndarray, nuc_type: NucType, a: float, factor: float):
    qT = _qT_array(z_i, t_array, b_R, vw)
    A2_2d_array_z = np.interp(qT, qT_lookup, A2_lookup)
    array2 = t_array ** 6 * nu(t_array, nuc_type, a) * A2_2d_array_z
    D = np.trapezoid(array2, t_array)
    return D * factor


@numba.njit(parallel=True, nogil=True)
def spec_den_v_core(
        a: float,
        A2_lookup: np.ndarray,
        log10tmin: float,
        log10tmax: float,
        nuc_type: NucType,
        nt: int,
        qT_lookup: np.ndarray,
        vw: float,
        z: np.ndarray):
    t_array = speedup.logspace(log10tmin, log10tmax, nt)
    b_R = (8. * np.pi) ** (1. / 3.)  # $\beta R_* = b_R v_w $

    # A2_2d_array = np.zeros((nz, nt))

    # array2 = np.zeros(nt)
    sd_v = np.zeros(z.size)  # array for spectral density of v
    factor = 1. / (b_R * vw) ** 6
    factor = 2 * factor  # because spectral density of v is 2 * P_v

    for i in numba.prange(z.size):
        sd_v[i] = _spec_den_v_core_loop(z[i], t_array, b_R, vw, qT_lookup, A2_lookup, nuc_type, a, factor)

    return sd_v


def spec_den_v(
        bub: bubble.Bubble,
        z: np.ndarray,
        a: float,
        nuc_type: NucType,
        nt: int = const.NPTDEFAULT[1],
        z_st_thresh: float = const.Z_ST_THRESH,
        cs: float | None = None,
        return_a2: bool = False):
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

    qT_lookup = 10 ** np.arange(log10zmin + log10tmin, log10zmax + log10tmax, dlog10z)
    A2_lookup = ssm.a2_e_conserving(bub=bub, z=qT_lookup, cs=cs, z_st_thresh=z_st_thresh)[0]
    # if qT_lookup.size != A2_lookup.size:
    #     raise ValueError(f"Lookup sizes don't match: {qT_lookup.size} != {A2_lookup.size}")

    ret = spec_den_v_core(
        a=a,
        A2_lookup=A2_lookup,
        log10tmin=log10tmin,
        log10tmax=log10tmax,
        nt=nt,
        nuc_type=nuc_type,
        qT_lookup=qT_lookup,
        vw=bub.v_wall,
        z=z
    )
    if return_a2:
        return ret, A2_lookup
    return ret
