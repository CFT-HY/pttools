"""Functions for computing the spectral density of the velocity field"""

import logging

import numba
import numpy as np

from pttools.speedup import njit
from pttools.ssm.nucleation import NucType, beta_R_star0, lifetime_distribution
import pttools.type_hints as th

logger = logging.getLogger(__name__)


@njit(cache=True)
def qT_from_z(
        z: th.FloatOrArr,
        T_tilde: th.FloatOrArr,
        beta_R: th.FloatOrArr) -> th.FloatOrArr:
    r"""$qT$
    $$qT = \frac{z \tilde{T}}{\beta R_*} = \frac{\tilde{T}q}{\beta}$$
    where $z = q L_f = q R_*$
    """
    return z * T_tilde / beta_R


@njit
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
    integrand = T_tilde ** 6 * lifetime_distribution(T_tilde, nuc_type, a) * A2
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
        nuc_type: NucType,
        T_tilde: th.FloatArr1D,
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

    sd_v = np.empty_like(z)
    for i in numba.prange(z.size):  # pylint: disable=not-an-iterable
        sd_v[i] = _spec_den_v_core_loop(
            A2_lookup=A2_lookup, qT_lookup=qT_lookup, T_tilde=T_tilde,
            a=a, beta_R=beta_R, factor=factor, nuc_type=nuc_type, z_i=z[i]
        )
    return sd_v

spec_den_v_core = njit(parallel=True, nogil=True)(_spec_den_v_core)
spec_den_v_core_single = njit(parallel=False, nogil=True)(_spec_den_v_core)


@njit(nogil=True)
def spec_den_v(
        # Arrays
        z: th.FloatArr1D,
        A2_lookup: th.FloatArr1D,
        qT_lookup: th.FloatArr1D,
        T_tilde: th.FloatArr1D,
        # Scalars and other inputs
        a: float,
        nuc_type: NucType,
        ubarf2: float,
        v_wall: float,
        bubble_spacing_enlargement_factor: float = 1.,
        # Settings
        parallel: bool = True) -> tuple[th.FloatArr1D, th.FloatArr1D]:
    r"""Spectral density of the velocity field $\tilde{P}_v$

    $$\tilde{P}_v(q)
    = \frac{1}{\bar{U}_f^2 R_{\ast}^3} P_v(q)
    = \frac{1}{\bar{U}_f^2 (\beta R_{\ast})^6} \int d\tilde{T} \nu(\tilde{T}) \tilde{T}^6
    \left| A \left( \frac{\tilde{T}q}{\beta} \right) \right|^2
    = \frac{\Lambda_\text{nucl}^6}{\bar{U}_f^2 (\beta R_{\ast,0})^6} \int d\tilde{T} \nu(\tilde{T}) \tilde{T}^6
    \left| A \left( \frac{\tilde{T}q}{\beta} \right) \right|^2$$

    The bubble spacing enlargement factor $\Lambda_\text{nucl}$ also affects
    :py:func:pttools.ssm.ssm.ubarf2_from_a2:.

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

    :param z: wavenumber range $z$
    :param A2_lookup: lookup array for $|A(z)|^2$
    :param qT_lookup: lookup array for $qT$
    :param T_tilde: $\tilde{T}$
    :param v_wall: wall speed $v_\text{wall}$
    :param a: a multiplier for the bubble lifetime distribution $\nu$
    :param nuc_type: nucleation type
    :param ubarf2: $\bar{U}_f^2$, computed with the nucleation history as in :gw_pt_ssm:`\ ` eq. 4.33, not eq. B.30
    :param parallel: whether to compute the result for each $z$ in parallel
    :return: $\tilde{P}_{\tilde{v}}$
    """
    if A2_lookup.shape != qT_lookup.shape:
        raise TypeError(
            "A2_lookup and qT_lookup must be of the same shape. "
            f"Got A2_lookup.shape={A2_lookup.shape}, qT_lookup.shape={qT_lookup.shape}"
        )

    if parallel:
        ret = spec_den_v_core(
            A2_lookup=A2_lookup,
            qT_lookup=qT_lookup,
            z=z,
            a=a,
            bubble_spacing_enlargement_factor=bubble_spacing_enlargement_factor,
            nuc_type=nuc_type,
            T_tilde=T_tilde,
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
            nuc_type=nuc_type,
            T_tilde=T_tilde,
            ubarf2=ubarf2,
            v_wall=v_wall
        )
    return ret
