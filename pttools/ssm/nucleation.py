"""Bubble nucleation"""

import enum
import logging

import numba
import numpy as np

from pttools.bubble.solution_type import SolutionType
from pttools.ssm import const
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr

logger = logging.getLogger(__name__)


@enum.unique
class NucType(str, enum.Enum):
    """Nucleation type"""
    EXPONENTIAL = "exponential"
    SIMULTANEOUS = "simultaneous"


#: Default nucleation type
DEFAULT_NUC_TYPE = NucType.EXPONENTIAL


@numba.njit
def beta(R_star: th.FloatOrArr, v_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""Nucleation rate parameter $\beta$, aka. inverse phase transition duration

    $$\beta = (8\pi)^\frac{1}{3} \frac{{v}_\text{wall}}{R_*}$$
    :gw_pt_ssm:`\ ` eq. 4.16, A.14
    :notes:`\ ` eq. 7.21

    This does not take into account the nucleation suppression.
    Please see :py:func:`pttools.bubble.nucleation.R_star` for further information.

    :param R_star: Mean bubble separation $R_*$
    :param v_wall: Wall velocity $v_w$
    :return: Inverse phase transition duration $\beta$
    """
    return (8 * np.pi)**(1/3) * v_wall / R_star


def beta_tilde(
        r_star: th.FloatOrArr,
        v_wall: th.FloatOrArr,
        beta_tilde_limit: float = const.BETA_TILDE_CONVERSION_MIN) -> th.FloatOrArr:
    r"""Nucleation rate parameter $\tilde{\beta}$, aka. "beta over H"
    $$\tilde{\beta} \equiv \frac{\beta}{H_*} = (8 \pi)^\frac{1}{3} \frac{\max ({v}_\text{wall}, c_s)}{{r}_*}$$
    :gowling_2021:`\ ` eq. 2.1

    This does not take into account the nucleation suppression.
    Please see :py:func:`pttools.bubble.nucleation.beta` and
    :py:func:`pttools.bubble.nucleation.r_star` for further information.

    :param r_star: Hubble-scaled mean bubble spacing $r_*$
    :param v_wall: Wall velocity $v_w$
    :return: Nucleation rate parameter $\tilde{\beta}$
    """
    b = beta(R_star=r_star, v_wall=v_wall)
    b_min = np.min(b)
    if b_min < beta_tilde_limit:
        logger.warning(
            "Got β/H*=%s < %s for r*=%s, v_wall=%s. "
            "The conversion to from r* to β/H* may not have been accurate, "
            "as this seems to be a very slow phase transition. Please see Caprini et al. (2020) p. 6.",
            b, beta_tilde_limit, r_star, v_wall
        )
    return b


@numba.njit
def beta_R_star0[T: FloatOrArr](v_wall: T) -> T:
    r"""$\beta R_{\ast,0}$
    $$\beta R_{\ast,0} = (8 \pi)^\frac{1}{3} v_{\text{wall}}$$
    This is a direct consequence of :py:func:beta:.
    This does not take into account the nucleation suppression.

    This is used in :py:func:spec_den_gw_scaled:,
    where $\beta$ is an arbitrary rate that is taken to be the nucleation rate parameter $\beta$ by convenience,
    and therefore does not need to take into account nucleation suppression.
    """
    return (8. * np.pi) ** (1. / 3.) * v_wall


def bubble_spacing_enlargement_factor[T: FloatOrArr](hx: T) -> T:
    r"""Bubble spacing enlargement factor $\Lambda$
    $$\Lambda(h_x) \equiv \frac{R_{\ast}}{R_{\ast}(0)} = I_h^{-\frac{1}{3}}(h_x)$$
    :ajmi_2022:`\ ` eq. 77
    """
    return Ih_approx(hx)**(-1/3)


def hx[T: FloatOrArr](f: T) -> T:
    r"""Fractional volume $h_x$ at which the symmetric phase is reheated enough to prevent further bubble nucleation
    $$h_x = \frac{f}{1 + f} = 1 - \frac{v_{\text{wall}}^3}{v_{\text{eff}}^3}$$
    :ajmi_2022:`\ ` eq. 56
    """
    return f / (1 + f)


def Ih_approx[T: FloatOrArr](hx: T) -> T:
    r"""Approximate $I_h(h_x)$
    $$I_h(h_x) = 1 + \frac{h_x \ln h_x}{1 - h_x}$$
    :ajmi_2022:`\ ` eq. 78
    """
    return 1 + (hx * np.log(hx)) / (1 - hx)


@numba.njit
def nu[T: FloatOrArr](T_tilde: T, nuc_type: NucType = NucType.SIMULTANEOUS, a: float = 1.) -> T:
    r"""
    Bubble lifetime distribution function

    This is normalized so that
    $$\int \nu(x) dx = 1$$
    :gw_pt_ssm:`\ ` p. 17

    For exponential nucleation:
    $$\nu_\text{exp} (\tilde{T}) = a e^{-a\tilde{T}}$$
    :gw_pt_ssm:`\ ` eq. 4.27

    For simultaneous nucleation:
    $$\nu_\text{sim} (\tilde{T}) = \frac{1}{2} a (a \tilde{T})^2 e^{\left( - \frac{1}{6} \tilde{aT}^3 \right)}$$
    :gw_pt_ssm:`\ ` eq. 4.32

    :param T_tilde: dimensionless time $\tilde{T}$
    :param nuc_type: nucleation type, simultaneous or exponential
    :param a: normalization factor $a$
        (This is for debugging, and $\nu$ should be normalized regardless of its value.)
    :return: bubble lifetime distribution $\nu$
    """
    # The exponential and simultaneous functions have been verified to be properly normalized regardless of a.
    if nuc_type == NucType.EXPONENTIAL.value:
        return a * np.exp(-a * T_tilde)
    if nuc_type == NucType.SIMULTANEOUS.value:
        return 0.5 * a * (a * T_tilde)**2 * np.exp(-(a * T_tilde) ** 3 / 6)
    raise ValueError(f"Nucleation type not recognized: \"{nuc_type}\"")


def nucleation_f(xi: th.FloatArr1D, T: th.FloatArr1D, beta_tilde: float, v_wall: float, v_sh: float):
    r"""Relative increase $f$ in the effective volume of the bubble
    $$f = \frac{3}{v_{\text{wall}}^3} \int_{v_{\text{wall}}}^{v_{\text{sh}}} \xi^2
    \left( 1 - e^{-\Delta S} \right) d\xi$$,
    where
    $$\Delta S
    \approx \frac{\partial S}{\partial t}
    = \frac{\partial t}{\partial T} \Delta T
    = \frac{\tilde{\beta} \Delta T(\xi)}{T_n}$$
    :ajmi_2022:`\ ` eq. 47-50
    """
    # Todo: Integrate with respect to xi^3
    inds = np.logical_and(v_wall < xi, xi < v_sh)
    xi_cut = xi[inds]
    return 3 / v_wall**3 * np.trapezoid(xi_cut**2 * (1 - np.exp(-beta_tilde * (T[inds] - T[-1]) / T[-1])), xi_cut)


def r_star(
        beta_over_H: float,
        v_wall: float,
        xi: th.FloatArr1D,
        T: th.FloatArr1D,
        sol_type: SolutionType) -> th.FloatOrArr:
    r"""Hubble-scaled mean bubble spacing $r_*(\beta)
    $$r_* = \Lambda(h_x) r_*(0)$$
    :ajmi_2022:`\ ` eq. 77
    Please see :py:func:`pttools.bubble.nucleation.R_star` for further information.
    """
    # if beta_over_H < beta_over_H_limit:
    #     logger.warning(
    #         "Got β/H*=%s < %s (and v_wall=%s). "
    #         "The conversion from β/H* to r* may not be accurate, as this seems to be a very slow phase transition. "
    #         "Please see Caprini et al. (2020) p. 6.",
    #         beta_over_H, beta_over_H_limit, v_wall
    #     )
    return R_star(beta=beta_over_H, v_wall=v_wall, xi=xi, T=T, beta_tilde=beta_over_H, sol_type=sol_type)


def r_star0(beta_over_H: th.FloatOrArr, v_wall: th.FloatOrArr):
    r"""Hubble-scaled mean bubble separation $r_*(0)$ in the absence of nucleation suppression
    $$r_* = (8\pi)^\frac{1}{3} \frac{{v}_\text{wall}}{\tilde{\beta}}$$
    :ajmi_2022:`\ ` eq. 1
    Please see :py:func:`pttools.bubble.nucleation.R_star0` for further information.
    """
    return R_star0(beta=beta_over_H, v_wall=v_wall)


def r_star_product(H_star: th.FloatOrArr, R_star: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Hubble-scaled mean bubble spacing $r_*$
    $$r_* = H_* R_*$$
    :gowling_2021:`\ ` eq. 2.2
    """
    return H_star * R_star


def R_star(
        beta: float,
        v_wall: float,
        xi: th.FloatArr1D,
        T: th.FloatArr1D,
        beta_tilde: float,
        sol_type: SolutionType) -> float:
    r"""Mean bubble separation $R_*$
    $$R_* = \Lambda(h_x) R_*(0)$$
    :ajmi_2022:`\ ` eq. 77

    This commonly used formula is wrong.
    $$R_* = \frac{(8\pi)^\frac{1}{3}}{\beta} \max ({v}_\text{wall}, c_s)$$
    These sources use this formula:
    :hindmarsh_2015:`\ ` p. 4
    :caprini_2020:`\ ` eq. 6
    :hakkinen_msc:`\ ` eq. 2.6

    For detonations (${v}_\text{wall} > c_s$),
    the typical separation between bubbles is set by the wall velocity $v_\text{wall}$.
    For bubbles expanding as deflagrations (${v}_\text{wall} < c_s$),
    the reheating of the plasma by the reaction front can suppress further bubble formation for large enough $\alpha$.
    This is often approximated by $\max ({v}_\text{wall}, c_s)$, but this is wrong.
    Especially for very slow PTs ($\frac{\beta}{H_*} \approx 1$),
    this approximation breaks down,
    and the mean bubble separation $R_*$ must be calculated directly from first principles.
    In such cases one may also want to take into account the expansion of the universe during the phase transition.
    :caprini_2020:`\ ` p. 6
    :gowling_2021:`\ ` p. 5
    :enqvist_1992:`\ ` eq. 4.10
    """
    if sol_type == SolutionType.DETON:
        return R_star0(beta, v_wall)
    elif sol_type in (SolutionType.SUB_DEF, SolutionType.HYBRID):
        f = nucleation_f(xi, T, beta_tilde, v_wall)
        return bubble_spacing_enlargement_factor(hx=hx(f)) * R_star0(beta, v_wall)
    raise ValueError(f"Invalid solution type: {sol_type}")


def R_star0(beta: th.FloatOrArr, v_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""Mean bubble separation $R_*(0)$ in the absence of nucleation suppression
    $$R_*(0) = n_*^{-\frac{1}{3}} = \frac{(8\pi)^\frac{1}{3}}{\beta} {v}_\text{wall}$$
    :ajmi_2022:`\ ` eq. 1

    :param beta: Nucleation rate parameter $\beta$
    :param v_wall: Wall velocity ${v}_w$
    :return: Mean bubble separation $R_*$
    """
    return (8 * np.pi)**(1/3) * v_wall / beta
