"""Bubble nucleation"""

import enum
import logging

import numba
import numpy as np

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
def beta(R_star: th.FloatOrArr, v_wall: th.FloatOrArr, cs: th.FloatOrArr = const.CS0) -> th.FloatOrArr:
    r"""Nucleation rate parameter $\beta$, aka. inverse phase transition duration

    $$\beta = (8\pi)^\frac{1}{3} \frac{\max ({v}_\text{wall}, c_s)}{R_*}$$
    Inverted from :caprini_2020:`\ ` eq. 6

    Without $c_s$:
    :gw_pt_ssm:`\ ` eq. 4.16, A.14
    :notes:`\ ` eq. 7.21

    Please see :py:func:`pttools.bubble.nucleation.R_star` for further information.

    :param R_star: Mean bubble separation $R_*$
    :param v_wall: Wall velocity $v_w$
    :param cs: Sound speed $c_s$
    :return: Inverse phase transition duration $\beta$
    """
    return (8 * np.pi)**(1/3) * np.maximum(v_wall, cs) / R_star


def beta_over_H(
        r_star: th.FloatOrArr,
        v_wall: th.FloatOrArr,
        cs: th.FloatOrArr = const.CS0,
        beta_over_H_limit: float = const.BETA_OVER_H_LIMIT) -> th.FloatOrArr:
    r"""Nucleation rate parameter $\tilde{\beta}$, aka. "beta over H"
    $$\tilde{\beta} \equiv \frac{\beta}{H_*} = (8 \pi)^\frac{1}{3} \frac{\max ({v}_\text{wall}, c_s)}{{r}_*}$$
    :gowling_2021:`\ ` eq. 2.1

    Please see :py:func:`pttools.bubble.nucleation.beta` and
    :py:func:`pttools.bubble.nucleation.r_star` for further information.

    :param r_star: Hubble-scaled mean bubble spacing $r_*$
    :param v_wall: Wall velocity $v_w$
    :param cs: Sound speed $c_s$
    :return: Nucleation rate parameter $\tilde{\beta}$
    """
    b = beta(R_star=r_star, v_wall=v_wall, cs=cs)
    b_min = np.min(b)
    if b_min < beta_over_H_limit:
        logger.warning(
            "Got β/H*=%s < %s for r*=%s, v_wall=%s, cs=%s. "
            "The conversion to from r* to β/H* may not have been accurate, "
            "as this seems to be a very slow phase transition. Please see Caprini et al. (2020) p. 6.",
            b, beta_over_H_limit, r_star, v_wall, cs
        )
    return b


@numba.njit
def nu[T: FloatOrArr](T_tilde: T, nuc_type: NucType = NucType.SIMULTANEOUS, a: float = 1.) -> T:
    r"""
    Bubble lifetime distribution function

    For exponential nucleation:
    $$\nu_\text{exp} (\tilde{T}) = a e^{-a\tilde{T}}$$
    :gw_pt_ssm:`\ ` eq. 4.27

    For simultaneous nucleation:
    $$\nu_\text{sim} (\tilde{T}) = \frac{1}{2} a (a \tilde{T})^2 e^{\left( - \frac{1}{6} \tilde{aT}^3 \right)}$$
    :gw_pt_ssm:`\ ` eq. 4.32

    :param T_tilde: dimensionless time $\tilde{T}$
    :param nuc_type: nucleation type, simultaneous or exponential
    :param a: normalization factor $a$
        (This is an old debugging parameter, and $\nu$ should be normalized regardless of its value.)
    :return: bubble lifetime distribution $\nu$
    """
    if nuc_type == NucType.EXPONENTIAL.value:
        return a * np.exp(-a * T_tilde)
    if nuc_type == NucType.SIMULTANEOUS.value:
        return 0.5 * a * (a * T_tilde)**2 * np.exp(-(a * T_tilde) ** 3 / 6)
    raise ValueError(f"Nucleation type not recognized: \"{nuc_type}\"")


def H(T: th.FloatOrArr, G: th.FloatOrArr = const.G) -> th.FloatOrArr:
    r"""Hubble parameter $H(T)$
    $$H(T) = \sqrt{G} T^2$$
    :notes:`\ ` p. 47

    This is a rough approximation.
    TODO: Compute the gravitational constant in GeV properly
    """
    return np.sqrt(G) * T**2


def r_star(
        beta_over_H: th.FloatOrArr,
        v_wall: th.FloatOrArr,
        cs: th.FloatOrArr = const.CS0,
        beta_over_H_limit: float = const.BETA_OVER_H_LIMIT) -> th.FloatOrArr:
    r"""Hubble-scaled mean bubble spacing $r_*(\beta)
    $$r_* = (8\pi)^\frac{1}{3} \frac{H_*}{\beta} \max ({v}_\text{wall}, c_s)$$
    Derived from :caprini_2020:`\ ` eq. 6

    Please see :py:func:`pttools.bubble.nucleation.R_star` for further information.
    """
    if beta_over_H < beta_over_H_limit:
        logger.warning(
            "Got β/H*=%s < %s (and v_wall=%s, cs=%s). "
            "The conversion from β/H* to r* may not be accurate, as this seems to be a very slow phase transition. "
            "Please see Caprini et al. (2020) p. 6.",
            beta_over_H, beta_over_H_limit, v_wall, cs
        )
    return R_star(beta=beta_over_H, v_wall=v_wall, cs=cs)


def r_star_product(H_star: th.FloatOrArr, R_star: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Hubble-scaled mean bubble spacing $r_*$
    $$r_* = H_* R_*$$
    :gowling_2021:`\ ` eq. 2.2
    """
    return H_star * R_star


def R_star(beta: th.FloatOrArr, v_wall: th.FloatOrArr, cs: th.FloatOrArr = const.CS0) -> th.FloatOrArr:
    r"""Mean bubble separation $R_*$

    $$R_* = \frac{(8\pi)^\frac{1}{3}}{\beta} \max ({v}_\text{wall}, c_s)$$
    :caprini_2020:`\ ` eq. 6
    :hakkinen_msc:`\ ` eq. 2.6

    For detonations (${v}_\text{wall} > c_s$),
    the typical separation between bubbles is set by the wall velocity $v_\text{wall}$.
    For bubbles expanding as deflagrations (${v}_\text{wall} < c_s$),
    the reheating of the plasma by the reaction front can suppress further bubble formation for large enough $\alpha$.
    This is approximated by $\max ({v}_\text{wall}, c_s)$ in the formula above.
    Please note that for very slow PTs ($\frac{\beta}{H_*} \approx 1$),
    this approximation breaks down,
    and the mean bubble separation $R_*$ must be calculated directly from first principles.
    In such cases one should also take into account the expansion of the universe during the phase transition.
    :caprini_2020:`\ ` p. 6
    :gowling_2021:`\ ` p. 5
    :enqvist_1992:`\ ` eq. 4.10

    This formula is derived for simultaneous nucleation,
    but it should be a reasonable approximation for exponential nucleation as well.
    :caprini_2020:`\ ` p. 16

    :param beta: Nucleation rate parameter $\beta$
    :param v_wall: Wall velocity ${v}_w$
    :param cs: Sound speed $c_s$
    :return: Mean bubble separation $R_*$
    """
    return (8 * np.pi)**(1/3) * np.maximum(v_wall, cs) / beta
