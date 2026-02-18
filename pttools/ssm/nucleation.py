"""Bubble nucleation"""

import enum

import numba
import numpy as np

from pttools.ssm import const
import pttools.type_hints as th


@enum.unique
class NucType(str, enum.Enum):
    """Nucleation type"""
    EXPONENTIAL = "exponential"
    SIMULTANEOUS = "simultaneous"


#: Default nucleation type
DEFAULT_NUC_TYPE = NucType.EXPONENTIAL


def beta(R_star: th.FloatOrArr, v_wall: th.FloatOrArr, cs: th.FloatOrArr = const.CS0) -> th.FloatOrArr:
    r"""Convert R_* to \beta for a given wall velocity

    $$\beta = \frac{8\pi}{3} \frac{\max (v_w, c_s)}{R_*}$$
    Inverted from :caprini_2020:`\ ` eq. 29

    :param R_star: Mean bubble separation $R_*$
    :param v_wall: Wall velocity $v_w$
    :param cs: Sound speed $c_s$
    :return: Inverse phase transition duration $\beta$
    """
    return (8 * np.pi)**(1/3) * np.maximum(v_wall, cs) / R_star


@numba.njit
def nu(T: th.FloatOrArr, nuc_type: NucType = NucType.SIMULTANEOUS, a: float = 1.) -> th.FloatOrArr:
    r"""
    Bubble lifetime distribution function

    :gw_pt_ssm:`\ ` eq. 4.27 & 4.32

    :param T: dimensionless time
    :param nuc_type: nucleation type, simultaneous or exponential
    :return: bubble lifetime distribution $\nu$
    """
    if nuc_type == NucType.SIMULTANEOUS.value:
        return 0.5 * a * (a*T)**2 * np.exp(-(a*T)**3 / 6)
    if nuc_type == NucType.EXPONENTIAL.value:
        return a * np.exp(-a*T)
    # raise ValueError(f"Nucleation type not recognized: \"{nuc_type}\"")
    raise ValueError("Nucleation type not recognized")


def H(T: th.FloatOrArr, G: float = const.G) -> th.FloatOrArr:
    r"""Hubble parameter $H(T)$
    $$H(T) = \sqrt{G} T^2$$
    :notes:`\ ` p. 47

    This is a rough approximation.
    TODO: Compute the gravitational constant in GeV properly
    """
    return np.sqrt(G) * T**2


def r_star(H_n: th.FloatOrArr, R_star: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Hubble-scaled mean bubble spacing $r_*$
    $$r_* = H_n R_*$$
    :gowling_2021:`\ ` eq. 2.2
    """
    return H_n * R_star


def r_star_from_beta(beta: th.FloatOrArr, v_wall: th.FloatOrArr, T_n: float) -> th.FloatOrArr:
    r"""$r_*(\beta)"""
    return r_star(H_n=H(T_n), R_star=R_star(beta, v_wall))


def r_star_from_beta_tilde(beta_tilde: th.FloatOrArr, v_wall: th.FloatOrArr) -> th.FloatOrArr:
    return R_star(beta=beta_tilde, v_wall=v_wall)


def R_star(beta: th.FloatOrArr, v_wall: th.FloatOrArr, cs: th.FloatOrArr = const.CS0) -> th.FloatOrArr:
    r"""Mean bubble separation $R_*$

    $$R_* = \frac{8\pi}{3} \frac{\max (v_w, c_s)}{\beta}$$
    :caprini_2020:`\ ` eq. 6
    :hakkinen_msc:`\ ` eq. 2.6
    The $c_s$ corrects the fact that the reheating of the plasma by the reaction front can suppress
    further bubble formation for large enough $\alpha$.

    :param beta: Inverse phase transition duration $\beta$
    :param v_wall: Wall velocity $v_w$
    :param cs: Sound speed $c_s$
    :return: Mean bubble separation $R_*$
    """
    return (8 * np.pi)**(1/3) * np.maximum(v_wall, cs) / beta
