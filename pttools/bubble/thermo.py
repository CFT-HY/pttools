import logging

import numpy as np

import pttools.type_hints as th
from .boundary import Phase
from . import props
from . import relativity
from pttools.models.model import Model

logger = logging.getLogger(__name__)


# Todo: Fix the equations in the docstrings

def ebar(model: Model, wn: float):
    """Energy is conserved, and therefore $\bar{e}=e_n$."""
    return model.e(wn, Phase.SYMMETRIC)


def entropy_density(model: "Model", w: np.ndarray, xi: np.ndarray, v_wall: float) -> float:
    r"""
    Volume-averaged entropy density
    $$s_\text{avg} = \frac{1}{v_w^3} \int (s(w,\phi) - s(w_n, \phi_s) \xi^3$$
    """
    phase = props.find_phase(xi, v_wall)
    return 1 / (v_wall**3) * np.trapz(model.s(w, phase) - model.s(w[-1], Phase.SYMMETRIC), xi**3)


def kappa(
        model: "Model",
        v: np.ndarray, w: np.ndarray, xi: np.ndarray,
        v_wall: float,
        delta_e_theta: float = None) -> float:
    if delta_e_theta is None:
        delta_e_theta = trace_anomaly(model, w, xi, v_wall)
    return kinetic_energy_density(v, w, xi, v_wall) / np.abs(delta_e_theta)


def kappa_approx(alpha_n: th.FloatOrArr) -> th.FloatOrArr:
    return alpha_n / (0.73 + 0.083*np.sqrt(alpha_n) + alpha_n)


def kinetic_energy_fraction(
        model: "Model",
        v: np.ndarray, w: np.ndarray, xi: np.ndarray,
        v_wall: float,
        ek: float = None, eb: float = None) -> float:
    r"""Kinetic energy fraction
    $$K = \frac{e_K}{\bar{e}}$$
    """
    if ek is None:
        ek = kinetic_energy_density(v, w, xi, v_wall)
    if eb is None:
        eb = ebar(model, w[-1])
    return ek / eb


# @numba.njit
def kinetic_energy_density(v: np.ndarray, w: np.ndarray, xi: np.ndarray, v_wall: float) -> float:
    r"""
    Volume-averaged kinetic energy density
    $$e_K = 4 \pi \int_0^{xi_\text{max}} d\xi \xi^2 w \gamma^2 v^2$$
    Each point is multiplied by $v$, and therefore having $\xi_{max}$ too far does not affect the results.
    :gw_pt_ssm:`\ ` eq. B.22

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param v_wall: $v_\text{wall}$
    :return: $e_K$
    """
    # The prefactor of 4*pi cancels, and the factor of 3 is not needed due to integrating with respect to xi**3.
    return 1 / (v_wall**3) * np.trapz(w * v**2 * relativity.gamma2(v), xi**3)


# @numba.njit
def mean_adiabatic_index(wb: th.FloatOrArr, eb: th.FloatOrArr) -> th.FloatOrArr:
    r"""Mean adiabatic index
    $$\Gamma = \frac{\bar{w}}{\bar{e}}$$
    """
    return wb / eb


def omega(
        model: "Model",
        w: np.ndarray, xi: np.ndarray,
        v_wall: float,
        delta_e_theta: float = None) -> float:
    if delta_e_theta is None:
        delta_e_theta = trace_anomaly(model, w, xi, v_wall)
    return thermal_energy_density(w, xi, v_wall) / np.abs(delta_e_theta)


# @numba.njit
def thermal_energy_density(w: np.ndarray, xi: np.ndarray, v_wall: float) -> float:
    r"""Thermal energy density
    $$\Delta e_Q = 4 \pi \int_0^{\xi_\text{max}} d\xi \xi^2 \frac{3}{4} (w - w_n)$$
    """
    return 1/(v_wall**3) * np.trapz(0.75*(w - w[-1]), xi**3)


def trace_anomaly(model: "Model", w: np.ndarray, xi: np.ndarray, v_wall: float) -> float:
    r"""Trace anomaly
    $$\Delta e_\theta = 4 \pi \int_0^{\xi_\text{max}} d\xi \xi^2 (\theta - \theta_n)$$
    """
    phase = props.find_phase(xi, v_wall)
    theta = model.theta(w, phase)
    theta_n = model.theta(w[-1], Phase.SYMMETRIC)
    return 1/(v_wall**3) * np.trapz((theta - theta_n), xi**3)


def ubarf2(v: np.ndarray, w: np.ndarray, xi: np.ndarray, v_wall: float, ek: float = None, wb: float = None, wn: float = None) -> float:
    r"""Enthalpy-weighted mean square fluid 4-velocity around the bubble
    $$\bar{U}_f^2 = \frac{3}{4\pi \bar{w} v_w^3} e_K$$
    """
    if ek is None:
        ek = kinetic_energy_density(v, w, xi, v_wall)
    # if wb is None:
    #     wb = wbar(w, xi, v_wall, wn)
    # return 3/(4*np.pi*wb*v_wall**3) * ek
    return ek / w[-1]


def wbar(w: np.ndarray, xi: np.ndarray, v_wall: float, wn: float = None):
    logger.warning("wbar is untested and may return false results")

    w_reverse = w[::-1]
    i_max = len(w_reverse) - np.argmax(w_reverse != w[-1]) - 1
    ret = 1/(v_wall**3) * np.trapz(w[:i_max], xi[:i_max]**3)
    if wn is not None and ret <= wn:
        logger.warning(f"Should have wbar > wn. Got: wbar={wn}, wn={wn}")
    return ret
