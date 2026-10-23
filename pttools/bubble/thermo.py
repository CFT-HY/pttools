r"""
Thermodynamic quantities.

The code for the integrals doesn't have the pre-factor $4\pi$,
since the integrals are computed with respect to $\xi^3$,
which is equivalent.

The volume-averaged and bubble volume averaged quantities are different, and should not be confused with each other.

The integrals are computed using the trapezoidal rule and with respect to $\xi^3$,
since outside the bubble the functions are constant, but with very few given points.
This scheme gives the correct results for these ranges,
as the trapezoidal rule computes the integral of a constant function exactly, even when only the endpoints are given.
If the integrals were with respect to $\xi$, the functions would have the factor $\xi^2$,
which would break this useful property and require a more dense grid of points.

Integration of an angle-independent function over a spherical shell:
$$\int f dV
= \int_{R_{\text{min}}}^{R_{\text{max}}} dr \int_0^{2\pi} d\phi \int_0^\pi d\theta f r^2 \sin \theta
= 4 \pi \int_{R_{\text{min}}}^{R_{\text{max}}} dr r^2 f
= \frac{4\pi}{3} \int_{R_{\text{min}}^3}^{R_{\text{max}}^3} dr^3 f$$
"""

from __future__ import annotations

import logging
import typing as tp

import numba
import numpy as np

from pttools.bubble import props, relativity
from pttools.bubble.check import check_wall_speed
from pttools.bubble.const import DEFAULT_ADIABATIC_INDEX
from pttools.bubble.phase import Phase
from pttools.speedup import njit
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr

if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger: logging.Logger = logging.getLogger(__name__)


def e_bar(model: Model, wn: float) -> float:
    r"""Total average energy density $\bar{e}$.

    This function assumes that the phase transition is rapid enough that the expansion of the universe can be neglected,
    and therefore energy is conserved, and $\bar{e}={e}_n$.
    :gw_pt_ssm:`\ ` p. 39

    :param model: Equation of state model
    :param wn: Nucleation enthalpy density ${w}_n$
    :return: Average energy density $\bar{e}$
    """
    return model.e(wn, Phase.SYMMETRIC)


def entropy_density_diff(
        model: Model,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        phase: th.FloatArr1D | None = None) -> float:
    r"""Bubble volume averaged entropy density.

    $$\frac{3}{4\pi {v}_\text{wall}^3} {s}_\text{avg}$$.

    :param model: Model
    :param w: Enthalpy density $w$
    :param xi: $\xi$
    :param v_wall: Wall velocity ${v}_\text{wall}$
    :param phase: Phase $\phi$. If not given, it's computed from $\xi$ and ${v}_\text{wall}$.
    """
    return 3/(4*np.pi * v_wall**3) * va_entropy_density_diff(model, w, xi, v_wall, phase)


def kappa(
        model: Model,
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        delta_e_theta: float | None = None) -> float:
    r"""$\kappa$, kinetic efficiency factor.

    $$\kappa = \frac{{e}_K}{\lvert\Delta {e}_\theta\rvert}$$.

    :param model: Equation of state model
    :param v: Fluid velocity $v$
    :param w: Enthalpy density $w$
    :param xi: $\xi$
    :param v_wall: Wall velocity ${v}_\text{wall}$
    :param delta_e_theta: Trace anomaly difference $\Delta {e}_\theta$.
        If not given, it's computed from the other arguments.
    :return: Kinetic efficiency factor $\kappa$
    """
    if delta_e_theta is None:
        delta_e_theta = va_trace_anomaly_diff(model, w, xi, v_wall)
    return va_kinetic_energy_density(v, w, xi) / np.abs(delta_e_theta)


def kappa_approx[T: FloatOrArr](alpha_n: T) -> T:
    r"""$\kappa$, approximate kinetic efficiency factor.

    $$\kappa \approx \frac{\alpha_n}{0.73 + 0.083\sqrt{\alpha_n} + \alpha_n}$$
    :notes:`\ ` eq. 7.44.

    :param alpha_n: Transition strength $\alpha_n$
    :return: Approximate kinetic efficiency factor $\kappa$
    """
    return alpha_n / (0.73 + 0.083*np.sqrt(alpha_n) + alpha_n)  # pyrefly: ignore[bad-return]


@njit(nogil=True)
def kinetic_energy_density(v: th.FloatArr1D, w: th.FloatArr1D, xi: th.FloatArr1D, v_wall: float) -> float:
    r"""Bubble volume averaged kinetic energy density.

    $$\frac{3}{4\pi {v}_w^3} {e}_K$$.

    :param v: Fluid velocity $v$
    :param w: Enthalpy density $w$
    :param xi: $\xi$
    :param v_wall: Wall velocity ${v}_\text{wall}$
    """
    return 3/(4*np.pi * v_wall**3) * va_kinetic_energy_density(v, w, xi)


def kinetic_energy_fraction(ek_bva: float, eb: float) -> float:
    r"""Bubble volume averaged kinetic energy fraction $K_\text{bva}$.

    $$K_\text{bva} = \frac{{e}_{K,\text{bva}}}{\bar{e}}$$
    :gw_pt_ssm:`\ ` eq. B.31
    :caprini_2020:`\ ` eq. 22
    This definition is independent of the equation of state.

    Please note that the enumerator of :gw_pt_ssm:`\ ` eq. B.35 assumes that
    $\alpha_n = \frac{4 \epsilon}{3 w_n}$.
    This corresponds to assuming the bag model.
    The denominator in turn assumes that
    $\bar{e} = e_n$, and by definition $e_n = 1 + \alpha_n + \delta_n$.

    :param ek_bva: Bubble volume averaged kinetic energy density ${e}_{K,\text{bva}}$
    :param eb: Total average energy density $\bar{e}$
    :return: Bubble volume averaged kinetic energy fraction $K_\text{bva}$
    """
    return ek_bva / eb


# @njit
def mean_adiabatic_index[T: FloatOrArr](wb: T, eb: T | float) -> T:
    r"""Mean adiabatic index $\Gamma$.

    $$\Gamma = \frac{\bar{w}}{\bar{e}}$$
    :gw_pt_ssm:`\ ` eq. B.33,
    :hindmarsh_2017:`\ ` p. 4.

    Note that this is not exactly equal to the adiabatic index of an ultrarelativistic fluid,
    as total energy $\bar{e}_0$ is divided between kinetic and thermal energy.
    :correia_2025:` \ ` p. 11

    :param wb: Average enthalpy density $\bar{w}$
    :param eb: Average energy density $\bar{e}$
    :return: Mean adiabatic index $\Gamma$
    """
    return wb / eb  # pyrefly: ignore[bad-return]


def mean_enthalpy_change(v: th.FloatArr1D, w: th.FloatArr1D, xi: th.FloatArr1D, v_wall: float) -> float:
    r"""
    Mean change in enthalpy in bubble relative to outside value.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param v_wall: $v_\text{wall}$
    :return: mean enthalpy change
    """
    #    def en_diff(v, dw, xi):
    #        return dw
    #    int1, int2 = split_integrate(en_diff, v, w - w[-1], xi**3, v_wall)
    #    integral = int1 + int2
    check_wall_speed(v_wall)
    integral = np.trapezoid((w - w[-1]), xi ** 3)
    return integral / v_wall ** 3  # pyrefly: ignore[bad-return]


@njit(cache=True)
def nu_gdh2024[T: FloatOrArr](omega: T) -> T:
    r"""$\nu_\text{gdh2024}$.

    $$\nu_\text{gdh2024} = \frac{1 - 3\omega}{1 + 3\omega}$$,
    where $\omega$ is the barotropic equation of state parameter.
    :giombi_2024_cs:`\ ` eq. 2.11, 2.41.
    """
    # typing.cast() is not used below, since Numba cannot compile it.
    return (1 - 3 * omega) / (1 + 3 * omega)  # pyrefly: ignore[bad-return]


def omega(
        model: Model,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        delta_e_theta: float | None = None) -> float:
    r"""Thermal efficiency factor $\omega$.

    $$\omega = \frac{\Delta {e}_Q}{\Delta {e}_\theta}$$
    :gw_pt_ssm:`\ ` eq. B.28.

    :param model: Equation of state model
    :param w: Enthalpy density $w$
    :param xi: $\xi$
    :param v_wall: Wall velocity ${v}_\text{wall}$
    :param delta_e_theta: Trace anomaly difference $\Delta {e}_\theta$.
        If not given, it's computed from the other arguments.
    :return: Thermal efficiency factor $\omega$
    """
    if delta_e_theta is None:
        delta_e_theta = va_trace_anomaly_diff(model, w, xi, v_wall)
    return va_thermal_energy_density_diff(w, xi) / np.abs(delta_e_theta)


def omega_barotropic[T: FloatOrArr](p: T, e: T | float) -> T:
    r"""Barotropic equation of state parameter $\omega$.

    $$\omega(T,\phi) = \frac{p(T,\phi)}{e(T,\phi)}$$
    :giombi_2024_cs:`\ ` p. 3
    In some sources this is known as the equation-of-state parameter for short, and denoted as $w$.
    """
    return p / e  # pyrefly: ignore[bad-return]


@njit(cache=True, nogil=True)
def shell_end_index(w: th.FloatArr1D) -> int:
    r"""Index of the end of the fluid shell, i.e. of the last point at which $w \neq w_n$.

    This is the shock for deflagrations and hybrids, and the wall for detonations.
    If $w = w_n$ everywhere, or if only the first point differs from $w_n$, this is the last index.

    :param w: Enthalpy density $w$, of which the last point is $w_n$
    :return: the index
    """
    # If w = w_n everywhere, argmax returns 0, and therefore the index is the last one.
    # https://stackoverflow.com/a/8768734
    i_max = int(w.size - np.argmax(w[::-1] != w[-1]) - 1)
    # If only the first point differs from w_n, the average is over the whole array.
    return w.size - 1 if i_max == 0 else i_max


def thermal_energy_density(v_shock: float, eqp: float) -> float:
    r"""Average thermal energy density after the phase transition.

    $$\frac{3}{4\pi {v}_\text{sh}^3} {e}_Q'$$.

    ${e}_Q'$ is the total thermal energy within the shock (see :py:func:`va_thermal_energy_density`),
    including that of the fluid between the wall and the shock, which was there already before the phase transition.
    Therefore, it's averaged over the volume within the shock, unlike the changes of the energies,
    such as :py:func:`kinetic_energy_density` and :py:func:`thermal_energy_density_diff`,
    which are averaged over the volume of the bubble.
    For detonations, ${v}_\text{sh} = {v}_\text{wall}$.

    :param v_shock: Shock speed ${v}_\text{sh}$
    :param eqp: Volume-averaged thermal energy density ${e}_Q'$
    :return: Average thermal energy density after the phase transition
    """
    return 3/(4*np.pi * v_shock**3) * eqp


def thermal_energy_density_diff(w: th.FloatArr1D, xi: th.FloatArr1D, v_wall: float) -> float:
    r"""Bubble volume averaged thermal energy density difference.

    $$\frac{3}{4\pi {v}_\text{wall}^3} \Delta e_Q$$.

    :param w: Enthalpy density $w$
    :param xi: $\xi$
    :param v_wall: Wall velocity ${v}_\text{wall}$
    :return: Bubble volume averaged thermal energy density difference
    """
    return 3/(4*np.pi * v_wall**3) * va_thermal_energy_density_diff(w, xi)


def thermal_energy_fraction[T: FloatOrArr](eq_bva: T, eb: T | float) -> T:
    r"""Thermal energy fraction $\frac{{e}_Q'}{\bar{e}}$.

    This is the fraction of the energy within the shock that is thermal energy after the phase transition.

    :param eq_bva: Average thermal energy density ${e}_Q'$ after the phase transition,
        see :py:func:`thermal_energy_density`
    :param eb: Average energy density $\bar{e}$
    :return: Thermal energy fraction $\frac{{e}_Q'}{\bar{e}}$
    """
    return eq_bva / eb  # pyrefly: ignore[bad-return]


def trace_anomaly_diff(
        model: Model,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        phase: th.FloatArr1D | None = None) -> float:
    r"""Bubble volume averaged trace anomaly $\epsilon$.

    $$\epsilon = \frac{3}{4\pi {v}_\text{wall}^3} \Delta {e}_\theta$$.

    :param model: Equation of state model
    :param w: Enthalpy density $w$
    :param xi: $\xi$
    :param v_wall: Wall velocity ${v}_\text{wall}$
    :param phase: Phase $\phi$. If not given, it's computed from $\xi$ and ${v}_\text{wall}$.
    :return: Bubble volume averaged trace anomaly $\epsilon$
    """
    return 3/(4*np.pi * v_wall**3) * va_trace_anomaly_diff(model, w, xi, v_wall, phase)


@njit(nogil=True)
def ubarf2(
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        ek_bva: float | None = None,
        w_bar: float | None = None) -> float:
    r"""$\bar{U}_f^2$, enthalpy-weighted mean square fluid 4-velocity around the bubble.

    $$\bar{U}_f^2 = \frac{3}{4\pi \bar{w} {v}_\text{wall}^3} {e}_K$$
    :gw_pt_ssm:`\ ` eq. B.30
    In some sources such as :giombi_2024_cs:`\ `, this is denoted as $v_\text{rms}^2$.

    This $\bar{U}_f^2$ should not be used with
    :py:func:pttools.ssm.spec_den_v.spec_den_v: or
    :py:func:pttools.ssm.spec_den_gw.spec_den_gw_scaling:,
    as they expect
    :py:func:pttools.ssm.ssm.ubarf2_from_a2: instead.

    :param v: Fluid velocity $v$
    :param w: Enthalpy density $w$
    :param xi: $\xi$
    :param v_wall: Wall velocity ${v}_\text{wall}$
    :param ek_bva: Bubble volume averaged kinetic energy density $e_{K,\text{bva}}$.
        If not given, it's computed from the other arguments.
    :param w_bar: Average enthalpy density $\bar{w}$.
        In various legacy functions that call this function, this is approximated as $\bar{w}=w_n$.
    :return: Enthalpy-weighted mean square fluid 4-velocity around the bubble $\bar{U}_f^2$
    """
    if ek_bva is None:
        ek_bva = kinetic_energy_density(v, w, xi, v_wall)
    if w_bar is None:
        w_bar = __w_bar(w=w, xi=xi, v_wall=v_wall)
    return ek_bva / w_bar


def warn_if_ubarf_relativistic(ubarf2_value: float, source: str = "") -> bool:
    r"""Log a warning if the RMS fluid 4-velocity $\bar{U}_f = \sqrt{\bar{U}_f^2}$ is above 1.

    $\bar{U}_f$ is the enthalpy-weighted RMS of the spatial fluid 4-velocity $\gamma v$
    (:py:func:`ubarf2`), so $\bar{U}_f > 1$ is not forbidden by causality,
    but it means that the fluid flow is relativistic,
    where the linear Sound Shell Model and the GW spectra computed from it are not reliable.
    It can also indicate a failed fluid solution, e.g. for strong transitions near the Chapman-Jouguet speed.

    :param ubarf2_value: $\bar{U}_f^2$, enthalpy-weighted mean square fluid 4-velocity
    :param source: description of the object for the warning message, e.g. the bubble label
    :return: whether $\bar{U}_f > 1$
    """
    if ubarf2_value > 1:
        logger.warning(
            "Got Ubarf=%s > 1%s: the fluid flow is relativistic, "
            "so the linear Sound Shell Model is not reliable. "
            "Please check also the fluid solution.",
            np.sqrt(ubarf2_value), f" for {source}" if source else ""
        )
        return True
    return False


@njit(cache=True)
def ubarf2_from_K[T: FloatOrArr](K: T, adiabatic_index: T | float = DEFAULT_ADIABATIC_INDEX) -> T:
    r"""RMS fluid velocity $\bar{U}_f^2(K)$.

    $$\bar{U}_f^2 \approx \frac{K}{\Gamma}$$
    This comes directly from
    $$K = \Gamma \bar{U}_\text{f}^2$$
    :notes:`\ ` eq. 7.39.

    This is used in :py:func:`pttools.ssm.scaling.H_star_eta_sh_full`.
    """
    return K / adiabatic_index  # pyrefly: ignore[bad-return]


def va_enthalpy_density(eq: float) -> float:
    r"""$w_\text{va}$, volume-averaged enthalpy density after the phase transition.

    $$w_\text{va} = \frac{4}{3} e_Q$$.

    This follows from $e_Q = e - \theta = \frac{3}{4} w$, where $\theta = \frac{1}{4}(e - 3p)$
    is the trace anomaly. As $e_Q$ is averaged over the volume within the shock,
    this is equal to :py:func:`w_bar` up to the numerical accuracy of the integrals.

    :param eq: Average thermal energy density $e_Q$ after the phase transition, see :py:func:`thermal_energy_density`
    """
    return 4/3 * eq


def va_entropy_density_diff(
        model: Model,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        phase: th.FloatArr1D | None = None) -> float:
    r"""
    ${s}_\text{avg}$, volume-averaged entropy density.

    $${s}_\text{avg} = \int d\xi \xi^2 (s(w,\phi) - s({w}_n, \phi_s)$$.
    """
    if phase is None:
        phase = tp.cast(th.FloatArr1D, props.find_phase(xi, v_wall))
    s_diff = model.s(w, phase) - model.s(w[-1], Phase.SYMMETRIC)
    return 4*np.pi/3 * np.trapezoid(s_diff, xi**3)  # pyrefly: ignore[bad-return]


@njit(nogil=True)
def va_kinetic_energy_density(v: th.FloatArr1D, w: th.FloatArr1D, xi: th.FloatArr1D) -> float:
    r"""
    ${e}_K$, volume-averaged kinetic energy density.

    $${e}_K = 4 \pi \int_0^{{\xi}_\text{max}} d\xi \xi^2 w \gamma^2 v^2$$
    Each point is multiplied by $v$, and therefore having ${\xi}_{max}$ too far does not affect the results.
    :gw_pt_ssm:`\ ` eq. B.22.

    :param v: Fluid velocity $v$
    :param w: Enthalpy density $w$
    :param xi: $\xi$
    :return: Volume-averaged kinetic energy density ${e}_K$
    """
    return 4*np.pi/3 * np.trapezoid(w * relativity.gamma2(v) * v**2, xi**3)  # pyrefly: ignore[bad-return]


def va_kinetic_energy_fraction(ek_va: float, eb: float) -> float:
    r"""$K_\text{va}$, volume-averaged kinetic energy fraction.

    $$K_\text{va} = \frac{e_{K,\text{va}}}{\bar{e}}$$.
    """
    return ek_va / eb


def va_thermal_energy_density(v_shock: float, wn: float, ek: float, delta_e_theta: float) -> float:
    r"""${e}_Q'$, volume-averaged thermal energy density after the phase transition.

    $${e}_Q' = {e}_Q + {e}_\theta - {e}_K' - {e}_\theta'
    = 4\pi \int_0^{{\xi}_\text{max}} d\xi \xi^2 \frac{3}{4} {w}_n - {e}_K' - \Delta {e}_\theta$$.
    """
    return np.pi * wn * v_shock**3 - ek - delta_e_theta


@njit(nogil=True)
def va_thermal_energy_density_diff(w: th.FloatArr1D, xi: th.FloatArr1D) -> float:
    r"""$\Delta {e}_Q$, volume-averaged thermal energy density difference.

    $$\Delta {e}_Q = 4 \pi \int_0^{\xi_\text{max}} d\xi \xi^2 \frac{3}{4} (w - {w}_n)$$
    :gw_pt_ssm:`\ ` eq. B.25.

    :param w: Enthalpy density $w$
    :param xi: $\xi$
    :return: Volume-averaged thermal energy density difference $\Delta e_Q$
    """
    return 4*np.pi/3 * np.trapezoid(0.75*(w - w[-1]), xi**3)  # pyrefly: ignore[bad-return]


def va_thermal_energy_fraction(eq_va: float, eb: float) -> float:
    r"""$Q_\text{va}$, volume-averaged thermal energy fraction.

    $$Q_\text{va} = \frac{e_{Q,\text{va}}}{\bar{e}}$$.

    :param eq_va: Volume-averaged thermal energy density $e_{Q,\text{va}}$
    :param eb: Average energy density $\bar{e}$
    :return: Volume-averaged thermal energy fraction $Q_\text{va}$
    """
    return eq_va / eb


def va_trace_anomaly_diff(
        model: Model,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        phase: th.FloatArr1D | None = None) -> float:
    r"""$\Delta {e}_\theta$, volume-averaged trace anomaly difference.

    $$\Delta {e}_\theta = 4 \pi \int_0^{{\xi}_\text{max}} d\xi \xi^2 (\theta - {\theta}_n)$$
    :gw_pt_ssm:`\ ` eq. B.25.

    :param model: Equation of state model
    :param w: Enthalpy density $w$
    :param xi: $\xi$
    :param v_wall: Wall velocity ${v}_\text{wall}$
    :param phase: Phase $\phi$. If not given, it's computed from $\xi$ and ${v}_\text{wall}$.
    :return: Volume-averaged trace anomaly difference $\Delta {e}_\theta$
    """
    if phase is None:
        phase = tp.cast(th.FloatArr1D, props.find_phase(xi, v_wall))
    theta = model.theta(w, phase)
    theta_n = model.theta(w[-1], Phase.SYMMETRIC)
    return 4*np.pi/3 * np.trapezoid((theta - theta_n), xi**3)  # pyrefly: ignore[bad-return]


@njit(cache=True, nogil=True)
def w_bar(w: th.FloatArr1D, xi: th.FloatArr1D, v_wall: float) -> float:
    r"""$\bar{w}$, average enthalpy density.

    This is the volume average of the enthalpy density over the sphere that contains the fluid shell:
    $$\bar{w} = \frac{3}{\xi_\text{max}^3} \int_0^{\xi_\text{max}} d\xi \xi^2 w(\xi)
    = \frac{1}{\xi_\text{max}^3} \int_0^{\xi_\text{max}^3} d(\xi^3) w(\xi),$$
    where $\xi_\text{max}$ is the largest $\xi$ at which $w \neq w_n$,
    i.e. the shock $v_\text{sh}$ for deflagrations and hybrids and the wall $v_\text{wall}$ for detonations.
    The integral is computed over $\xi^3$ with the trapezoidal rule,
    which is exact for the constant regions of the profile, such as the interior of the bubble.
    If the profile has no point with $w \neq w_n$, $\bar{w} = w_n$.

    The energy of the self-similar flow is conserved, and therefore the average of the energy density
    $T^{00} = e + w \gamma^2 v^2$ over the same sphere is $\bar{e} = e_n$, see :py:func:`e_bar`.
    For the bag model, $e = \frac{3}{4} w + \theta$, and therefore
    $$\bar{w} = \frac{4}{3} \left( e_n - \langle \theta \rangle - \langle w \gamma^2 v^2 \rangle \right),$$
    where the angle brackets denote the averages over the sphere.
    For deflagrations and hybrids, the sphere includes the fluid in the symmetric phase between the wall and the shock.

    :param w: Enthalpy density $w$
    :param xi: $\xi$, in increasing order
    :param v_wall: Wall velocity ${v}_\text{wall}$. This is not used, as $\xi_\text{max}$ is found from $w$.
    :return: Average enthalpy density $\bar{w}$
    """
    i_max = shell_end_index(w)
    ret = 1/(xi[i_max]**3) * np.trapezoid(w[:i_max+1], xi[:i_max+1]**3)
    wn = w[-1]
    if not (ret is None or np.isnan(ret)) and ret <= wn:
        with numba.objmode:
            logger.warning("Should have w_bar > wn. Got: w_bar=%s, wn=%s", ret, wn)
    return ret


def w_bar_energy(
        model: Model,
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        phase: th.FloatArr1D | None = None) -> float:
    r"""$\bar{w}$, average enthalpy density, computed from the energy budget of the bubble.

    This is the same quantity as :py:func:`w_bar`, but computed with the chain
    :py:func:`va_kinetic_energy_density`, :py:func:`va_trace_anomaly_diff`, :py:func:`va_thermal_energy_density`,
    :py:func:`thermal_energy_density` and :py:func:`va_enthalpy_density`:
    $$\bar{w} = \frac{4}{3} \frac{{e}_Q'}{{V}_\text{max}}
    = \frac{4}{3} \frac{3}{4 \pi \xi_\text{max}^3}
    \left( \pi w_n \xi_\text{max}^3 - {e}_K - \Delta {e}_\theta \right)
    = w_n - \frac{4}{3} \left( \langle w \gamma^2 v^2 \rangle + \langle \theta - \theta_n \rangle \right),$$
    where ${V}_\text{max} = \frac{4 \pi}{3} \xi_\text{max}^3$ is the volume within the shock,
    the angle brackets denote averages over it,
    and $\xi_\text{max}$ is found as in :py:func:`w_bar`.
    This follows from the conservation of the energy $T^{00} = e + w \gamma^2 v^2$ within the shock
    and from $e - \theta = \frac{3}{4} w$, where $\theta = \frac{1}{4}(e - 3p)$ is the trace anomaly.
    Therefore, it's equal to :py:func:`w_bar` to the extent that the numerical solution conserves energy.

    :param model: Equation of state model
    :param v: Fluid velocity $v$
    :param w: Enthalpy density $w$
    :param xi: $\xi$, in increasing order
    :param v_wall: Wall velocity ${v}_\text{wall}$
    :param phase: Phase $\phi$. If not given, it's computed from $\xi$ and ${v}_\text{wall}$.
    :return: Average enthalpy density $\bar{w}$
    """
    xi_max = xi[shell_end_index(w)]
    eqp = va_thermal_energy_density(
        v_shock=xi_max,
        wn=w[-1],
        ek=va_kinetic_energy_density(v, w, xi),
        delta_e_theta=va_trace_anomaly_diff(model, w, xi, v_wall, phase))
    return va_enthalpy_density(thermal_energy_density(v_shock=xi_max, eqp=eqp))


__w_bar = w_bar
