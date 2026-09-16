r"""Energy budget approximations.

These approximations are based on :espinosa_2010:`\ `.
"""

import typing as tp

import numpy as np
import scipy.optimize

from pttools.bubble import Phase
from pttools.bubble.chapman_jouguet import v_chapman_jouguet_bag
from pttools.bubble.const import CS0, DEFAULT_ADIABATIC_INDEX
from pttools.models import Model
from pttools.speedup import njit
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr

# The functions in this file don't call code from other files and are therefore safe to cache.


@np.vectorize
def alpha_n_from_ubarf(
        v_wall: th.FloatOrArr,
        ubarf: th.FloatOrArr,
        cs: th.FloatOrArr = CS0,
        adiabatic_index: th.FloatOrArr = DEFAULT_ADIABATIC_INDEX,
        alpha_n_min: float = 1e-8,
        alpha_n_max: float = 1e12,
        xtol: float = 1e-6) -> th.FloatOrArr:
    r"""Phase transition strength $\alpha(\bar{U}_f)$.

    The calculation of $\bar{U}_f$ is not easy to invert,
    so we calculate $\bar{U}_f$ for different $\alpha$
    until we find an $\alpha$ that minimizes the difference
    between the calculated $\bar{U}_f$ and the input $\bar{U}_f$ value.

    :param v_wall: Wall velocity $v_\text{wall}$
    :param ubarf: List of rms fluid velocities $\bar{U}_f$
    :param adiabatic_index: Adiabatic index $\Gamma$
    :return: Array of phase transition strengths $\alpha$
    """
    # try:
    return scipy.optimize.brentq(
        alpha_n_from_ubarf_solvable,
        args=(ubarf, v_wall, cs, adiabatic_index),
        a=alpha_n_min, b=alpha_n_max, xtol=xtol
    )
    # except ValueError as err:
    #     print(
    #         "Ubarf at a:", _ubarf(v_wall=v_wall, alpha_n=a, cs=cs, adiabatic_index=adiabatic_index),
    #         "Ubarf at b:", _ubarf(v_wall=v_wall, alpha_n=b, cs=cs, adiabatic_index=adiabatic_index),
    #         "Target ubarf:", ubarf
    #     )
    #     raise err


@njit
def alpha_n_from_ubarf_solvable(
        alpha_n: float,
        ubarf_target: float,
        v_wall: float,
        cs: float,
        adiabatic_index: float) -> float:
    return ubarf_approx(v_wall=v_wall, alpha_n=alpha_n, cs=cs, adiabatic_index=adiabatic_index) - ubarf_target


@njit(cache=True)
def delta_kappa_approx[T: FloatOrArr](alpha_n: T) -> T:
    r"""Approximation for $\delta \kappa$.

    $$\delta \kappa \approx -0.9 \log \frac{\sqrt{\alpha_n}}{1 + \sqrt{\alpha_n}}$$
    :espinosa_2010:`\ `, eq. 101
    """
    return -0.9 * np.log(np.sqrt(alpha_n) / (1 + np.sqrt(alpha_n)))


def delta_n[T: FloatOrArr](model: "Model", wn: T) -> T:
    r"""$\delta_n$ for $K$.

    $$\delta_n = \frac{4 \theta_-}{3 w_n}$$
    For the bag model with $V_- = 0$, $\delta_n = 0$.
    :notes:`\ `, eq. 7.43

    :param model: equation of state
    :param wn: $w_n$, enthalpy at nucleation temperature in the symmetric phase
    :return: $\delta_n$
    """
    return tp.cast(T, 4 * model.theta_temp(model.temp(wn, Phase.SYMMETRIC), Phase.BROKEN) / (3 * wn))


@njit(cache=True)
def kappa_a(v_wall: th.FloatOrArr, alpha_n: th.FloatOrArr) -> th.FloatOrArr:
    r"""Approximation for $\kappa_a$.

    $$\kappa_A \approx v_{\text{wall}}^\frac{6}{5} \frac{6.9 \alpha_n}{1.36 - 0.037 \sqrt{\alpha_n} + \alpha_n}$$
    :espinosa_2010:`\ `, eq. 95
    For small wall speeds xi_w << cs
    """
    return v_wall**(6/5) * 6.9 * alpha_n / (1.36 - 0.037 * np.sqrt(alpha_n) + alpha_n)


@njit(cache=True)
def kappa_b[T: FloatOrArr](alpha_n: T) -> T:
    r"""Approximation for $\kappa_b$.

    $$\kappa_B \approx \frac{\alpha_n^\frac{2}{5}}{0.017 + (0.997 + \alpha_n)^\frac{2}{5}}$$
    :espinosa_2010:`\ `, eq. 96
    For the transition from subsonic to supersonic deflagrations, xi_w = cs
    """
    return alpha_n**(2/5) / (0.017 + (0.997 + alpha_n)**(2/5))  # type: ignore[return-value]


@njit(cache=True)
def kappa_c[T: FloatOrArr](alpha_n: T) -> T:
    r"""Approximation for $\kappa_c$.

    $$\kappa_C \approx \frac{\sqrt{\alpha_n}}{0.135 + \sqrt{0.98 + \alpha_n}}$$
    :espinosa_2010:`\ `, eq. 97
    For Jouguet detonations xi_w = xi_j
    """
    return np.sqrt(alpha_n) / (0.135 + np.sqrt(0.98 + alpha_n))  # type: ignore[return-value]


@njit(cache=True)
def kappa_d[T: FloatOrArr](alpha_n: T) -> T:
    r"""Approximation for $\kappa_d$.

    $$\kappa_D \approx \frac{\alpha_n}{0.73 + 0.083 \sqrt{\alpha_n} + \alpha_n}$$
    :espinosa_2010:`\ `, eq. 98
    $\xi_w$ => 1 v. large wall speed
    """
    return alpha_n / (0.73 + 0.083 * np.sqrt(alpha_n) + alpha_n)  # type: ignore[return-value]


@njit
def kappa_detonation_approx(v_wall: th.FloatOrArr, alpha_n: th.FloatOrArr, v_cj: float | None = None) -> th.FloatOrArr:
    r"""Approximation of $\kappa$ for detonations
    $$
    \kappa(v_{\text{wall}} > v_{CJ}) \approx \frac{
    (v_{CJ} - 1)^3 * v_{CJ}^{5/2} * v_{\text{wall}}^{-5/2} * \kappa_C * \kappa_D
    }{
    ((v_{CJ} - 1)^3 - (v_{\text{wall}} - 1)^3)) * v_{CJ}^{5/2} + \kappa_C + (v_{\text{wall}} - 1)^3 * \kappa_D
    }
    $$
    :espinosa_2010:`\ `, eq. 100.
    """
    if v_cj is None:
        v_cj = v_chapman_jouguet_bag(alpha_plus=alpha_n)
    kc = kappa_c(alpha_n)
    kd = kappa_d(alpha_n)
    return (
        ((v_cj - 1)**3 * v_cj**(5/2) * v_wall**(-5/2) * kc * kd) /
        (((v_cj - 1)**3 - (v_wall - 1)**3) * v_cj**(5/2) * kc + (v_wall - 1)**3 * kd)
    )


@njit
def kappa_hybrid_approx(v_wall: th.FloatOrArr, alpha_n: th.FloatOrArr, cs: th.FloatOrArr = CS0) -> th.FloatOrArr:
    r"""Approximation of $\kappa$ for hybrids, aka. supersonic deflagrations.

    $$
    \kappa(c_s < v_{\text{wall}} < v_{CJ}) \approx \kappa_B
    + (v_{\text{wall}} - c_s) \delta \kappa
    + \frac{(v_{\text{wall}} - c_s)^3}{(v_{CJ} - c_s)^3}
    \left(\kappa_C - \kappa_B - (v_{CJ} - c_s) \delta \kappa \right)
    $$
    :espinosa_2010:`\ `, eq. 102
    """
    kb = kappa_b(alpha_n)
    kc = kappa_c(alpha_n)
    dk = delta_kappa_approx(alpha_n)
    v_cj = v_chapman_jouguet_bag(alpha_plus=alpha_n)
    return kb + (v_wall - cs) * dk + ((v_wall - cs)**3 / (v_cj - cs)**3) * (kc - kb - (v_cj - cs) * dk)


@njit(cache=True)
def kappa_sub_def_approx(v_wall: th.FloatOrArr, alpha_n: th.FloatOrArr, cs: th.FloatOrArr = CS0) -> th.FloatOrArr:
    r"""Approximation of $\kappa$ for subsonic deflagrations.

    $$\kappa(v_{\text{wall}} < c_s) \approx \frac{
    c_s^\frac{11}{5} \kappa_A \kappa_B
    }{
    (c_s^\frac{11}{5} - v_{\text{wall}}^\frac{11}{5}) \kappa_B + v_{\text{wall}} c_s^\frac{6}{5} \kappa_A
    }$$
    :espinosa_2010:`\ `, eq. 99
    """
    ka = kappa_a(v_wall, alpha_n)
    kb = kappa_b(alpha_n)
    return cs**(11/5) * ka * kb / ((cs**(11/5) - v_wall**(11/5)) * kb + v_wall * cs**(6/5) * ka)


@njit
def kappa_v_approx(
        v_wall: float,
        alpha_n: th.FloatOrArr,
        cs: float = CS0,
        v_cj: float | None = None) -> th.FloatOrArr:
    r"""Fluid efficiency $\kappa_v$.

    The fluid efficiency gives the fraction of vacuum energy that is
    turned into kinetic energy during the phase transition.

    :param v_wall: Wall velocity $v_\text{wall}$
    :param alpha_n: Phase transition strength $\alpha_n$
    :param cs: Sound speed $c_s$
    :param v_cj: Chapman-Jouguet speed $v_{CJ}$. If not provided, it will be calculated from $\alpha_n$.
    :return: Fluid efficiency $\kappa_v$
    """
    if v_cj is None:
        v_cj = v_chapman_jouguet_bag(alpha_plus=alpha_n)

    if v_wall == cs:
        # This is from the original PTtools code.
        return kappa_b(alpha_n)
    if v_wall < cs:
        return kappa_sub_def_approx(v_wall, alpha_n, cs)
    if v_wall == v_cj:
        return kappa_c(alpha_n)
    # Todo: This approximation was present in the original PTtools code. Why?
    # if v_wall > 0.85:
    #     return kappa_d(alpha_n)
    if v_wall > v_cj:
        return kappa_detonation_approx(v_wall, alpha_n, v_cj)
    return kappa_hybrid_approx(v_wall, alpha_n, cs)


@njit
def kinetic_energy_fraction_approx[T: FloatOrArr](
        v_wall: float,
        alpha_n: T,
        model: Model | None = None,
        cs: float = CS0,
        v_cj: float | None = None) -> T:
    r"""Approximation for the kinetic energy fraction $K$.

    $$K \approx \kappa \frac{\alpha_n}{1 + \alpha_n + \delta_n}$$
    :notes:`\ ` eq. 7.43.
    A version without $\delta_n$ is used in
    :hakkinen_msc:`\ ` eq. 2.40.

    Some sources have a pre-factor of 0.6, such as
    $$K \approx 0.6 \kappa \frac{\alpha_n}{1 + \alpha_n}$$
    :caprini_2024:`\ ` p. 9.
    """
    dn = 0. if model is None else delta_n(model, wn=model.wn(alpha_n))
    return kappa_v_approx(v_wall=v_wall, alpha_n=alpha_n, cs=cs, v_cj=v_cj) * alpha_n / (1 + alpha_n + dn)


@njit
def ubarf_approx(
        v_wall: float,
        alpha_n: th.FloatOrArr,
        delta_n: th.FloatOrArr = 0.,
        cs: float = CS0,
        adiabatic_index: th.FloatOrArr = DEFAULT_ADIABATIC_INDEX) -> th.FloatOrArr:
    r"""Approximate RMS fluid velocity $\bar{U}_f(v_\text{wall}, \alpha_n)$.

    $$
    \bar{U}_f = \sqrt{\frac{K}{\Gamma}}
    = \sqrt{\frac{\kappa \alpha_n}{\Gamma (1 + \alpha_n + \delta_n)}}
    \approx \sqrt{\frac{\kappa \alpha_n}{\Gamma (1 + \alpha_n)}}
    $$
    :notes:`\ `, eq. 7.39, 7.43,
    :caprini_2020:`\ `, eq. 10

    :param v_wall: Wall velocity $v_\text{wall}$
    :param alpha_n: Phase transition strength $\alpha_n$
    :param delta_n: $\delta_n$
    :param cs: Sound speed $c_s$
    :param adiabatic_index: Adiabatic index $\Gamma$
    :return: Measure of the RMS fluid velocity $\bar{U}_f$
    """
    return np.sqrt(
        kappa_v_approx(v_wall=v_wall, alpha_n=alpha_n, cs=cs) * alpha_n /
        (adiabatic_index * (1. + alpha_n + delta_n))
    )


def ubarf_approx_K[T: FloatOrArr](K: T) -> T:
    r"""Approximate RMS fluid velocity $\bar{U}_f(K)$.

    $$\bar{U}_f \approx \approx \sqrt{K}$$
    :ajmi:`\ ` p. 9.
    This is used in :py:func:`pttools.omgw0.scaling.H_star_tau_sh_approx`.
    """
    return np.sqrt(K)
