r"""Shared reference functions for the low-k tests.

The reference formulas are written here independently of :py:mod:`pttools.ssm.low_k`,
directly from the equations of :giombi_2024_cs:`\ ` and :giombi_2026:`\ `.
"""

import typing as tp

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, jv, jvp, yv, yvp

from pttools.ssm.low_k import Pv_analytical
import pttools.type_hints as th

#: $k_p R_* = 2 \pi$, peak of the analytical $\tilde{P}_v$ (:giombi_2026:`\ ` below eq. 3.1)
KP: float = 2 * np.pi
#: Momenta $x = p R_*$ that cover the support of the analytical $\tilde{P}_v$
X: th.FloatArr1D = np.logspace(-3, 4, 5000)
#: Analytical $\tilde{P}_v$ for $\bar{U}_f = 1$
P_TILDE_V: th.FloatArr1D = Pv_analytical(X, kp=KP, ubarf2=1.)
#: $\nu$ values of a softened equation of state ($c_s^2 < 1/3$)
NUS: tuple[float, ...] = (0.1, 0.25, 0.4)

#: Cylindrical Bessel function or its derivative $f(\nu, x)$
BesselFun = tp.Callable[..., tp.Any]


def cs_from_nu(nu: float) -> float:
    r"""$c_s$ corresponding to $\nu = \frac{1 - 3\omega}{1 + 3\omega}$ with $c_s^2 = \omega$."""
    return np.sqrt((1 - nu) / (3 * (1 + nu)))


def nu_from_cs(cs: float) -> float:
    r"""$\nu = \frac{1 - 3 c_s^2}{1 + 3 c_s^2}$, :giombi_2024_cs:`\ ` eq. 2.11."""
    return (1 - 3 * cs**2) / (1 + 3 * cs**2)


def _sph(fun: BesselFun, order: float, x: float) -> float:
    r"""Spherical Bessel function of a non-integer order from the cylindrical one."""
    return np.sqrt(np.pi / (2 * x)) * fun(order + 0.5, x)


def _riccati_derivative(fun: BesselFun, funp: BesselFun, order: float, x: float) -> float:
    r"""$\frac{d}{dx} \left[ x f_{\nu}(x) \right]$ for a spherical Bessel function $f_\nu$."""
    mu = order + 0.5
    return np.sqrt(np.pi / 2) * (fun(mu, x) / (2 * np.sqrt(x)) + np.sqrt(x) * funp(mu, x))


def green_derivative(eta: float, eta1: float, k: float, nu: float) -> float:
    r"""$G_k'(\eta, \eta_1) = \partial_\eta G_k(\eta, \eta_1)$ for $\eta > \eta_1$.

    $$G_k(\eta, \eta_1) = -k \eta \eta_1
    \left[ j_{\nu}(k\eta) y_{\nu}(k\eta_1) - j_{\nu}(k\eta_1) y_{\nu}(k\eta) \right]$$
    :giombi_2026:`\ ` eq. 2.17.
    """
    return -eta1 * k * (
        _riccati_derivative(jv, jvp, nu, k * eta) * _sph(yv, nu, k * eta1)
        - _sph(jv, nu, k * eta1) * _riccati_derivative(yv, yvp, nu, k * eta)
    )


def kernel_low_numerical(
        z: float,
        nu: float,
        tau_star: float,
        tau_end: float,
        friction: float | None = None,
        n_phase: int = 16) -> float:
    r"""Non-oscillating part of the low-frequency kernel computed from the Green's function.

    $$\Delta = \iint_{\tau_*}^{\tau_\text{end}} \frac{d\tau_1 d\tau_2}{\tau_*^2}
    \left( \frac{\tau_*^2}{\tau_1 \tau_2} \right)^{1+\nu}
    \left\langle G_z'(\tau, \tau_1) G_z'(\tau, \tau_2) \right\rangle
    \cos(c_s x \tau_-) \cos(c_s y \tau_-)$$
    :giombi_2026:`\ ` eq. 2.38, first term.
    In the limit $z \rightarrow 0$ we have $y \rightarrow x$,
    and the non-oscillating part of $\cos^2(c_s x \tau_-)$ is $\frac{1}{2}$.
    The integrand then factorizes, and the double integral is the square of a single integral.
    The average $\langle \cdot \rangle$ is over one period of the gravitational wave
    at a late time $\tau \gg \tau_\text{end}$, as in :giombi_2024_cs:`\ ` eq. 2.39.
    The $n$ evenly spaced samples of the phase average the oscillations exactly.

    The exponent $1 + \nu$ of the weight consists of $1 - \nu$ from the background (:giombi_2024_cs:`\ ` eq. 2.16)
    and $2\nu$ from the acoustic friction $\tilde{v} \propto (\eta_* / \eta)^\nu$ (eq. 2.28).
    If the friction exponent $d$ is given, the weight is $(\tau_*^2 / \tau_1 \tau_2)^{1 - \nu + 2d}$ instead,
    whereas the Green's function still has the order $\nu$ of the background.

    :param z: $z = kR_*$
    :param nu: $\nu$ of the background
    :param tau_star: $\tau_*$
    :param tau_end: $\tau_\text{end}$
    :param friction: $d$, the acoustic friction exponent. If not given, $d = \nu$.
    :param n_phase: number of samples of the phase of the GW
    :return: the kernel
    """
    exponent = 1 + nu if friction is None else 1 - nu + 2 * friction
    tau0 = 1e4 / z
    period = 2 * np.pi / z
    squares = []
    for tau in tau0 + period * np.arange(n_phase) / n_phase:
        integral = quad(
            lambda t, tau=tau: (tau_star / t)**exponent / tau_star * green_derivative(tau, t, z, nu),
            tau_star, tau_end, epsabs=0, epsrel=1e-11, limit=200
        )[0]
        squares.append(integral**2)
    return 0.5 * float(np.mean(squares))


def p_tilde_gw_low(z: th.FloatArr1D, nu: float, r_star: float, eta_ratio: float, iv: float) -> th.FloatArr1D:
    r"""$\tilde{P}_\text{gw}^\text{low}$ from :giombi_2026:`\ ` eq. 3.5a.

    $$\mathcal{P}_\text{gw,sw}^\text{lo,low} =
    3 (1 + \nu) \left( \Gamma \bar{U}_f^2 \right)^2 r_*^{2\nu} \left( \frac{a_*}{a_r} \right)^\frac{2\nu}{1+\nu}
    \frac{z^{3 - 2\nu}}{2\pi^2} \left( \frac{1 + \nu}{2} \right)^{-2\nu}
    \frac{\Gamma^2 \left( \frac{1}{2} + \nu \right)}{2\pi} (1 + \nu) \frac{16}{15} \mathcal{I}_v
    \Upsilon^2 \left( \frac{\eta_*}{\eta_\text{end}}, 2\nu \right)$$
    divided by the prefactor of eq. 2.35,
    $$3 (1 + \nu) \left( \Gamma \bar{U}_f^2 \right)^2 r_* \left( \frac{a_*}{a_r} \right)^\frac{2\nu}{1+\nu}
    \frac{z^3}{2\pi^2}.$$
    """
    upsilon = (1 - (1 + eta_ratio)**(-2 * nu)) / (2 * nu)
    return r_star**(2 * nu - 1) * z**(-2 * nu) * ((1 + nu) / 2)**(-2 * nu) \
        * gamma(0.5 + nu)**2 / (2 * np.pi) * (1 + nu) * 16 / 15 * iv * upsilon**2
