"""Kernel functions."""

import math

import numpy as np

from pttools.speedup import njit
from pttools.ssm.barotropic import Upsilon
from pttools.type_hints import FloatOrArr


@njit(cache=True)
def kernel_int_bracket[T: FloatOrArr](cs: T) -> T:
    r"""The term in brackets in $\tilde{P}_\text{gw}^\text{int}$.

    $$3 - 2 c_s^2 - \frac{3}{c_s}(1 - c_s^2) \text{arctanh}(c_s)$$
    :giombi_2024_cs:`\ ` eq. 3.11.

    :param cs: $c_s$, speed of sound
    :return: the term in brackets
    """
    # typing.cast() is not used below, since Numba cannot compile it.
    return 3 - 2 * cs ** 2 - 3 / cs * (1 - cs ** 2) * np.arctanh(cs)  # pyrefly: ignore[bad-return]


@njit(cache=True)
def acoustic_friction_exponent(nu: float, cs: float) -> float:
    r"""$d$, the exponent of the decay $\tilde{v} \propto \eta^{-d}$ of the sound waves due to acoustic friction.

    The momentum conservation of the fluid in an expanding background gives the velocity equation
    $$\tilde{v}' + (1 - 3c_s^2) \mathcal{H} \tilde{v} - c_s^2 p \tilde{\lambda} = 0$$
    (:giombi_2026:`\ ` eq. B.2b (arXiv v2) at leading order, there with $c_s^2 = \omega$),
    where $c_s^2 = \bar{p}' / \bar{e}'$ and $\mathcal{H} = \frac{1 + \nu}{\eta}$.
    The friction term $\frac{D}{\eta} \tilde{v}$ with $D = (1 + \nu)(1 - 3c_s^2)$
    damps the amplitude of the sound waves as $\eta^{-D/2}$, and therefore
    $$d = \frac{(1 + \nu)(1 - 3c_s^2)}{2}.$$
    For $c_s^2 = \omega$, i.e. $\nu = \frac{1 - 3c_s^2}{1 + 3c_s^2}$, this is $d = \nu$,
    which is the acoustic friction factor $(\eta_* / \eta)^\nu$ of :giombi_2024_cs:`\ ` eq. 2.28 and sec. 4.2.

    Caveat: here $c_s^2 = \bar{p}' / \bar{e}'$ is the adiabatic speed of sound along the background evolution.
    :py:class:`pttools.ssm.SSMSpectrum` gives the speed of sound of the model in the broken phase
    at the volume-averaged enthalpy instead, which is the same only for a single-phase barotropic fluid.
    Moreover, the articles assume a constant $\omega = \bar{p} / \bar{e}$, i.e. $c_s^2 = \omega$,
    and $c_s^2 \neq \omega$ is possible only when e.g. a vacuum energy contributes to $\bar{p}$ and $\bar{e}$,
    in which case $\omega$ is not constant in time.
    Therefore, the $d$ given by this function for $c_s^2 \neq \omega$ should be considered an estimate.

    :param nu: $\nu_\text{gdh2024}$ of the background expansion
    :param cs: $c_s$, speed of sound
    :return: $d$
    """
    return (1 + nu) * (1 - 3 * cs**2) / 2


@njit
def kernel_low[T: FloatOrArr](
        z: T,
        nu: float,
        tau_star: T | float,
        tau_end: T | float,
        cs: float | None = None) -> T:
    r"""$\Delta_\text{low}$, low-frequency kernel.

    $$\Delta_\text{low} =
    \left( \frac{z \tau_*}{2} \right)^{-2\nu}
    \frac{\Gamma^2 \left( \frac{1}{2} + \nu \right)}{4 \pi}
    \Upsilon_{2d}^2 \left( \frac{\tau_*}{\tau_\text{end}} \right),$$
    where $d = \nu$ by default.
    This is the non-oscillating part of the leading-order sound wave kernel $\Delta_\text{sw}^\text{lo}$ for
    super-horizon modes $k \eta_* = z \tau_* \ll 1$ in the limit $z \rightarrow 0$
    (:giombi_2026:`\ ` eq. 2.41a and C.6a (arXiv v2)),
    and a part of :giombi_2026:`\ ` eq. 3.5a.
    The factor $\Upsilon_{2d}$ comes from the time integral
    $\int_{\tau_*}^{\tau_\text{end}} \frac{d\tau}{\tau_*} \left( \frac{\tau_*}{\tau} \right)^{1 + 2d}$,
    and it is squared, as the kernel is a double integral over $\tau_1$ and $\tau_2$.
    For $\nu = 0$ this reduces to the bag model kernel :py:func:`kernel_low_bag`,
    $\frac{1}{4} \ln^2 \left( \frac{\tau_\text{end}}{\tau_*} \right)$.

    The articles assume $c_s^2 = \omega$, in which case $\nu$ describes both the expansion of the background
    and the acoustic friction of the sound waves (:giombi_2024_cs:`\ ` sec. 4.2).
    The factors $\left( \frac{z \tau_*}{2} \right)^{-2\nu} \Gamma^2 \left( \frac{1}{2} + \nu \right)$
    come from the Green's function of the GWs in the expanding background (:giombi_2026:`\ ` eq. 2.40 (arXiv v2)).
    In the time integral, the factor $(\tau_* / \tau)^{1 - \nu}$ from the background (:giombi_2024_cs:`\ ` eq. 2.16)
    and $(\tau / \tau_*)^{-\nu}$ from the Green's function combine with $(\tau_* / \tau)^{2d}$
    from the acoustic friction of the two sound waves (:giombi_2024_cs:`\ ` eq. 2.28, 2.31) to
    $(\tau_* / \tau)^{1 + 2d}$.
    If $c_s$ is given, the friction exponent $d$ is computed from it with
    :py:func:`acoustic_friction_exponent`, which allows $c_s^2 \neq \omega$.
    This generalisation is not in the articles, and is therefore disabled by default.
    It should be considered an estimate, see the caveat in :py:func:`acoustic_friction_exponent`.

    :param z: $z = kR_*$
    :param nu: $\nu_\text{gdh2024}$, a scalar
    :param tau_star: $\tau_* = \frac{\eta_*}{R_*}$
    :param tau_end: $\tau_\text{end} = \frac{\eta_\text{end}}{R_*}$
    :param cs: $c_s$, speed of sound for the acoustic friction. If not given, $d = \nu$.
    :return: $\Delta_\text{low}$
    """
    d = nu if cs is None else acoustic_friction_exponent(nu=nu, cs=cs)
    return (
        (0.5 * z * tau_star) ** (-2 * nu)  # pyrefly: ignore[bad-return]
        * math.gamma(0.5 + nu) ** 2 / (4 * np.pi)
        * Upsilon(r=tau_star / tau_end, l=2 * d) ** 2
    )


@njit(cache=True)
def kernel_low_bag[T: FloatOrArr](tau_star: T, tau_end: T | float) -> T:
    r"""$\Delta_\text{low}^{\eta=0}$, low-frequency kernel for bag model (radiation domination).

    $$\Delta_\text{low}^{\eta=0} (x, \tau_*, \tau_\text{end}) \rightarrow_{k \rightarrow 0}
    \frac{1}{4} \ln^2 \left( \frac{\tau_\text{end}}{\tau_*} \right)$$
    :giombi_2024_cs:`\ ` eq. 3.4
    The ci and si terms are negligible and have been dropped.

    :param tau_star: $\tau_* = \frac{\eta_*}{R_*}$
    :param tau_end: $\tau_\text{end} = \frac{\eta_\text{end}}{R_*}$
    :return: $\Delta_\text{low}^{\eta=0}$
    """
    return 0.25 * np.log(tau_end / tau_star)**2  # pyrefly: ignore[bad-return]
