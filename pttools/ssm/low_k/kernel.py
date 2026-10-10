"""Kernel functions."""

import numpy as np
from scipy.special import gamma

from pttools.ssm.barotropic import Upsilon
from pttools.type_hints import FloatOrArr


def kernel_int_bracket[T: FloatOrArr](cs: T) -> T:
    r"""The term in brackets in $\tilde{P}_\text{gw}^\text{int}$.

    $$3 - 2 c_s^2 - \frac{3}{c_s}(1 - c_s^2) \text{arctanh}(c_s)$$
    :giombi_2024_cs:`\ ` eq. 3.11.
    """
    # typing.cast() is not used below, since Numba cannot compile it.
    return 3 - 2 * cs ** 2 - 3 / cs * (1 - cs ** 2) * np.arctanh(cs)  # pyrefly: ignore[bad-return]


def kernel_low[T: FloatOrArr](z: T, nu: T | float, tau_star: T | float, tau_end: T | float) -> T:
    r"""$\Delta_\text{low}$, low-frequency kernel for $c_s \neq \frac{1}{\sqrt{3}}$.

    $$\Delta_\text{low} =
    \left( \frac{z \tau_*}{2} \right)^{-2\nu}
    \frac{\Gamma^2 \left( \frac{1}{2} + \nu \right)}{4 \pi}
    \Upsilon_{2\nu}^2 \left( \frac{\tau_*}{\tau_\text{end}} \right)$$
    This is the non-oscillating part of the leading-order sound wave kernel $\Delta_\text{sw}^\text{lo}$ for
    super-horizon modes $k \eta_* = z \tau_* \ll 1$ in the limit $z \rightarrow 0$
    (:giombi_2026:`\ ` eq. 2.41a and C.6a),
    and a part of :giombi_2026:`\ ` eq. 3.5a.
    The factor $\Upsilon_{2\nu}$ comes from the time integral
    $\int_{\tau_*}^{\tau_\text{end}} \frac{d\tau}{\tau_*} \left( \frac{\tau_*}{\tau} \right)^{1 + 2\nu}$,
    and it is squared, as the kernel is a double integral over $\tau_1$ and $\tau_2$.
    For $\nu = 0$ this reduces to :py:func:`kernel_low_bag`.
    """
    return (
        (0.5 * z * tau_star) ** (-2 * nu)  # pyrefly: ignore[bad-return]
        * gamma(0.5 + nu) ** 2 / (4 * np.pi)
        * Upsilon(r=tau_star / tau_end, l=2 * nu) ** 2
    )


def kernel_low_bag[T: FloatOrArr](tau_star: T, tau_end: T | float) -> T:
    r"""$\Delta_\text{low}^{\eta=0}$, low-frequency kernel for bag model (radiation domination).

    $$\Delta_\text{low}^{\eta=0} (x, \tau_*, \tau_\text{end}) \rightarrow_{k \rightarrow 0}
    \frac{1}{4} \ln^2 \left( \frac{\tau_\text{end}}{\tau_*} \right)$$
    :giombi_2024_cs:`\ ` eq. 3.4
    The ci and si terms are negligible and have been dropped.
    """
    return 0.25 * np.log(tau_end / tau_star)**2  # pyrefly: ignore[bad-return]
