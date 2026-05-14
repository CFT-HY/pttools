"""Kernel functions"""

import numpy as np
from scipy.special import gamma

from pttools.ssm.barotropic import Upsilon
from pttools.type_hints import FloatOrArr


def kernel_int_bracket[T: FloatOrArr](cs: T) -> T:
    r"""The term in brackets in $\tilde{P}_\text{gw}^\text{int}$
    $$3 - 2 c_s^2 - \frac{3}{c_s}(1 - c_s^2) \arctanh(c_s)$$
    :giombi_2024_cs:`\ ` eq. 3.11
    """
    return 3 - 2 * cs ** 2 - 3 / cs * (1 - cs ** 2) * np.arctanh(cs)


def kernel_low(z: FloatOrArr, nu: FloatOrArr, tau_star: FloatOrArr, tau_end: FloatOrArr) -> FloatOrArr:
    r"""Low-frequency kernel $\Delta_\text{low}$ for $c_s \neq \frac{1}{\sqrt{3}}$
    $$\Delta_\text{low} = \left( \frac{z \tau_*} \right)^{-2\nu}
    \frac{\Gamma \left( \frac{1}{2} + \nu \right)}{4 \pi}
    \Upsilon_{2\nu} \left( \frac{\tau_*}{\tau_\text{end}} \right)$$
    This is a part of
    :giombi_2026:`\ ` eq. 3.5a
    """
    # TODO: Ensure that this formula from Lorenzo's code is correct.
    return (0.5 * z * tau_star) ** (-2 * nu) * \
        gamma(0.5 + nu) ** 2 / (4 * np.pi) * \
        Upsilon(r=tau_star / tau_end, l=2 * nu)


def kernel_low_bag(tau_star: FloatOrArr, tau_end: FloatOrArr):
    r"""Low-frequency kernel for bag model (radiation domination)
    $$\Delta_\text{low}^{\eta=0} (x, \tau_*, \tau_\text{end}) \rightarrow_{k \rightarrow 0}
    \frac{1}{4} \ln^2 \left( \frac{\tau_\text{end}}{\tau_*} \right)$$
    :giombi_2024_cs:`\ ` eq. 3.4
    The ci and si terms are negligible and have been dropped.
    """
    return 0.25 * np.log(tau_end / tau_star)**2
