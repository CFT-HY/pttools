"""Scaling factors"""

import numpy as np

from pttools.bubble.const import DEFAULT_NU_GDH2024
from pttools.ssm.barotropic import H_eta
from pttools.type_hints import FloatOrArr


def H_star_tau_sh(r_star: FloatOrArr, ubarf: FloatOrArr) -> FloatOrArr:
    r"""Hubble-scaled shock appearance timescale $H_* \tau_\text{sh}$
    $$H_* \tau_\text{sh} = \frac{r_*}{\bar{U}_f}$$
    :ajmi_2022:`\ ` p. 9
    """
    return r_star / ubarf


def H_star_tau_sh_approx(r_star: FloatOrArr, K: FloatOrArr) -> FloatOrArr:
    r"""Approximation of Hubble-scaled shock appearance timescale $H_* \tau_\text{sh}$
    $$H_* \tau_\text{sh} = \frac{r_*}{\bar{U}_f} \approx \frac{r_*}{\sqrt{K}$$
    :ajmi_2022:`\ ` p. 9
    """
    return r_star / np.sqrt(K)


def H_star_tau_v(source_lifetime_factor: FloatOrArr, nu: FloatOrArr = DEFAULT_NU_GDH2024) -> FloatOrArr:
    r"""$H_* \tau_v$
    $$\mathcal{H} \tau_v = \mathcal{H}_* \eta_* \Upsilon_\ell$$
    :ajmi_2022:`\ ` eq. 80
    :gowling_2021:`\ ` eq. 2.7
    """
    return H_eta(nu) * source_lifetime_factor


def H_star_tau_v_old[T: FloatOrArr](H_star_tau_sh: T) -> T:
    r"""Old approximation of $H_* \tau_v$
    $$H_* \tau_v \approx 1 - \frac{1}{\sqrt{1 + 2x}}$$,
    where $x = H_* \tau_\text{sh}$.
    :ajmi_2022:`\ ` eq. 80
    :gowling_2021:`\ ` eq. 2.7
    This is an approximation, and the source lifetime factor should be used instead.

    :param H_star_tau_sh: $H_* \tau_\text{sh}$
    """
    return 1 - (1 + 2 * H_star_tau_sh) ** (-0.5)


def J(r_star: FloatOrArr, H_star_tau_v: FloatOrArr) -> FloatOrArr:
    """$J = H_* R_* H_* \tau_v$
    :ajmi_2022:`\ ` eq. 81
    :gowling_2021:`\ ` eq. 2.8
    This is an approximation, and the source lifetime factor should be used instead.
    """
    return r_star * H_star_tau_v
