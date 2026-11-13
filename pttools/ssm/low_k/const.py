"""Constants for the low-k module."""

import math

__all__ = ["IV_ANALYTICAL", "JV_ANALYTICAL", "ZP_SW"]

IV_ANALYTICAL: float = 1 / (32 * math.pi ** 2)
r"""Source contribution $\mathcal{I}_v$ (analytical approximation)
$$\mathcal{I}_v = \frac{1}{32 \pi^2}$$
:giombi_2024_cs:`\ ` p. 11
:giombi_2026:`\ ` below eq. 3.6
"""

JV_ANALYTICAL: float = 1 / (128 * math.pi ** 4)
r"""$\mathcal{J}_v$ (analytical approximation)
$$\mathcal{J}_v = \frac{\mathcal{I}_v}{(2\pi)^2} = \frac{1}{128 \pi^4}$$
:giombi_2026:`\ ` below eq. 3.6
"""

ZP_SW: float = 2 * math.pi
r"""$z_{p,\text{sw}} = k_p R_* = 2\pi$, the sound wave scale in units of the mean bubble spacing.

$$k_p = \frac{2\pi}{R_*}$$
is the characteristic wavenumber of the sound waves, which :giombi_2024_cs:`\ ` (sec. 1) and
:giombi_2026:`\ ` (below eq. 3.1) call the peak frequency or peak wavenumber $k_p$.
In the code it is called ``kp_sw`` to distinguish it from the peak of the GW power spectrum,
$k_\star$ of the same articles, which is called ``kp_gw``
(:py:func:`pttools.ssm.low_k.intersection.zp_gw_approx`, and $z_p$ of the broken power law fits of
:py:mod:`pttools.omgw0`).
For the analytical $\tilde{P}_v$ of :py:func:`pttools.ssm.low_k.analytical.Pv_analytical`,
$k_p$ is $p$ at the maximum of $p P_v(p)$.
For the $\tilde{P}_v$ of the Sound Shell Model it is not, see :py:func:`pttools.ssm.low_k.intersection.xp_v`.
"""
