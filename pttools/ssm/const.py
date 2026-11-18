"""Constants for the Sound Shell Model module."""

import typing as tp

import numpy as np

from pttools import bubble
import pttools.type_hints as th

__all__ = [
    "BETA_TILDE_MIN",
    "CS0",
    "CS0_2",
    "DEFAULT_A_STAR_A_R_RATIO",
    "DEFAULT_EPS_LOOKUP",
    "DEFAULT_LIFETIME_DISTRIBUTION_A",
    "DEFAULT_NUC_PARM",
    "DEFAULT_N_PT",
    "DEFAULT_N_SH",
    "DEFAULT_N_T",
    "DEFAULT_N_XI_SSM",
    "DEFAULT_N_Z_LOOKUP",
    "DEFAULT_OMEGA_TILDE_GW",
    "DEFAULT_R_STAR",
    "DEFAULT_Y",
    "DZ_ST_BLEND",
    "QT_INTEGRALS_MAX",
    "QT_INTEGRALS_MIN",
    "QT_INTEGRALS_POINTS_PER_DECADE",
    "T_TILDE_MAX",
    "T_TILDE_MIN",
    "X_IV_MAX",
    "X_IV_MIN",
    "X_IV_POINTS_PER_DECADE",
    "Z_ST_THRESH",
    "NptType",
]

#: Number of points (in old bag code)
NptType = th.IntArr1D | tuple[int, int, int]

# -----
# Default values
# -----
DEFAULT_A_STAR_A_R_RATIO: float = 1.
r"""
Default $\frac{a_*}{a_r}$,
the ratio of the scale factor $a_*$ at GW formation and $a_r$ at return to radiation dominance
:giombi_2024_cs:`\ ` eq. 2.18
"""

DEFAULT_N_SH: float = 1.
r"""
Default number of shock formation times $N_\text{sh}$
Note that $N_\text{sh}$ is not necessarily an integer.
:giombi_2024_cs:`\ ` eq. 4.1
"""

#: Default small offset used when generating the lookup table of $z$ values for the GW spectrum
DEFAULT_EPS_LOOKUP: float = 1e-8

#: Default parameter $a$ of the bubble lifetime distribution,
#: see :py:func:`pttools.ssm.nucleation.lifetime_distribution`
DEFAULT_LIFETIME_DISTRIBUTION_A: float = 1.

DEFAULT_N_T: int = 10000
r"""Default number $N_\text{T}$ of T (and $\tilde{T}$) values for bubble lifetime distribution integration
See :gw_pt_ssm:`\ ` p. 20
"""

DEFAULT_N_XI_SSM: int = 2000
r"""Default number $N_\xi$ of $\xi$ points used in SSM computations
See :gw_pt_ssm:`\ ` p. 20
"""

DEFAULT_N_Z_LOOKUP: int = 10000
r"""
Default number of wavevectors used in the velocity convolution integrations.
This should be at least as large as the default number of GW frequencies.
From :gw_pt_ssm:`\ ` p. 20
"""

#: Default number of points (in old bag code)
DEFAULT_N_PT: NptType = (DEFAULT_N_XI_SSM, DEFAULT_N_T, DEFAULT_N_Z_LOOKUP)

#: Default nucleation parameters
DEFAULT_NUC_PARM: tuple[int] = (1,)

DEFAULT_OMEGA_TILDE_GW: float = 0.012
r"""
Default $\tilde{\Omega}_\text{gw}$.

This is derived in
:py:func:`pttools.ssm.scaling.omega_tilde_gw`.

The value $\tilde{\Omega}_\text{gw} \approx 0.012$ is from
:hindmarsh_2017:`\ ` p. 13.
Please note that there is a typo in the original article: $0.12 \rightarrow 0.012$: :hindmarsh_2017_erratum:`\ `.
This value is obtained numerically from simulations.
This value is used in
:hakkinen_msc:`\ ` p. 28.

A rougher approximation of $\tilde{\Omega}_\text{gw}$ is given by
:caprini_2020:`\ ` p. 16
"""


#: Default $r_*$
DEFAULT_R_STAR: float = 1.

DEFAULT_Y: th.FloatArr1D = np.logspace(-1, 3, 1000)
r"""Default range for wavenumbers $y$
The range is from :gw_pt_ssm:`\ ` p. 20, but there TODO
"""

# It seems that NPTDEFAULT should be something like NXIDEFAULT/(2.pi), otherwise one
# gets a GW power spectrum which drifts up at high k.
#
# The maximum trustworthy k is approx NXIDEFAULT/(2.pi)
#
# NTDEFAULT can be left as it is, or even reduced to 100


# -----
# Limits
# -----

# BETA_TILDE_CONVERSION_MIN: float = 10.
# r"""
# Lower limit for $\tilde{\beta} \equiv \beta/H_*$ conversion
# The conversion $r_* \rightarrow \tilde{\beta}$ should give a warning when under this limit.
# This is because the $\tilde{\beta} \leftrightarrow r_*$ conversion breaks down
# in the case of a very slow phase transition, which corresponds to $\beta/H_* \approx 1$.
# :caprini_2020:`\ ` p. 6
# """

BETA_TILDE_MIN: float = 3.8
r"""
Lower limit for $\tilde{\beta} \equiv \beta/H_*$
This is a conservative estimate.
Below this value, the abundance of primordial black holes would be too high.
:giombi_2026:`\ ` below eq. 1.1
:lewicki_2023:`\ `
"""

QT_INTEGRALS_MAX: float = 2e5
r"""
Minimum upper limit of the $z = q\tilde{T}$ array of the integrals of the spectra,
see :py:func:`pttools.ssm.ssm.qT_integrals`.
For the integral of $\bar{U}_f^2$ in :py:func:`pttools.ssm.ssm.ubarf2_from_a2`,
the integrand $z^2 {\lvert A(z) \rvert}^2$ decays approximately as $z^{-2}$,
and therefore the relative error of the integral is approximately inversely proportional to its upper limit.
With this value, the error is below $4 \cdot 10^{-4}$ for the tested deflagrations, hybrids and detonations,
compared with up to 5 % for the upper limit of $2 \cdot 10^3$, and up to 44 % for $2 \cdot 10^2$.
"""

QT_INTEGRALS_MIN: float = 1e-3
r"""
Lower limit of the $z = q\tilde{T}$ array of the integrals of the spectra,
see :py:func:`pttools.ssm.ssm.qT_integrals`.
For the integral of $\bar{U}_f^2$, the part below 0.1 is below $10^{-7}$ of the total
for the tested deflagrations, hybrids and detonations.
The array extends lower, as $\tilde{P}_v(x)$ of $\mathcal{I}_v$ needs ${\lvert A \rvert}^2$ at
$z = x \tilde{T} / (\beta R_*)$ for $x \geq$ :py:data:`X_IV_MIN`.
Below this, ${\lvert A \rvert}^2$ is extrapolated as a constant,
which affects $\tilde{P}_v(x)$ only for $\tilde{T} \lesssim 10^{-2} \beta R_* / x$,
where the integrand of $\tilde{P}_v$ is suppressed by $\tilde{T}^6$.
"""

QT_INTEGRALS_POINTS_PER_DECADE: float = 1000.
r"""
Number of points per decade of the $z = q\tilde{T}$ array of the integrals of the spectra,
see :py:func:`pttools.ssm.ssm.qT_integrals`.
The integrand of $\bar{U}_f^2$ oscillates, and with this density,
the sampling error of the integral is up to about $2 \cdot 10^{-3}$
for the tested deflagrations, hybrids and detonations,
compared with up to 2 % for 300 and 6 % for 100 points per decade.
Increasing the density further does not reduce the error significantly,
as the oscillations become faster at large $z$.
"""

T_TILDE_MAX: float = 20.0
r"""
Maximum $\tilde{T}$ in bubble lifetime distribution integration
From :gw_pt_ssm:`\ ` p. 20
"""

T_TILDE_MIN: float = 0.01
r"""
Minimum $\tilde{T}$ in bubble lifetime distribution integration
From :gw_pt_ssm:`\ ` p. 20
"""

Z_ST_THRESH: float = 50.
"""
Default dimensionless wavenumber above which to use
:py:func:`pttools.ssm.sin_transform_approx.sin_transform_approx`
for
:py:func:`pttools.ssm.sin_transform.sin_transform`.
"""


# -----
# Constants
# -----

#: Default sound speed $c_s$
CS0: tp.Final[float] = bubble.CS0
#: Default sound speed squared $c_s^2$
CS0_2: tp.Final[float] = bubble.CS0_2

X_IV_MAX: float = 100.
r"""
Upper limit of the $x$ array of the source integral $\mathcal{I}_v$ of the low-$k$ approximation,
see :py:func:`pttools.ssm.low_k.integration.Iv`.
As $\tilde{P}_v \propto x^{-4}$ at large $x$ ($x^{-3}$ for some hybrids), the integrand $x^2 \tilde{P}_v^2$
decays as $x^{-6}$ or faster.
For the tested deflagrations, hybrids and detonations with exponential and simultaneous nucleation,
the part of $\mathcal{I}_v$ above this is below $10^{-4}$, compared with up to $2.4 \cdot 10^{-3}$ above 30
and $9 \cdot 10^{-2}$ above 10.
"""

X_IV_MIN: float = 0.1
r"""
Lower limit of the $x$ array of the source integral $\mathcal{I}_v$, see :py:data:`X_IV_MAX`.
For the tested bubbles, the part of $\mathcal{I}_v$ below this is below $2 \cdot 10^{-5}$.
"""

X_IV_POINTS_PER_DECADE: float = 300.
r"""
Number of points per decade of the $x$ array of the source integral $\mathcal{I}_v$, see :py:data:`X_IV_MAX`.
$\tilde{P}_v$ is smooth, and with this density, $\mathcal{I}_v$ is accurate to about $10^{-4}$.
"""

DZ_ST_BLEND: float = np.pi
"""
Default wavenumber overlap for matching
:py:func:`pttools.ssm.sin_transform_approx.sin_transform_approx`.
"""
