"""Functions for computing the spectral density of the gravitational waves"""

import numba
from numba.extending import overload
import numpy as np

from pttools.bubble.const import DEFAULT_ADIABATIC_INDEX, DEFAULT_NU_GDH2024
from pttools.speedup import NUMBA_ENABLE_CACHE, logspace
from pttools.ssm.barotropic import H_eta
from pttools.ssm.const import CS0, DEFAULT_N_Z_LOOKUP, DEFAULT_R_STAR
from pttools.ssm.rho import rho_delta_factor, rho_delta_frac, x_minus, x_plus
from pttools.type_hints import FloatArr1D, FloatOrArr, NumbaFunc


@numba.njit
def gen_lookup(
        y: FloatArr1D,
        cs: float = CS0,
        n_x_lookup: int = DEFAULT_N_Z_LOOKUP,
        eps: float = 0.) -> FloatArr1D:
    """
    :param y: Input array
    :param cs: Speed of sound $c_s$
    :param n_x_lookup: Number of points for the generated lookup table
    :param eps: Seems to be needed for max(z) <= 100. E.g. 1e-8. Why?
    :return: Generated lookup array for $x$
    """
    x_minus_min, x_plus_max = lookup_limits(y, cs, eps)
    # If the eps were summed instead of multiplied, then we would have to check for negative z_minus_min.
    # if z_minus_min <= 0:
    #     z_minus_min_old = z_minus_min
    #     eps_old = eps
    #     while z_minus_min <= 0:
    #         eps *= 0.1
    #         z_minus_min, z_plus_max = lookup_limits(y, cs, eps)
    #     with numba.objmode:
    #         logger.warning(
    #             "Got z_minus_min=%s <= 0 with eps=%s. Recomputed to %s with eps=%s.",
    #             z_minus_min_old, eps_old, z_minus_min, eps
    #         )

    # The variable to integrate over in eq. 3.44 and 3.47
    return logspace(np.log10(x_minus_min), np.log10(x_plus_max), n_x_lookup)


@numba.njit
def limits_from_lookup[T: FloatOrArr](x_lookup: FloatArr1D, cs: T = CS0) -> tuple[T, T]:
    r"""Limits of x from a lookup
    $$y_\pm = 2 x_{\pm} \frac{c_s}{1 \pm c_s}$$
    The inverse of :py:func:lookup_limits: from :gw_pt_ssm:`\ ` p. 12
    """
    y_min = x_lookup.min() * 2. * cs / (1. - cs)
    y_max = x_lookup.max() * 2. * cs / (1. + cs)
    return y_min, y_max


@numba.njit
def lookup_limits(z: FloatArr1D, cs: float = CS0, eps: float = 0.) -> tuple[float, float]:
    r"""
    $$x_\pm = z \frac{1 \pm c_s}{2c_s}$$
    :giombi_2024_cs: \ ` p. 13
    :giombi_2026: \ ` p. 25
    This is denoted as $z_\pm$ on :gw_pt_ssm:`\ ` p. 12.
    """
    x_minus_min = x_minus(z=z.min(), cs=cs) * (1 - eps)
    x_plus_max = x_plus(z=z.max(), cs=cs) * (1 + eps)
    return x_minus_min, x_plus_max


def spec_den_gw_scaling(
        ubarf2: float,
        mean_adiabatic_index: FloatOrArr = DEFAULT_ADIABATIC_INDEX,
        r_star: FloatOrArr = DEFAULT_R_STAR,
        nu: FloatOrArr = DEFAULT_NU_GDH2024,
        dilution_of_e: FloatOrArr = 1.,
        suppression_factor: FloatOrArr = 1.) -> FloatOrArr:
    r"""Scaling factor for $\tilde{P}_\text{gw}$

    $$3 (\Gamma \bar{U}_f)^2 r_* \mathcal{H} \eta_*
    \left( \frac{a_*}{a_r} \right)^\frac{2\nu}{1 + \nu}
    \Sigma
    \tilde{P}_\text{gw}$$
    :giombi_2026:`\ ` eq. 3.9a

    The pre-factors of :gw_pt_ssm:`\ ` eq. 3.44, 3.45 and 3.56 provide
    $$\frac{1}{H} \frac{1}{12 H^2} \frac{k^3}{2 \pi^2} \left( 16 \pi G \bar{w} \bar{U}_f^2 \right)^2 L_f^4
    = 3 \left( \frac{\bar{w}}{\bar{e}} \bar{U}_f^2 \right)^2 H L_f \frac{(k L_f)^3}{2 \pi^2}
    = 3 \left( \Gamma \bar{U}_f^2 \right)^2 r_* \frac{z}{2 \pi^2}$$
    In the first intermediate step, we have used the Friedmann equation $H^2 = \frac{8\pi G}{3} \bar{e}$.
    In the second step, we have defined $\Gamma \equiv \frac{\bar{w}}{\bar{e}}$ and $r_* \equiv H L_f$.

    The pre-factor can be related to the kinetic energy fraction with
    $$K = \Gamma \bar{U}_f^2.$$
    :gw_pt_ssm:`\ ` eq. B.32
    However, this is exact only when using the definitions $\Gamma \equiv \frac{\bar{w}{\bar{e}}$ and
    $\bar{U}_f^2 \equiv \frac{3}{4 \pi \bar{w} v_\text{wall}^3} e_K$.
    Therefore, this function takes in $\Gamma$ and $\bar{U}_f^2$ as separate input arguments instead of $K$.

    Another reason that this function takes in $\Gamma$ and $\bar{U}_f^2$ as separate input parameters
    is to ensure that the $\bar{U}_f^2$ cancels out exactly the $\bar{U}_f^{-2}$ in
    :py:func:spec_den_v:.
    Therefore, please ensure that you give the same $\bar{U}_f^2$ to both functions.

    :param ubarf2: $\bar{U}_f^2$. Use the same value as for :py:func:spec_den_v:.
    :param mean_adiabatic_index: Mean adiabatic index $\Gamma \equiv \frac{\bar{w}}{\bar{e}}$.
    :param r_star: Hubble-scaled mean bubble separation $r_* \equiv H R_* \equiv H L_f$.
    :param nu: $\nu_\text{gdh2024}$
    :param dilution_of_e: energy dilution factor
    :param suppression_factor: Suppression factor from comparison with lattice simulations
    """
    return 3. * (mean_adiabatic_index * ubarf2)**2 * r_star * H_eta(nu) * dilution_of_e * suppression_factor


def _spec_den_gw_core(
        z_lookup: FloatArr1D,
        P_tilde_v_lookup: FloatArr1D,
        y: FloatArr1D,
        cs: float = CS0,
        source_lifetime_factor: float = 1.,
        nz_int: int | None = None) -> tuple[FloatArr1D, FloatArr1D]:
    r"""Core computation for :py:func:spec_den_gw_scaled:
    :giombi_2024_cs:`\ ` eq. 3.13
    Old version:
    :gw_pt_ssm:`\ ` eq. 3.47 and 3.48

    Please note that in the older formulas the variable $x$ is called $z$.
    """
    # Creating a second variable is required by Numba
    nz_int2 = z_lookup.size if nz_int is None else nz_int

    # Precompute shared intermediate results
    x_plus_factor = x_plus(z=1., cs=cs)
    x_minus_factor = x_minus(z=1., cs=cs)
    p_gw_factor = rho_delta_factor(cs2=cs**2) / (4 * np.pi * cs)

    p_gw = np.zeros_like(y)
    for i in numba.prange(y.size):
        # As defined on page 12 between eq. 3.44 and 3.45
        xp = y[i] * x_plus_factor
        xm = y[i] * x_minus_factor
        # Create a range of x to integrate over
        x = logspace(np.log10(xm), np.log10(xp), nz_int2)
        # The integrand in eq. 3.47
        integrand = rho_delta_frac(x=x, xp=xp, xm=xm) * \
                    np.interp(x, z_lookup, P_tilde_v_lookup) * \
                    np.interp((xp + xm - x), z_lookup, P_tilde_v_lookup)
        p_gw[i] = source_lifetime_factor * p_gw_factor / y[i] * np.trapezoid(integrand, x)

    return p_gw, y

_spec_den_gw_core_single = numba.njit(nogil=True)(_spec_den_gw_core)
_spec_den_gw_core_parallel = numba.njit(parallel=True, nogil=True)(_spec_den_gw_core)


def _spec_den_gw_y(
        z_lookup: FloatArr1D,
        P_tilde_v_lookup: FloatArr1D,
        y: FloatArr1D | None = None,
        cs: float = CS0,
        source_lifetime_factor: float = 1.,
        nz_int: int | None = None,
        parallel: bool = True) -> tuple[FloatArr1D, FloatArr1D]:

    z_lookup_min, z_lookup_max = lookup_limits(y, cs)
    if z_lookup.max() < z_lookup_max or z_lookup.min() > z_lookup_min:
        raise ValueError("Range of z_lookup is not large enough.")

    if parallel:
        return _spec_den_gw_core_parallel(
            z_lookup=z_lookup, P_tilde_v_lookup=P_tilde_v_lookup, y=y,
            cs=cs, source_lifetime_factor=source_lifetime_factor,
            nz_int=nz_int
        )
    return _spec_den_gw_core_single(
        z_lookup=z_lookup, P_tilde_v_lookup=P_tilde_v_lookup, y=y,
        cs=cs, source_lifetime_factor=source_lifetime_factor,
        nz_int=nz_int
    )


def _spec_den_gw_no_y(
        z_lookup: FloatArr1D,
        P_tilde_v_lookup: FloatArr1D,
        y: FloatArr1D | None = None,
        cs: float = CS0,
        source_lifetime_factor: float = 1.,
        nz_int: int | None = None,
        parallel: bool = True) -> tuple[FloatArr1D, FloatArr1D]:
    z_min, z_max = limits_from_lookup(z_lookup, cs=cs)
    y = logspace(np.log10(z_min), np.log10(z_max), z_lookup.size)
    if parallel:
        return _spec_den_gw_core_parallel(
            z_lookup=z_lookup, P_tilde_v_lookup=P_tilde_v_lookup, y=y,
            cs=cs, source_lifetime_factor=source_lifetime_factor,
            nz_int=nz_int
        )
    return _spec_den_gw_core_single(
        z_lookup=z_lookup, P_tilde_v_lookup=P_tilde_v_lookup, y=y,
        cs=cs, source_lifetime_factor=source_lifetime_factor,
        nz_int=nz_int
    )


def spec_den_gw(
        # Arrays
        z_lookup: FloatArr1D,
        P_tilde_v_lookup: FloatArr1D,
        y: FloatArr1D | None = None,
        # Scalars
        cs: float = CS0,
        source_lifetime_factor: float = 1.,
        # Settings
        nz_int: int | None = None,
        parallel: bool = True) -> tuple[FloatArr1D, FloatArr1D] | NumbaFunc:
    r"""
    Spectral density of gravitational wave power, $\tilde{P}_\text{gw}(z)$

    $$\tilde{P}_\text{gw}(z) =
    \frac{1}{4\pi z c_s} \left( \frac{1 - c_s^2}{c_s^2} \right)^2 \Upsilon_\ell
    \int_{x_{-}}^{x_+} dx \frac{(x - x_+)^2(x - x_{-})^2}{x(x_+ + x_{-} - x)} \tilde{P}_v(x) \tilde{P}_v(x_+ + x_{-} - x)$$
    :giombi_2024_cs:`\ ` eq. 3.13
    Older versions of this formula are available in
    :gw_pt_ssm:`\ ` eq. 3.47, 3.48
    :maki_msc:`\ ` eq. 3.47, 3.48
    :gowling_phd:`\ ` eq. 3.33

    If you give this function $\bar{U}_f^2 \tilde{P}_{\tilde{v}}$ instead of $\tilde{P}_{\tilde{v}}$, then
    you get $\bar{U}_f^4 \tilde{P}_\text{gw}$ as output.

    :param z_lookup: Lookup table for the $z = qL_f$ values corresponding to P_v_lookup
    :param P_tilde_v_lookup: $\tilde{P}_v(z)$,
        a lookup table for the spectral density of the Fourier transform of the velocity field,
        not the spectral density of plane wave coefficients, which is lower by a factor of 2.
    :param y: $y = kL_f = kR*$ corresponding to z_lookup. If not given, will be created from z_lookup.
    :param cs: Speed of sound $c_s$ in the broken phase after the phase transition
    :param source_lifetime_factor: Source lifetime factor $\Upsilon_l$
    :param nz_int: Number of $z$ points for integration
    :param parallel: Whether to run with multiple threads
    :return: $\tilde{P}_\text{gw}(z)$
    """
    # The equation numbers in gw_pt_ssm and maki_msc happen to be the same.

    if z_lookup.shape != P_tilde_v_lookup.shape:
        raise TypeError(
            "z_lookup and P_tilde_v_lookup must be of the same shape. "
            f"Got z_lookup.shape={z_lookup.shape}, P_tilde_v_lookup.shape={P_tilde_v_lookup.shape}"
        )

    if isinstance(y, np.ndarray):
        return _spec_den_gw_y(
            z_lookup=z_lookup, P_tilde_v_lookup=P_tilde_v_lookup, y=y,
            cs=cs, source_lifetime_factor=source_lifetime_factor,
            nz_int=nz_int, parallel=parallel
        )
    if y is None:
        return _spec_den_gw_no_y(
            z_lookup=z_lookup, P_tilde_v_lookup=P_tilde_v_lookup, y=y,
            cs=cs, source_lifetime_factor=source_lifetime_factor,
            nz_int=nz_int, parallel=parallel
        )
    raise TypeError(f"Unknown type for y: {type(y)}")


@overload(spec_den_gw, jit_options={"nopython": True, "nogil": True, "cache": NUMBA_ENABLE_CACHE})
def _spec_den_gw_scaled_numba(
        # Arrays
        z_lookup: FloatArr1D,
        P_tilde_v_lookup: FloatArr1D,
        y: FloatArr1D | None = None,
        # Scalars
        cs: float = CS0,
        source_lifetime_factor: float = 1.,
        # Settings
        nz_int: int | None = None,
        parallel: bool = True) -> tuple[FloatArr1D, FloatArr1D] | NumbaFunc:
    if isinstance(y, numba.types.Array):
        return _spec_den_gw_y
    if isinstance(y, (numba.types.NoneType, numba.types.Omitted)):
        return _spec_den_gw_no_y
    raise TypeError(f"Unknown type for y: {type(y)}")
