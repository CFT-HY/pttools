"""Functions for computing the spectral density of the gravitational waves"""

import typing as tp

import numba
from numba.extending import overload
import numpy as np

import pttools.type_hints as th
from pttools import speedup
from pttools.ssmtools import const


@numba.njit
def gen_lookup(y: np.ndarray, cs: float, n_z_lookup: int = const.N_Z_LOOKUP_DEFAULT, eps: float = 0.) -> np.ndarray:
    """
    :param y: Input array
    :param cs: Speed of sound $c_s$
    :param n_z_lookup: Number of points for the generated lookup table
    :param eps: Seems to be needed for max(z) <= 100. E.g. 1e-8. Why?
    :return: Generated lookup array for z
    """
    z_minus_min, z_plus_max = lookup_limits(y, cs, eps)
    # The variable to integrate over in eq. 3.44 and 3.47
    return speedup.logspace(np.log10(z_minus_min), np.log10(z_plus_max), n_z_lookup)


@numba.njit
def lookup_limits(y: np.ndarray, cs: float, eps: float = 0.) -> tp.Tuple[float, float]:
    """Defined on p. 12 between eq. 3.44 and 3.45"""
    return y.min() * 0.5 * (1. - cs) / cs - eps, y.max() * 0.5 * (1. + cs) / cs + eps


@numba.njit(parallel=True, nogil=True)
def _spec_den_gw_scaled_core(
        z_lookup: np.ndarray,
        P_v_lookup: np.ndarray,
        y: np.ndarray,
        cs: float,
        Gamma: float,
        source_lifetime_factor: float,
        nz_int: int) -> tp.Tuple[np.ndarray, np.ndarray]:
    r""":gw_pt_ssm:`\ ` eq. 3.47 and 3.48
    The variable naming corresponds to the article.
    """
    if z_lookup.shape != P_v_lookup.shape:
        raise TypeError("z_lookup and P_v_lookup must be of the same shape.")

    # This trickery is required by Numba
    nz_int2 = z_lookup.size if nz_int is None else nz_int

    # Precompute shared intermediate results
    cs2: float = cs ** 2
    z_plus_factor = (1 + cs) / (2 * cs)
    z_minus_factor = (1 - cs) / (2 * cs)
    p_gw_factor = ((1 - cs2) / cs2) ** 2 / (4 * np.pi * cs)

    p_gw: np.ndarray = np.zeros_like(y)
    for i in numba.prange(y.size):
        # As defined on page 12 between eq. 3.44 and 3.45
        z_plus = y[i] * z_plus_factor
        z_minus = y[i] * z_minus_factor
        # Create a range of z to integrate over
        z = speedup.logspace(np.log10(z_minus), np.log10(z_plus), nz_int2)
        # The integrand in eq. 3.47
        integrand = \
            ((z - z_plus) ** 2 * (z - z_minus) ** 2) / (z * (z_plus + z_minus - z)) \
            * np.interp(z, z_lookup, P_v_lookup) \
            * np.interp((z_plus + z_minus - z), z_lookup, P_v_lookup)
        p_gw[i] = p_gw_factor / y[i] * np.trapezoid(integrand, z)

    # Eq. 3.48 has a factor of 3 Gamma^2
    # The P_v_lookup is 0.5 \tilde{P}_v, which gives a factor of (1/2)^2 = 1/4
    # Combined, these result in 3/4 Gamma^2
    return 0.75 * Gamma ** 2 * p_gw * source_lifetime_factor, y


def _spec_den_gw_scaled_y(
        z_lookup: np.ndarray,
        P_v_lookup: np.ndarray,
        y: np.ndarray,
        cs: float,
        Gamma: float,
        source_lifetime_factor: float,
        nz_int: int) -> tp.Tuple[np.ndarray, np.ndarray]:

    z_lookup_min, z_lookup_max = lookup_limits(y, cs)
    if z_lookup.max() < z_lookup_max or z_lookup.min() > z_lookup_min:
        raise ValueError("Range of z_lookup is not large enough.")

    return _spec_den_gw_scaled_core(z_lookup, P_v_lookup, y, cs, Gamma, source_lifetime_factor, nz_int)


def _spec_den_gw_scaled_no_y(
        z_lookup: np.ndarray,
        P_v_lookup: np.ndarray,
        y: None,
        cs: float,
        Gamma: float,
        source_lifetime_factor: float,
        nz_int: int) -> tp.Tuple[np.ndarray, np.ndarray]:
    # This process is the reverse of to gen_lookup()
    zmax = z_lookup.max() * 2. * cs / (1. + cs)
    zmin = z_lookup.min() * 2. * cs / (1. - cs)
    y = speedup.logspace(np.log10(zmin), np.log10(zmax), z_lookup.size)
    return _spec_den_gw_scaled_core(z_lookup, P_v_lookup, y, cs, Gamma, source_lifetime_factor, nz_int)


def spec_den_gw_scaled(
        z_lookup: np.ndarray,
        P_v_lookup: np.ndarray,
        y: np.ndarray = None,
        cs: float = const.CS0,
        Gamma: float = const.GAMMA,
        source_lifetime_factor: float = 1.,
        nz_int: int = None) -> tp.Union[tp.Tuple[np.ndarray, np.ndarray], th.NumbaFunc]:
    r"""
    Spectral density of scaled gravitational wave power

    :param z_lookup: Lookup table for the $z = qL_f$ values corresponding to P_v_lookup
    :param P_v_lookup: $\bar{U}_f^2 \tilde{P}_v (z)$,
        a lookup table for the spectral density of the Fourier transform of the velocity field,
        not the spectral density of plane wave coefficients, which is lower by a factor of 2.
    :param y: $y = kL_f = kR*$ corresponding to z_lookup. If not given, will be created from z_lookup.
    :param cs: Speed of sound (in the broken phase after the phase transition)
    :param Gamma: Mean adiabatic index $\Gamma = \frac{\bar{w}}{\bar{e}}$
    :return: $\hat{\mathcal{P}}$ Eq. 3.33 of Chloe's thesis, which should be ($3\Gamma \bar{U}_f$) Eq. 3.47
        Eq. 3.46 converted to the spectral density and divided by (H L_f)

    The factor of 3 comes from the Friedmann equation
    3H^2/(8pi G)
    """
    if isinstance(y, np.ndarray):
        return _spec_den_gw_scaled_y(z_lookup, P_v_lookup, y, cs, Gamma, source_lifetime_factor, nz_int)
    if y is None:
        return _spec_den_gw_scaled_no_y(z_lookup, P_v_lookup, y, cs, Gamma, source_lifetime_factor, nz_int)
    raise TypeError(f"Unknown type for z: {type(y)}")


@overload(spec_den_gw_scaled, jit_options={"nopython": True, "nogil": True})
def _spec_den_gw_scaled_numba(
        xlookup: np.ndarray,
        P_vlookup: np.ndarray,
        z: np.ndarray = None,
        cs: float = const.CS0) -> tp.Union[tp.Tuple[np.ndarray, np.ndarray], th.NumbaFunc]:
    if isinstance(z, numba.types.Array):
        return _spec_den_gw_scaled_y
    if isinstance(z, (numba.types.NoneType, numba.types.Omitted)):
        return _spec_den_gw_scaled_no_y
    raise TypeError(f"Unknown type for z: {type(z)}")
