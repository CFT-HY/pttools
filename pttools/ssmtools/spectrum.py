"""Functions for computing GW power spectra"""

import enum
import typing as tp

import numpy as np

import pttools.type_hints as th
from pttools import bubble
from . import const
from . import ssm


@enum.unique
class NucType(str, enum.Enum):
    EXPONENTIAL = "exponential"
    SIMULTANEOUS = "simultaneous"


DEFAULT_NUC_TYPE = NucType.EXPONENTIAL


def nu(T: th.FLOAT_OR_ARR, nuc_type: NucType = NucType.SIMULTANEOUS, args=(1,)) -> th.FLOAT_OR_ARR:
    """
    Bubble lifetime distribution function as function of (dimensionless) time T.
    ``nuc_type`` allows ``simultaneous`` or ``exponential`` bubble nucleation.
    """
    if nuc_type is NucType.SIMULTANEOUS:
        a = args[0]
        dist = 0.5 * a * (a*T)**2 * np.exp(-(a*T)**3 / 6)
    elif nuc_type is NucType.EXPONENTIAL:
        a = args[0]
        dist = a * np.exp(-a*T)
    else:
        raise ValueError(f"Nucleation type not recognized: \"{nuc_type}\"")

    return dist


def pow_spec(z: th.FLOAT_OR_ARR, spec_den: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
    Power spectrum from spectral density at dimensionless wavenumber z.
    """
    return z**3 / (2. * np.pi ** 2) * spec_den


def parse_params(params: bubble.PHYSICAL_PARAMS_TYPE):
    vw = params[0]
    alpha = params[1]
    if len(params) > 2:
        nuc_type = params[2]
    else:
        nuc_type = DEFAULT_NUC_TYPE
    if len(params) > 3:
        nuc_args = params[3]
    else:
        nuc_args = const.DEFAULT_NUC_PARM

    return vw, alpha, nuc_type, nuc_args


def spec_den_v(
        z: np.ndarray,
        params: bubble.PHYSICAL_PARAMS_TYPE,
        npt: const.NPT_TYPE = const.NPTDEFAULT,
        filename: str = None,
        skip: int = 1,
        method: ssm.Method = ssm.Method.E_CONSERVING,
        de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
        z_st_thresh=const.Z_ST_THRESH):
    """
    Returns dimensionless velocity spectral density $\bar{P}_v$, given array $z = qR_*$ and parameters:
    vw = params[0]       scalar
    alpha = params[1]    scalar
    nuc_type = params[2] string [exponential* | simultaneous]
    nuc_args = params[3] tuple  default (1,)

    Gets fluid velocity profile from bubble toolbox or from file if specified.
    Convolves 1-bubble Fourier transform $|A(q T)|^2$ with bubble wall
    lifetime distribution $\nu(T \beta)$ specified by ``nuc_type`` and ``nuc_args``.
    """
    bubble.check_physical_params(params)

    nz = z.size
    #    nxi = npt[0]
    nt = npt[1]
    #    nq = npt[2]

    log10zmin = np.log10(min(z))
    log10zmax = np.log10(max(z))
    dlog10z = (log10zmax - log10zmin) / nz

    tmin = const.T_TILDE_MIN
    tmax = const.T_TILDE_MAX
    log10tmin = np.log10(tmin)
    log10tmax = np.log10(tmax)

    t_array = np.logspace(log10tmin, log10tmax, nt)

    qT_lookup = 10 ** (np.arange(log10zmin + log10tmin, log10zmax + log10tmax, dlog10z))

    if filename is None:
        vw, alpha, nuc_type, nuc_args = parse_params(params)
        A2_lookup = ssm.A2_ssm_func(qT_lookup, vw, alpha, npt, method, de_method, z_st_thresh)
    else:
        vw, alpha, nuc_type, nuc_args = parse_params(params)
        A2_lookup = ssm.A2_e_conserving_file(qT_lookup, filename, alpha, skip, npt, z_st_thresh)

    b_R = (8. * np.pi) ** (1. / 3.)  # $\beta R_* = b_R v_w $

    def qT_array(qRstar, Ttilde):
        return qRstar * Ttilde / (b_R * vw)

    A2_2d_array = np.zeros((nz, nt))

    for i in range(nz):
        A2_2d_array[i] = np.interp(qT_array(z[i], t_array), qT_lookup, A2_lookup)

    # array2 = np.zeros(nt)
    sd_v = np.zeros(nz)  # array for spectral density of v
    factor = 1. / (b_R * vw) ** 6
    factor = 2 * factor  # because spectral density of v is 2 * P_v

    for s in range(nz):
        array2 = t_array ** 6 * nu(t_array, nuc_type, nuc_args) * A2_2d_array[s]
        D = np.trapz(array2, t_array)
        sd_v[s] = D * factor

    return sd_v


def spec_den_gw_scaled(
        xlookup: np.ndarray,
        P_vlookup: np.ndarray,
        z: np.ndarray = None) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Spectral density of scaled gravitational wave power at values of kR* given
    by input z array, or at len(xlookup) values of kR* between the min and max
    of xlookup where the GW power can be computed.
    (xlookup, P_vlookup) is used as a lookup table to specify function.
    P_vlookup is the spectral density of the FT of the velocity field,
    not the spectral density of plane wave coeffs, which is lower by a
    factor of 2.
    """

    if z is None:
        nx = len(xlookup)
        zmax = max(xlookup) / (0.5 * (1. + const.CS0) / const.CS0)
        zmin = min(xlookup) / (0.5 * (1. - const.CS0) / const.CS0)
        z = np.logspace(np.log10(zmin), np.log10(zmax), nx)
    else:
        #        nx = len(z)
        nx = len(xlookup)
        xlargest = max(z) * 0.5 * (1. + const.CS0) / const.CS0
        xsmallest = min(z) * 0.5 * (1. - const.CS0) / const.CS0

        if max(xlookup) < xlargest or min(xlookup) > xsmallest:
            raise ValueError("Range of xlookup is not large enough.")

    p_gw = np.zeros_like(z)

    for i in range(z.size):
        xplus = z[i] / const.CS0 * (1. + const.CS0) / 2.
        xminus = z[i] / const.CS0 * (1. - const.CS0) / 2.
        x = np.logspace(np.log10(xminus), np.log10(xplus), nx)
        integrand = \
            (x - xplus) ** 2 * (x - xminus) ** 2 / x / (xplus + xminus - x) \
            * np.interp(x, xlookup, P_vlookup) \
            * np.interp((xplus + xminus - x), xlookup, P_vlookup)
        p_gw_factor = ((1 - const.CS0 ** 2) / const.CS0 ** 2) ** 2 / (4 * np.pi * z[i] * const.CS0)
        p_gw[i] = p_gw_factor * np.trapz(integrand, x)

    # Here we are using G = 2P_v (v spec den is twice plane wave amplitude spec den).
    # Eq 3.48 in SSM paper gives a factor 3.Gamma^2.P_v.P_v = 3 * (4/3)^2.P_v.P_v
    # Hence overall should use (4/3).G.G
    return (4. / 3.) * p_gw, z


def power_v(
        z: np.ndarray,
        params: bubble.PHYSICAL_PARAMS_TYPE,
        npt=const.NPTDEFAULT,
        filename: str = None,
        skip: int = 1,
        method: ssm.Method = ssm.Method.E_CONSERVING,
        de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
        z_st_thresh: float = const.Z_ST_THRESH) -> np.ndarray:
    """
    Power spectrum of velocity field in Sound Shell Model.
        vw = params[0]       scalar
        alpha = params[1]    scalar
        nuc_type = params[2] string [exponential* | simultaneous]
        nuc_args = params[3] tuple  default (1,)
    """
    bubble.check_physical_params(params)

    p_v = spec_den_v(z, params, npt, filename, skip, method, de_method)
    return pow_spec(z, p_v)


def power_gw_scaled(
        z: np.ndarray,
        params: bubble.PHYSICAL_PARAMS_TYPE,
        npt=const.NPTDEFAULT,
        filename: str = None,
        skip: int = 1,
        method: ssm.Method = ssm.Method.E_CONSERVING,
        de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
        z_st_thresh: float = const.Z_ST_THRESH) -> np.ndarray:
    """
    Scaled GW power spectrum at array of z = kR* values, where R* is mean bubble centre
    separation and k is comoving wavenumber.  To convert to predicted spectrum,
    multiply by $(H_n R_*)(H_n \tau_v)$, where $H_n$ is the Hubble rate at the
    nucleation time, and $\tau_v$ is the lifetime of the shear stress source.

    Input parameters
        vw = params[0]       scalar  (required) [0 < vw < 1]
        alpha = params[1]    scalar  (required) [0 < alpha_n < alpha_n_max(v_w)]
        nuc_type = params[2] string  (optional) [exponential* | simultaneous]
        nuc_args = params[3] tuple   (optional) default (1,)

    Steps:
    1. Getting velocity field spectral density
    2. Geeting gw spectral density
    3. turning SD into power
    """
    if np.any(z <= 0.0):
        raise ValueError("z values must all be positive.")

    bubble.check_physical_params(params)

    eps = 1e-8  # Seems to be needed for max(z) <= 100. Why?
    #    nx = len(z) - this can be too few for velocity PS convolutions
    nx = npt[2]
    xmax = max(z) * (0.5 * (1. + const.CS0) / const.CS0) + eps
    xmin = min(z) * (0.5 * (1. - const.CS0) / const.CS0) - eps

    x = np.logspace(np.log10(xmin), np.log10(xmax), nx)

    sd_v = spec_den_v(x, params, npt, filename, skip, method, de_method, z_st_thresh)
    sd_gw, y = spec_den_gw_scaled(x, sd_v, z)
    return pow_spec(z, sd_gw)
