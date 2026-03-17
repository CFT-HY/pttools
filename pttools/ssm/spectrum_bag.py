"""Sound Shell Model spectrum calculations specific to the bag equation of state"""

import logging

import numpy as np

from pttools import bubble
from pttools.ssm import const, ssm, ssm_bag, spectrum
from pttools.ssm.spec_den_v import spec_den_v_core, spec_den_v_core_single
import pttools.type_hints as th

logger = logging.getLogger(__name__)


def convert_params(params: bubble.PhysicalParams) -> bubble.PhysicalParams:
    """Convert the physical parameters from a list to a tuple if necessary."""
    if isinstance(params, list):
        logger.warning("Specifying the model parameters as a list is deprecated. Please use a tuple instead.")
        return tuple(params)
    return params


# @numba.njit
def parse_params(params: bubble.PhysicalParams) -> tuple[float, float, spectrum.NucType, bubble.NucArgs]:
    r"""
    Parse physical parameters from the tuple.

    :param params: tuple of physical parameters
    :return: $v_\text{wall}, \alpha$, nucleation type, nucleation arguments
    """
    v_wall = params[0]
    alpha = params[1]
    if len(params) > 2:
        nuc_type = params[2]
    else:
        nuc_type = spectrum.DEFAULT_NUC_TYPE
    if len(params) > 3:
        nuc_args = params[3]
    else:
        nuc_args = const.DEFAULT_NUC_PARM

    return v_wall, alpha, nuc_type, nuc_args


def power_gw_scaled_bag(
        z: th.FloatArr1D,
        params: bubble.PhysicalParams,
        npt: const.NptType = const.DEFAULT_N_PT,
        filename: str | None = None,
        skip: int = 1,
        method: ssm.Method = ssm.Method.E_CONSERVING,
        de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
        z_st_thresh: float = const.Z_ST_THRESH,
        parallel: bool = True) -> th.FloatArr1D:
    """
    Scaled GW power spectrum at array of z = kR* values, where R* is mean bubble centre
    separation and k is comoving wavenumber.  To convert to predicted spectrum,
    multiply by $(H_n R_*)(H_n \tau_v)$, where $H_n$ is the Hubble rate at the
    nucleation time, and $\tau_v$ is the lifetime of the shear stress source.

    Physical parameters

    - vw = params[0]       scalar  (required) [0 < vw < 1]
    - alpha = params[1]    scalar  (required) [0 < alpha_n < alpha_n_max(v_w)]
    - nuc_type = params[2] string  (optional) [exponential* | simultaneous]
    - nuc_args = params[3] tuple   (optional) default (1,)

    Steps:

    1. Getting velocity field spectral density
    2. Geeting gw spectral density
    3. turning SD into power

    :param z: array $z = qR_*$
    :param params: physical parameters, see the description above
    :param npt: number of points
    :param filename: path to load A2 values from
    :return: scaled GW power spectrum
    """
    if np.any(z <= 0.0):
        raise ValueError("z values must all be positive.")
    params = convert_params(params)

    bubble.check_physical_params(params)

    # Todo: unify this generation
    eps = 1e-8  # Seems to be needed for max(z) <= 100. Why?
    #    nx = len(z) - this can be too few for velocity PS convolutions
    nx = npt[2]
    xmax = max(z) * (0.5 * (1. + const.CS0) / const.CS0) + eps
    xmin = min(z) * (0.5 * (1. - const.CS0) / const.CS0) - eps

    x = np.logspace(np.log10(xmin), np.log10(xmax), nx)

    sd_v = spec_den_v_bag(x, params, npt, filename, skip, method, de_method, z_st_thresh, parallel=parallel)
    sd_gw, y = spectrum.spec_den_gw_scaled(x, sd_v, z, parallel=parallel)
    return spectrum.pow_spec(z, sd_gw)


def power_v_bag(
        z: th.FloatArr1D,
        params: bubble.PhysicalParams,
        npt: const.NptType = const.DEFAULT_N_PT,
        filename: str | None = None,
        skip: int = 1,
        method: ssm.Method = ssm.Method.E_CONSERVING,
        de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
        z_st_thresh: float = const.Z_ST_THRESH,
        parallel: bool = True) -> th.FloatArr1D:
    """
    Power spectrum of the velocity field in the Sound Shell Model.

    - vw = params[0]       scalar
    - alpha = params[1]    scalar
    - nuc_type = params[2] string [exponential* | simultaneous]
    - nuc_args = params[3] tuple  default (1,)

    :param z: array $z = qR_*$
    :param params: physical parameters, see the description above
    :param npt: number of points
    :param filename: path to load A2 values from
    :param z_st_thresh: not used
    :return: power spectrum of the velocity field
    """
    bubble.check_physical_params(params)

    p_v = spec_den_v_bag(z, params, npt, filename, skip, method, de_method, parallel=parallel)
    return spectrum.pow_spec(z, p_v)


def spec_den_v_bag(
        z: th.FloatArr1D,
        params: bubble.PhysicalParams,
        npt: const.NptType = const.DEFAULT_N_PT,
        filename: str | None = None,
        skip: int = 1,
        method: ssm.Method = ssm.Method.E_CONSERVING,
        de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
        z_st_thresh=const.Z_ST_THRESH,
        parallel: bool = True):
    r"""
    Get dimensionless velocity spectral density $\bar{P}_v$.

    Gets fluid velocity profile from bubble toolbox or from file if specified.
    Convolves 1-bubble Fourier transform $|A(q T)|^2$ with bubble wall
    lifetime distribution $\nu(T \beta)$ specified by "nuc_type" and "nuc_args".

    :param z: array $z = qR_*$
    :param params: tuple of
        vw (scalar), alpha (scalar),
        nuc_type (string [exponential* | simultaneous]),
        nuc_args (tuple, default (1,))
    :param npt: number of points
    :return: dimensionless velocity spectral density $\bar{P}_v$
    """
    params = convert_params(params)
    bubble.check_physical_params(params)

    nz = z.size
    # nxi = npt[0]
    nT = npt[1]
    # nq = npt[2]

    # z limits
    log10zmin = np.log10(min(z))
    log10zmax = np.log10(max(z))
    dlog10z = (log10zmax - log10zmin) / nz

    # t limits
    T_min = const.T_TILDE_MIN
    T_max = const.T_TILDE_MAX
    log10_T_min = np.log10(T_min)
    log10_T_max = np.log10(T_max)

    qT_lookup = 10 ** np.arange(log10zmin + log10_T_min, log10zmax + log10_T_max, dlog10z)

    v_wall, alpha, nuc_type, nuc_args = parse_params(params)
    if filename is None:
        A2_lookup = ssm_bag.a2_ssm_func_bag(
            z=qT_lookup, v_wall=v_wall, alpha=alpha,
            npt=npt, method=method, de_method=de_method, z_st_thresh=z_st_thresh
        )
    else:
        A2_lookup = ssm_bag.a2_e_conserving_bag_file(
            z=qT_lookup, filename=filename, alpha=alpha,
            skip=skip, npt=npt, z_st_thresh=z_st_thresh
        )

    # if qT_lookup.size != A2_lookup.size:
    #     raise ValueError(f"Lookup sizes don't match: {qT_lookup.size} != {A2_lookup.size}")

    if parallel:
        return spec_den_v_core(
            a=nuc_args[0],
            A2_lookup=A2_lookup,
            log10tmin=log10_T_min,
            log10tmax=log10_T_max,
            nt=nT,
            nuc_type=nuc_type,
            qT_lookup=qT_lookup,
            v_wall=v_wall,
            z=z
        )
    return spec_den_v_core_single(
        a=nuc_args[0],
        A2_lookup=A2_lookup,
        log10tmin=log10_T_min,
        log10tmax=log10_T_max,
        nt=nT,
        nuc_type=nuc_type,
        qT_lookup=qT_lookup,
        v_wall=v_wall,
        z=z
    )
