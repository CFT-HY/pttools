"""Functions for computing GW power spectra"""

import enum
import logging
import typing as tp

import matplotlib.pyplot as plt
import numba
import numpy as np

import pttools.type_hints as th
from pttools import bubble
from pttools import speedup
from . import const
from . import ssm

if tp.TYPE_CHECKING:
    from pttools.analysis.utils import FigAndAxes

logger = logging.getLogger(__name__)


@enum.unique
class NucType(str, enum.Enum):
    """Nucleation type"""
    EXPONENTIAL = "exponential"
    SIMULTANEOUS = "simultaneous"


#: Default nucleation type
DEFAULT_NUC_TYPE = NucType.EXPONENTIAL


class Spectrum:
    """Gravitational wave simulation object"""
    def __init__(
            self,
            bubble: bubble.Bubble,
            z: np.ndarray = None,
            z_st_thresh: float = const.Z_ST_THRESH,
            nuc_type: NucType = DEFAULT_NUC_TYPE,
            # method: ssm.Method = ssm.Method.E_CONSERVING,
            # de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
            # nxi: int = 5000,
            nt: int = 10000,
            # nq: int = 1000,
            compute: bool = True):
        """
        :param bubble: Fluid simulation object
        :param z: $z = kR*$ array
        """
        self.bubble = bubble
        # self.de_method = de_method
        # self.method = method
        self.nuc_type = nuc_type
        self.z = np.logspace(np.log10(0.2), np.log10(1000), 5000) if z is None else z
        self.z_st_thresh = z_st_thresh
        # self.nxi = nxi
        self.nt = nt
        # self.nq = nq

        self.cs: tp.Optional[float] = None
        # Todo: fill the missing descriptions
        #: $P_v(q)$
        self.spec_den_v: tp.Optional[np.ndarray] = None
        #: ???
        self.spec_den_gw: tp.Optional[np.ndarray] = None
        #: $\mathcal{P}_{\tilde{v}}(q)$
        self.pow_v: tp.Optional[np.ndarray] = None
        #: ???
        self.pow_gw: tp.Optional[np.ndarray] = None

        if compute:
            self.compute()

    def compute(self):
        if not self.bubble.solved:
            self.bubble.solve()
        self.cs = np.sqrt(self.bubble.model.cs2(self.bubble.va_enthalpy_density, bubble.Phase.BROKEN))

        self.spec_den_v = spec_den_v(
            bub=self.bubble, z=self.z, a=1.,
            nuc_type=self.nuc_type, nt=self.nt, z_st_thresh=self.z_st_thresh, cs=self.cs
        )
        self.pow_v = pow_spec(self.z, spec_den=self.spec_den_v)
        # V2_pow_v = np.trapz(pow_v/self.z, self.z)

        self.spec_den_gw, y = spec_den_gw_scaled(self.z, self.spec_den_v, cs=self.cs)
        self.pow_gw = pow_spec(self.z, spec_den=self.spec_den_gw)
        # gw_power = np.trapz(self.pow_gw/y, y)

    def plot(self, fig: plt.Figure = None, ax: plt.Axes = None, path: str = None) -> "FigAndAxes":
        from pttools.analysis.plot_spectrum import plot_spectrum
        return plot_spectrum(self, fig, ax, path)

    def plot_v(self, fig: plt.Figure = None, ax: plt.Axes = None, path: str = None) -> "FigAndAxes":
        from pttools.analysis.plot_spectrum import plot_spectrum_v
        return plot_spectrum_v(self, fig, ax, path)

    def plot_spec_den_gw(self, fig: plt.Figure = None, ax: plt.Axes = None, path: str = None) -> "FigAndAxes":
        from pttools.analysis.plot_spectrum import plot_spectrum_spec_den_gw
        return plot_spectrum_spec_den_gw(self, fig, ax, path)

    def plot_spec_den_v(self, fig: plt.Figure = None, ax: plt.Axes = None, path: str = None) -> "FigAndAxes":
        from pttools.analysis.plot_spectrum import plot_spectrum_spec_den_v
        return plot_spectrum_spec_den_v(self, fig, ax, path)

    def plot_multi(self, fig: plt.Figure = None, path: str = None) -> plt.Figure:
        from pttools.analysis.plot_spectrum import plot_spectrum_multi
        return plot_spectrum_multi(self, fig, path)


def convert_params(params: bubble.PhysicalParams) -> bubble.PhysicalParams:
    """Convert the physical parameters from a list to a tuple if necessary."""
    if isinstance(params, list):
        logger.warning("Specifying the model parameters as a list is deprecated. Please use a tuple instead.")
        return tuple(params)
    return params


@numba.njit
def nu(T: th.FloatOrArr, nuc_type: NucType = NucType.SIMULTANEOUS, a: float = 1.) -> th.FloatOrArr:
    r"""
    Bubble lifetime distribution function

    :gw_pt_ssm:`\ ` eq. 4.27 & 4.32

    :param T: dimensionless time
    :param nuc_type: nucleation type, simultaneous or exponential
    :return: bubble lifetime distribution $\nu$
    """
    if nuc_type == NucType.SIMULTANEOUS.value:
        return 0.5 * a * (a*T)**2 * np.exp(-(a*T)**3 / 6)
    if nuc_type == NucType.EXPONENTIAL.value:
        return a * np.exp(-a*T)
    # raise ValueError(f"Nucleation type not recognized: \"{nuc_type}\"")
    raise ValueError("Nucleation type not recognized")


# @numba.njit
def parse_params(params: bubble.PhysicalParams) -> tp.Tuple[float, float, NucType, bubble.NucArgs]:
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
        nuc_type = DEFAULT_NUC_TYPE
    if len(params) > 3:
        nuc_args = params[3]
    else:
        nuc_args = const.DEFAULT_NUC_PARM

    return v_wall, alpha, nuc_type, nuc_args


def pow_spec(z: th.FloatOrArr, spec_den: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Power spectrum from spectral density at dimensionless wavenumber z.

    :gw_pt_ssm:`\ ` eq. 4.18, but without the factor of 2.

    :param z: dimensionless wavenumber $z$
    :param spec_den: spectral density
    :return: power spectrum
    """
    return z**3 / (2. * np.pi ** 2) * spec_den


def power_gw_scaled_bag(
        z: np.ndarray,
        params: bubble.PhysicalParams,
        npt: const.NptType = const.NPTDEFAULT,
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

    eps = 1e-8  # Seems to be needed for max(z) <= 100. Why?
    #    nx = len(z) - this can be too few for velocity PS convolutions
    nx = npt[2]
    xmax = max(z) * (0.5 * (1. + const.CS0) / const.CS0) + eps
    xmin = min(z) * (0.5 * (1. - const.CS0) / const.CS0) - eps

    x = np.logspace(np.log10(xmin), np.log10(xmax), nx)

    sd_v = spec_den_v_bag(x, params, npt, filename, skip, method, de_method, z_st_thresh)
    sd_gw, y = spec_den_gw_scaled(x, sd_v, z)
    return pow_spec(z, sd_gw)


def power_v_bag(
        z: np.ndarray,
        params: bubble.PhysicalParams,
        npt: const.NptType = const.NPTDEFAULT,
        filename: str = None,
        skip: int = 1,
        method: ssm.Method = ssm.Method.E_CONSERVING,
        de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
        z_st_thresh: float = const.Z_ST_THRESH) -> np.ndarray:
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

    p_v = spec_den_v_bag(z, params, npt, filename, skip, method, de_method)
    return pow_spec(z, p_v)


@numba.njit(parallel=True)
def _spec_den_gw_scaled_core(
        xlookup: np.ndarray,
        P_vlookup: np.ndarray,
        z: np.ndarray,
        cs: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    r""":gw_pt_ssm:`\ ` eq. 3.47

    Variable z in paper is x here, and y in paper is z here
    """
    cs2 = cs**2
    nx = len(xlookup)
    p_gw = np.zeros_like(z)

    for i in numba.prange(z.size):
        xplus = z[i] / cs * (1. + cs) / 2.
        xminus = z[i] / cs * (1. - cs) / 2.
        # x = np.logspace(np.log10(xminus), np.log10(xplus), nx)
        x = speedup.logspace(np.log10(xminus), np.log10(xplus), nx)
        integrand = \
            (x - xplus) ** 2 * (x - xminus) ** 2 / x / (xplus + xminus - x) \
            * np.interp(x, xlookup, P_vlookup) \
            * np.interp((xplus + xminus - x), xlookup, P_vlookup)
        p_gw_factor = ((1 - cs2) / cs2) ** 2 / (4 * np.pi * z[i] * cs)
        p_gw[i] = p_gw_factor * np.trapz(integrand, x)

    # Here we are using G = 2P_v (v spec den is twice plane wave amplitude spec den).
    # Eq 3.48 in SSM paper gives a factor 3.Gamma^2.P_v.P_v = 3 * (4/3)^2.P_v.P_v
    # Hence overall should use (4/3).G.G
    return (4. / 3.) * p_gw, z


@numba.njit(nogil=True)
def _spec_den_gw_scaled_z(
        xlookup: np.ndarray,
        P_vlookup: np.ndarray,
        z: np.ndarray,
        cs: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    # nx = len(z)
    # nx = len(xlookup)
    # Integration limits
    xlargest = max(z) * 0.5 * (1. + cs) / cs
    xsmallest = min(z) * 0.5 * (1. - cs) / cs

    if max(xlookup) < xlargest or min(xlookup) > xsmallest:
        raise ValueError("Range of xlookup is not large enough.")

    return _spec_den_gw_scaled_core(xlookup, P_vlookup, z, cs)


@numba.njit(nogil=True)
def _spec_den_gw_scaled_no_z(
        xlookup: np.ndarray,
        P_vlookup: np.ndarray,
        z: None,
        cs: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    nx = len(xlookup)
    zmax = max(xlookup) / (0.5 * (1. + cs) / cs)
    zmin = min(xlookup) / (0.5 * (1. - cs) / cs)
    new_z = speedup.logspace(np.log10(zmin), np.log10(zmax), nx)
    return _spec_den_gw_scaled_core(xlookup, P_vlookup, new_z, cs)


@numba.generated_jit(nopython=True, nogil=True)
def spec_den_gw_scaled(
        xlookup: np.ndarray,
        P_vlookup: np.ndarray,
        z: np.ndarray = None,
        cs: float = const.CS0) -> tp.Union[tp.Tuple[np.ndarray, np.ndarray], th.NumbaFunc]:
    r"""
    Spectral density of scaled gravitational wave power at values of kR* given
    by input z array, or at len(xlookup) values of kR* between the min and max
    of xlookup where the GW power can be computed.
    (xlookup, P_vlookup) is used as a lookup table to specify function.
    P_vlookup is the spectral density of the FT of the velocity field,
    not the spectral density of plane wave coeffs, which is lower by a
    factor of 2.

    :return: $\hat{\mathcal{P}}$ Eq. 3.33 of Chloe's thesis, which should be ($3\Gamma \bar{U}_f$) Eq. 3.47
        Eq. 3.46 converted to the spectral density and divided by (H L_f)

    The factor of 3 comes from the Friedmann equation
    3H^2/(8pi G)
    """
    if isinstance(z, numba.types.Array):
        return _spec_den_gw_scaled_z
    if isinstance(z, (numba.types.NoneType, numba.types.Omitted)):
        return _spec_den_gw_scaled_no_z
    if isinstance(z, np.ndarray):
        return _spec_den_gw_scaled_z(xlookup, P_vlookup, z, cs)
    if z is None:
        return _spec_den_gw_scaled_no_z(xlookup, P_vlookup, z, cs)
    raise TypeError(f"Unknown type for z: {type(z)}")


@numba.njit
def _qT_array(qRstar, Ttilde, b_R, vw):
    return qRstar * Ttilde / (b_R * vw)


@numba.njit
def _spec_den_v_core_loop(
        z_i: float, t_array: np.ndarray, b_R: float, vw: float,
        qT_lookup: np.ndarray, A2_lookup: np.ndarray, nuc_type: NucType, a: float, factor: float):
    qT = _qT_array(z_i, t_array, b_R, vw)
    A2_2d_array_z = np.interp(qT, qT_lookup, A2_lookup)
    array2 = t_array ** 6 * nu(t_array, nuc_type, a) * A2_2d_array_z
    D = np.trapz(array2, t_array)
    return D * factor


@numba.njit(parallel=True, nogil=True)
def _spec_den_v_core(
        a: float,
        A2_lookup: np.ndarray,
        log10tmin: float,
        log10tmax: float,
        nuc_type: NucType,
        nt: int,
        qT_lookup: np.ndarray,
        vw: float,
        z: np.ndarray):
    t_array = speedup.logspace(log10tmin, log10tmax, nt)
    b_R = (8. * np.pi) ** (1. / 3.)  # $\beta R_* = b_R v_w $

    # A2_2d_array = np.zeros((nz, nt))

    # array2 = np.zeros(nt)
    sd_v = np.zeros(z.size)  # array for spectral density of v
    factor = 1. / (b_R * vw) ** 6
    factor = 2 * factor  # because spectral density of v is 2 * P_v

    for i in numba.prange(z.size):
        sd_v[i] = _spec_den_v_core_loop(z[i], t_array, b_R, vw, qT_lookup, A2_lookup, nuc_type, a, factor)

    return sd_v


def spec_den_v(
        bub: bubble.Bubble,
        z: np.ndarray,
        a: float,
        nuc_type: NucType,
        nt: int = const.NPTDEFAULT[1],
        z_st_thresh: float = const.Z_ST_THRESH,
        cs: float = None):
    """The full spectral density of the velocity field

    This is twice the spectral density of the plane wave components of the velocity field

    :return: 2 * $P_v(q)$ of eq. 4.17
    """
    # z limits
    log10zmin = np.log10(np.min(z))
    log10zmax = np.log10(np.max(z))
    dlog10z = (log10zmax - log10zmin) / z.size

    # t limits
    tmin = const.T_TILDE_MIN
    tmax = const.T_TILDE_MAX
    log10tmin = np.log10(tmin)
    log10tmax = np.log10(tmax)

    qT_lookup = 10 ** np.arange(log10zmin + log10tmin, log10zmax + log10tmax, dlog10z)
    A2_lookup = ssm.a2_e_conserving(bub=bub, z=qT_lookup, z_st_thresh=z_st_thresh, cs=cs)[0]
    # if qT_lookup.size != A2_lookup.size:
    #     raise ValueError(f"Lookup sizes don't match: {qT_lookup.size} != {A2_lookup.size}")

    return _spec_den_v_core(
        a=a,
        A2_lookup=A2_lookup,
        log10tmin=log10tmin,
        log10tmax=log10tmax,
        nt=nt,
        nuc_type=nuc_type,
        qT_lookup=qT_lookup,
        vw=bub.v_wall,
        z=z
    )


def spec_den_v_bag(
        z: np.ndarray,
        params: bubble.PhysicalParams,
        npt: const.NptType = const.NPTDEFAULT,
        filename: str = None,
        skip: int = 1,
        method: ssm.Method = ssm.Method.E_CONSERVING,
        de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
        z_st_thresh=const.Z_ST_THRESH):
    r"""
    Get dimensionless velocity spectral density $\bar{P}_v$.

    Gets fluid velocity profile from bubble toolbox or from file if specified.
    Convolves 1-bubble Fourier transform $|A(q T)|^2$ with bubble wall
    lifetime distribution $\nu(T \beta)$ specified by "nuc_type" and "nuc_args".

    :param z: array $z = qR_*$
    :param params: tuple of vw (scalar), alpha (scalar), nuc_type (string [exponential* | simultaneous]), nuc_args (tuple, default (1,))
    :param npt: number of points
    :return: dimensionless velocity spectral density $\bar{P}_v$
    """
    params = convert_params(params)
    bubble.check_physical_params(params)

    nz = z.size
    # nxi = npt[0]
    nt = npt[1]
    # nq = npt[2]

    # z limits
    log10zmin = np.log10(min(z))
    log10zmax = np.log10(max(z))
    dlog10z = (log10zmax - log10zmin) / nz

    # t limits
    tmin = const.T_TILDE_MIN
    tmax = const.T_TILDE_MAX
    log10tmin = np.log10(tmin)
    log10tmax = np.log10(tmax)

    qT_lookup = 10 ** np.arange(log10zmin + log10tmin, log10zmax + log10tmax, dlog10z)

    vw, alpha, nuc_type, nuc_args = parse_params(params)
    if filename is None:
        A2_lookup = ssm.a2_ssm_func_bag(
            z=qT_lookup, v_wall=vw, alpha=alpha,
            npt=npt, method=method, de_method=de_method, z_st_thresh=z_st_thresh
        )
    else:
        A2_lookup = ssm.a2_e_conserving_bag_file(
            z=qT_lookup, filename=filename, alpha=alpha,
            skip=skip, npt=npt, z_st_thresh=z_st_thresh
        )

    # if qT_lookup.size != A2_lookup.size:
    #     raise ValueError(f"Lookup sizes don't match: {qT_lookup.size} != {A2_lookup.size}")

    return _spec_den_v_core(
        a=nuc_args[0],
        A2_lookup=A2_lookup,
        log10tmin=log10tmin,
        log10tmax=log10tmax,
        nt=nt,
        nuc_type=nuc_type,
        qT_lookup=qT_lookup,
        vw=vw,
        z=z
    )
