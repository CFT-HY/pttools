"""Sound Shell Model functions"""

import enum
import logging

import numba
import numpy as np

from pttools import speedup
from pttools.ssm import const, lifetime_distribution_momentum, NucType
from pttools.ssm.calculators import resample_uniform_xi
from pttools.ssm.nucleation import beta_R_star0, lifetime_distribution
from pttools.ssm.sin_transform import sin_transform
import pttools.type_hints as th

logger = logging.getLogger(__name__)


@enum.unique
class DE_Method(str, enum.Enum):
    r"""Method for computing $|A(z)|^2$"""
    # TODO: Improve this docstring
    ALTERNATE = "alternate"
    STANDARD = "standard"


@enum.unique
class Method(str, enum.Enum):
    r"""Method for computing $|A(z)|^2$"""
    # TODO: Improve this docstring
    E_CONSERVING = "e_conserving"
    F_ONLY = "f_only"
    WITH_G = "with_g"


@numba.njit(nogil=True)
def A2_e_conserving(
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        e: th.FloatArr1D,
        z: th.FloatArr1D,
        v_wall: float,
        v_sh: float,
        cs: float,
        z_st_thresh: float = const.Z_ST_THRESH,
        n_xi: int = const.DEFAULT_N_XI_SSM,
        parallel: bool = True,
        lambda_correction: bool = False) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]:
    r"""
    Returns the value of $|A(z)|^2$, where
    $|\text{Plane wave amplitude}|^2 = T^3 | A(z)|^2$.

    :param v: velocity profile $v$
    :param w: enthalpy profile $w$
    :param xi: $\xi$ of the profiles
    :param e: energy density profile $e$
    :param v_wall: wall speed $v_\text{wall}$
    :param v_sh: shock speed $v_\text{sh}$
    :param cs: speed of sound $c_s$
    :param z: array of scaled wavenumbers $z = kR_*$.
    :param z_st_thresh: limit above which to use approximate sin transform
    :param n_xi: Number of $\xi$ points for uniform resampling
    :param parallel: Whether to use multiple threads
    :param lambda_correction: whether to enable a non-linear correction for $\lambda$
    :return: $|A(z)|^2, \frac{1}{2} f'(z)^2, \frac{1}{2}(c_s l(z))^2$ (same size as $z$)
    """
    #: $f(z)$
    f_val = f(z=z, xi=xi, v=v, v_wall=v_wall, v_sh=v_sh, z_st_thresh=z_st_thresh, parallel=parallel)
    #: $f'(z)$
    fp_val = speedup.gradient(f_val) / speedup.gradient(z)
    #: $\lambda(z)$
    lm = lam(v=v, w=w, e=e, non_linear_correction=lambda_correction)
    #: $l(z)$
    l_val = l(z=z, xi=xi, lam=lm, v_wall=v_wall, v_sh=v_sh, z_st_thresh=z_st_thresh, n_xi=n_xi, parallel=parallel)

    return A2_fp_csl(fp=fp_val, cs=cs, l=l_val), 0.5 * fp_val ** 2, 0.5 * (cs * l_val) ** 2


@numba.njit(cache=True)
def A2_fp_csl(fp: th.FloatArr, cs: float, l: th.FloatArr) -> th.FloatArr:
    r"""$|A(z)|^2$ from $f'(z)$ and $c_s l(z)$
    $$|A(z)|^2 = = \frac{1}{4} \left[ (f'(z))^2 + (c_s l(z))^2 \right]$$
    :gw_pt_ssm:`\ ` eq. 4.11
    This contains information about the shape of the fluid shells.
    """
    return 0.25 * (fp ** 2 + (cs * l) ** 2)


@numba.njit
def f(
        z: th.FloatArr,
        xi: th.FloatArr,
        v: th.FloatArr,
        v_wall: float,
        v_sh: float,
        z_st_thresh: float = const.Z_ST_THRESH,
        parallel: bool = True) -> th.FloatArr:
    r"""$f(z)$
    $$f(z) = \frac{4\pi}{z} \int_0^\infty d\xi v_\text{ip}(\xi) \sin(z\xi)$$
    :gw_pt_ssm:`\ ` eq. 4.5
    """
    return 4. * np.pi / z * sin_transform(
        z=z, xi=xi, f=v, z_st_thresh=z_st_thresh, v_wall=v_wall, v_sh=v_sh, parallel=parallel
    )


@numba.njit(nogil=True)
def l(
        z: th.FloatArr,
        xi: th.FloatArr,
        lam: th.FloatArr,
        v_wall: float,
        v_sh: float,
        z_st_thresh: float = const.Z_ST_THRESH,
        n_xi: int = const.DEFAULT_N_XI_SSM,
        parallel: bool = True) -> th.FloatArr:
    r"""$l(z)$
    $$l(z) = \frac{4\pi}{z} \int_0^\infty d\xi \lambda_\text{ip}(\xi) \xi \sin(z\xi)$$
    :gw_pt_ssm:`\ ` eq. 4.8
    """
    xi_re, lam_re = resample_uniform_xi(xi, lam, n_xi)

    # Some old implementation
    # lam_re = np.interp(xi_re,xi,lam_orig)
    # lam_ft = np.zeros_like(z)
    # for j in range(lam_ft.size):
    #     # Need to fix problem with ST of lam for detonations
    #     lam_ft[j] = (4.*np.pi/z[j]) * \
    #         calculators.sin_transform(z[j], xi_re, xi_re*lam_re, z_st_thresh=max(z))

    return 4. * np.pi / z * sin_transform(
        z=z, xi=xi_re, f=xi_re * lam_re, z_st_thresh=z_st_thresh, v_wall=v_wall, v_sh=v_sh, parallel=parallel
    )


@numba.njit(cache=True)
def lam(
        v: th.FloatArr,
        w: th.FloatArr,
        e: th.FloatArr,
        e_bar: float | None = None,
        w_bar: float | None = None,
        non_linear_correction: bool = False) -> th.FloatArr:
    r"""Energy fluctuation variable $\lambda(x)$
    $$\lambda(x) = \frac{e(x) - \bar{e}}{\bar{w}}$$
    :gw_pt_ssm:`\ ` eq. 3.20

    :param v: $v$
    :param w: $w$
    :param e: $e$
    :param e_bar: Mean energy density $\bar{e}$. If not given, assumed to be $\bar{e} = e_n$.
    :param w_bar: Mean enthalpy density $\bar{w}$. If not given, assumed to be $\bar{w} = w_n$.
    :param non_linear_correction:
        Enable a non-linear correction.
        This is a remnant from Mark's old experiments and doesn't make much difference at small $\alpha_n$.
    """
    if e_bar is None:
        e_bar = e[-1]
    if w_bar is None:
        w_bar = w[-1]
    # A similar step is done in de_from_w_bag()
    lm = (e - e_bar) / w_bar
    if non_linear_correction:
        lm += w * v * v / w_bar
    return lm


@numba.njit
def qT_lookup(T_tilde: th.FloatArr1D, z: th.FloatArr1D) -> th.FloatArr1D:
    """$z=qT$ lookup
    This is used by
    :py:func:pttools.ssm.spec_den_v: and
    :py:func:pttools.ssm.A2_e_conserving:.
    """
    # z limits
    log10_z_min = np.log10(np.min(z))
    log10_z_max = np.log10(np.max(z))

    # try:
    # Todo: Check whether this could be replaced by logspace
    return 10 ** np.arange(
        log10_z_min + np.log10(T_tilde.min()),
        log10_z_max + np.log10(T_tilde.max()),
        step=(log10_z_max - log10_z_min) / z.size
    )
    # except ValueError as e:
    #     logger.error(
    #         "Could not compute qT_lookup with log10_z_min=%s, log10_T_min=%s, log10_z_max=%s, log10_T_max=%s, dlog10z=%s",
    #         log10_z_min, log10_T_min, log10_z_max, log10_T_max, dlog10z
    #     )
    #     raise e


@numba.njit
def T_tilde(T_tilde_min: float, T_tilde_max: float, n: int):
    r"""Generate $\tilde{T}$ array"""
    return speedup.logspace(np.log10(T_tilde_min), np.log10(T_tilde_max), n)


@numba.njit
def ubarf2_from_a2(
        T_tilde: th.FloatArr1D,
        z: th.FloatArr1D,
        A2: th.FloatArr1D,
        v_wall: float,
        nuc_type: NucType,
        bubble_spacing_enlargement_factor: float = 1.) -> float:
    r"""Mean square fluid velocity $\bar{U}_f^2 \left( |A(z)|^2 \right)$
    $$\bar{U}_f^2
    = \int \frac{dq}{q} \mathcal{P}_\tilde{v}(a)
    = \frac{2}{(\beta R_*)^3}
    \int d\tilde{T} \nu(\tilde{T}) \tilde{T}^3
    \int dz \frac{z^2}{2\pi^2} |A(z)|^2$$
    :gw_pt_ssm:`\ ` eq. 4.33
    This version takes into account the nucleation history, unlike
    :py:func:pttools.bubble.thermo.ubarf2:.

    The use of $\Lambda_{\text{nuc}}$ needs to be kept consistent with
    :py:func:pttools.ssm.spec_den_v.spec_den_v:.
    Please note that eq. 4.34 assumes that $R_* = R_{*,0}$,
    which is why it's not used here.
    """
    if z.shape != A2.shape:
        raise TypeError(
            "z and A2 must be of the same shape. "
            f"Got z.shape={z.shape}, A2.shape={A2.shape}"
        )

    nu3 = 6. \
        if nuc_type in (NucType.EXPONENTIAL, NucType.SIMULTANEOUS) \
        else lifetime_distribution_momentum(nu=lifetime_distribution(T_tilde, nuc_type), T_tilde=T_tilde, n=3)
    beta_R = beta_R_star0(v_wall) / bubble_spacing_enlargement_factor
    return 2 / (beta_R**3 * 2 * np.pi**2) * nu3 * np.trapezoid(z ** 2 * A2, z)
