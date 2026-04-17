"""Sound Shell Model functions"""

import enum
import logging

import numba
import numpy as np

from pttools import speedup
from pttools.ssm import const
from pttools.ssm.calculators import resample_uniform_xi
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
def a2_e_conserving(
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

    :param z: array of scaled wavenumbers $z = kR_*$.
    :param lambda_correction: whether to enable a non-linear correction for $\lambda$
    :return: $|A(z)|^2$, fp2_2, lam2
    """
    f_val = f(z=z, xi=xi, v=v, v_wall=v_wall, v_sh=v_sh, z_st_thresh=z_st_thresh, parallel=parallel)

    v_ft = speedup.gradient(f_val) / speedup.gradient(z)

    lm = lam(v=v, w=w, e=e, non_linear_correction=lambda_correction)
    lam_ft = l(z=z, xi=xi, lam=lm, v_wall=v_wall, v_sh=v_sh, z_st_thresh=z_st_thresh, n_xi=n_xi, parallel=parallel)

    return a2_fp_csl(fp=v_ft, cs=cs, l=lam_ft), v_ft ** 2 / 2, (cs * lam_ft) ** 2 / 2


@numba.njit
def a2_fp_csl(fp: th.FloatArr, cs: float, l: th.FloatArr) -> th.FloatArr:
    r"""$|A(z)|^2$
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


@numba.njit
def lam(v: th.FloatArr, w: th.FloatArr, e: th.FloatArr, non_linear_correction: bool = False) -> th.FloatArr:
    r"""Energy fluctuation variable $\lambda(x)$
    $$\lambda(x) = \frac{e(x) - \bar{e}}{\bar{w}}$$
    :gw_pt_ssm:`\ ` eq. 3.20

    :param v: $v$
    :param w: $w$
    :param e: $e$
    :param non_linear_correction:
        Enable a non-linear correction.
        This is a remnant from Mark's old experiments and doesn't make much difference at small $\alpha_n$.
    """
    # This corresponds to de_from_w_bag
    # TODO: Is w[-1]=wn the same as \bar{w}?
    lm = (e - e[-1]) / w[-1]
    if non_linear_correction:
        lm += w * v * v / w[-1]
    return lm
