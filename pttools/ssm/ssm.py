"""Sound Shell Model functions"""

import enum
import logging

import numpy as np

from pttools.bubble.bubble import Bubble
from pttools import speedup
from pttools.ssm import const
from pttools.ssm.calculators import resample_uniform_xi
from pttools.ssm.sin_transform import sin_transform

logger = logging.getLogger(__name__)


@enum.unique
class DE_Method(str, enum.Enum):
    ALTERNATE = "alternate"
    STANDARD = "standard"


@enum.unique
class Method(str, enum.Enum):
    E_CONSERVING = "e_conserving"
    F_ONLY = "f_only"
    WITH_G = "with_g"


def a2_e_conserving(
        bub: Bubble,
        z: np.ndarray,
        cs: float,
        z_st_thresh: float = const.Z_ST_THRESH,
        nxi: int = const.NPTDEFAULT[0],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Returns the value of $|A(z)|^2$, where
    $|\text{Plane wave amplitude}|^2 = T^3 | A(z)|^2$.

    :param z: array of scaled wavenumbers $z = kR_*$.
    :return: $|A(z)|^2$, fp2_2, lam2
    """
    if not bub.solved:
        bub.solve()
    v_ip, w_ip, xi = bub.v, bub.w, bub.xi

    # :gw_pt_ssm:`\ ` eq. 4.5
    f = (4. * np.pi / z) * sin_transform(z, xi, v_ip, z_st_thresh, v_wall=bub.v_wall, v_sh=bub.v_sh)

    v_ft = speedup.gradient(f) / speedup.gradient(z)

    # This corresponds to de_from_w_bag
    e = bub.model.e(bub.w, bub.phase)
    lam_orig = (e - e[-1]) / w_ip[-1]

    lam_orig += w_ip * v_ip * v_ip / w_ip[-1]  # This doesn't make much difference at small alpha

    xi_re, lam_re = resample_uniform_xi(xi, lam_orig, nxi)

    # lam_re = np.interp(xi_re,xi,lam_orig)
    # lam_ft = np.zeros_like(z)
    # for j in range(lam_ft.size):
    #     # Need to fix problem with ST of lam for detonations
    #     lam_ft[j] = (4.*np.pi/z[j]) * \
    #         calculators.sin_transform(z[j], xi_re, xi_re*lam_re, z_st_thresh=max(z))

    # :gw_pt_ssm:`\ ` eq. 4.8
    lam_ft = (4. * np.pi / z) * sin_transform(
        z, xi_re, xi_re * lam_re, z_st_thresh, v_wall=bub.v_wall, v_sh=bub.v_sh)

    # :gw_pt_ssm:`\ ` eq. 4.11

    A2 = 0.25 * (v_ft ** 2 + (cs * lam_ft) ** 2)

    return A2, v_ft ** 2 / 2, (cs * lam_ft) ** 2 / 2
