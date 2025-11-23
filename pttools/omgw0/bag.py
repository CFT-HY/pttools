r"""$\Omega_{\text{gw},0}$ for the bag model"""

import numpy as np

import pttools.bubble.ke_frac_approx as K
from pttools.omgw0 import const
from pttools.omgw0.factors import J
from pttools.omgw0.freq import f0
import pttools.omgw0.suppression as sup_mod
from pttools.ssm import NptType, NPTDEFAULT
from pttools import ssm


def omgw0_bag(
        freqs: np.ndarray,
        vw: float,
        alpha: float,
        r_star: float,
        T: float = const.T_DEFAULT,
        npt: NptType = NPTDEFAULT,
        sup: sup_mod.Suppression = sup_mod.DEFAULT,
        sup_method: sup_mod.SuppressionMethod = sup_mod.SuppressionMethod.DEFAULT):
    r"""
    For given set of thermodynamic parameters vw, alpha, rs and Tn calculates the power spectrum using
    the SSM as encoded in the PTtools module (omgwi)
    :gowling_2021:`\ ` eq. 2.14
    """
    params = (vw, alpha, ssm.NucType.EXPONENTIAL, (1,))
    fp0 = f0(r_star, T)
    z = freqs/fp0

    K_frac = K.calc_ke_frac(vw, alpha)
    omgwi = ssm.power_gw_scaled_bag(z, params, npt=npt)

    # entry options for power_gw_scaled
    #          z: np.ndarray,
    #        params: bubble.PHYSICAL_PARAMS_TYPE,
    #        npt=const.NPTDEFAULT,
    #        filename: str = None,
    #        skip: int = 1,
    #        method: ssm.Method = ssm.Method.E_CONSERVING,
    #        de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
    #        z_st_thresh: float = const.Z_ST_THRESH)

    if sup_method == sup_mod.SuppressionMethod.NONE:
        return const.FGW0 * J(r_star, K_frac) * omgwi
    if sup_method == sup_mod.SuppressionMethod.NO_EXT:
        sup_fac = sup.suppression(vw, alpha, method=sup_method)
        return const.FGW0 * J(r_star, K_frac) * omgwi * sup_fac
    if sup_method == sup_mod.SuppressionMethod.EXT_CONSTANT:
        sup_fac = sup.suppression(vw, alpha, method=sup_method)
        return const.FGW0 * J(r_star, K_frac) * omgwi * sup_fac
    raise ValueError(f"Invalid suppression method: {sup_method}")
