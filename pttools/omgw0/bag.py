r"""$\Omega_{\text{gw},0}$ for the bag model"""

from pttools.bubble.energy_budget import kinetic_energy_fraction_approx
from pttools.omgw0 import const
from pttools.omgw0.freq import f0
import pttools.omgw0.suppression as sup_mod
from pttools.ssm import DEFAULT_N_PT, NptType, NucType, \
    H_star_tau_sh_approx, H_star_tau_v_old, J as J_func, power_gw_bag
import pttools.type_hints as th


def omgw0_bag(
        freqs: th.FloatArr1D,
        vw: float,
        alpha: float,
        r_star: float,
        T_star: float = const.DEFAULT_T_STAR,
        npt: NptType = DEFAULT_N_PT,
        sup: sup_mod.Suppression = sup_mod.DEFAULT,
        sup_method: sup_mod.SuppressionMethod = sup_mod.SuppressionMethod.DEFAULT,
        parallel: bool = True) -> th.FloatArr1D:
    r"""
    For given set of thermodynamic parameters vw, alpha, rs and T_star calculates the power spectrum using
    the SSM as encoded in the PTtools module (omgwi)
    :gowling_2021:`\ ` eq. 2.14
    """
    params = (vw, alpha, NucType.EXPONENTIAL, (1,))
    fp0 = f0(r_star, T_star)
    z = freqs / fp0

    K = kinetic_energy_fraction_approx(vw, alpha)
    omgwi = power_gw_bag(z, params, npt=npt, parallel=parallel)

    # entry options for power_gw_scaled
    #          z: th.FloatArr1D,
    #        params: bubble.PHYSICAL_PARAMS_TYPE,
    #        npt=const.NPTDEFAULT,
    #        filename: str = None,
    #        skip: int = 1,
    #        method: ssm.Method = ssm.Method.E_CONSERVING,
    #        de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
    #        z_st_thresh: float = const.Z_ST_THRESH)

    J = J_func(r_star=r_star, H_star_tau_v=H_star_tau_v_old(H_star_tau_sh=H_star_tau_sh_approx(r_star=r_star, K=K)))
    if sup_method == sup_mod.SuppressionMethod.NONE:
        return const.F_GW0 * J * omgwi
    if sup_method == sup_mod.SuppressionMethod.NO_EXT:
        sup_fac = sup.suppression(vw, alpha, method=sup_method)
        return const.F_GW0 * J * omgwi * sup_fac
    if sup_method == sup_mod.SuppressionMethod.EXT_CONSTANT:
        sup_fac = sup.suppression(vw, alpha, method=sup_method)
        return const.F_GW0 * J * omgwi * sup_fac
    raise ValueError(f"Invalid suppression method: {sup_method}")
