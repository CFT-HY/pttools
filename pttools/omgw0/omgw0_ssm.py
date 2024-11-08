r"""
Calculate the physical gravitational wave power spectrum $\Omega_{\rm gw}(f)$
as a function of physical frequency $f$ in the Sound shell model.

Created on 10/11/21

@author: chloeg, markh
"""

import numpy as np

import pttools.omgw0.ke_frac_approx as K
import pttools.omgw0.suppression as sup
from pttools.ssmtools.const import NPTDEFAULT, NptType
import pttools.ssmtools.spectrum as ssm
import pttools.type_hints as th
from . import const


################################
# SGWB as calculated by PTtools (SSM)
################################

def f(z: th.FloatOrArr, r_star: th.FloatOrArr, f_star0: th.FloatOrArr) -> th.FloatOrArr:
    r"""Convert the dimensionless wavenumber $z$ to frequency today by taking into account the redshift.
    $$f = \frac{z}{r_*} f_{*,0}$$,
    :gowling_2021:`\ ` eq. 2.12
    :param z: dimensionless wavenumber $z$
    :param r_star: Hubble-scaled mean bubble spacing
    """
    return z/r_star * f_star0


def f0(rs: th.FloatOrArr, T_n: th.FloatOrArr = const.T_default, g_star: float = 100) -> th.FloatOrArr:
    r"""Factor required to take into account the redshift of the frequency scale"""
    return f_star0(T_n, g_star) / rs


def f_star0(T_n: th.FloatOrArr, g_star: th.FloatOrArr = 100) -> th.FloatOrArr:
    r"""
    Conversion factor between the frequencies at the time of the nucleation and frequencies today.
    $$f_{*,0} = 2.6 \cdot 10^{-6} \text{Hz} \left( \frac{T_n}{100 \text{GeV}} \right) \left( \frac{g_*}{100} \right)^{\frac{1}{6}}$$,
    :gowling_2021:`\ ` eq. 2.13
    :param T_n: Nucleation temperature
    :param g_star: Degrees of freedom at the time the GWs were produced. The default value is from the article.
    :return:
    """
    return const.fs0_ref * (T_n / 100) * (g_star / 100)**(1/6)


def F_gw0(
        g_star: th.FloatOrArr,
        g0: th.FloatOrArr = const.G0,
        gs0: th.FloatOrArr = const.GS0,
        gs_star: th.FloatOrArr = None,
        om_gamma0: th.FloatOrArr = const.OMEGA_RADIATION) -> th.FloatOrArr:
    r"""Power attenuation following the end of the radiation era
    $$F_{\text{gw},0} = \Omega_{\gamma,0} \left( \frac{g_{s0}}{g_{s*}} \right)^{4/9} \frac{g_*}{g_0}
    = (3.57 \pm 0.05) \cdot 10^{-5} \left( \frac{100}{g_*} \right)^{1/3}$$
    :gowling_2021:`\ ` eq. 2.11
    """
    if gs0 is None or gs_star is None or g0 is None or om_gamma0 is None:
        return 3.57e-5 * (100/g_star)**(1/3)
    return om_gamma0 * (gs0 / gs_star)**(4/9) * g_star / g0


def J(r_star: th.FloatOrArr, K_frac: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Pre-factor to convert power_gw_scaled to predicted spectrum
    approximation of $(H_n R_*)(H_n \tau_v)$
    updating to properly convert from flow time to source time

    $$J = H_n R_* H_n \tau_v = r_* \left(1 - \frac{1}{\sqrt{1 + 2x}}$$
    :gowling_2021:`\ ` eq. 2.8
    """
    sqrt_K = np.sqrt(K_frac)
    return r_star * (1 - 1/(np.sqrt(1 + 2*r_star/sqrt_K)))


def omgw0_bag(
        freqs: np.ndarray,
        vw: float,
        alpha: float,
        r_star: float,
        T: float = const.T_default,
        npt: NptType = NPTDEFAULT,
        suppression: sup.SuppressionMethod = const.SUP_METHOD_DEFAULT):
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

    if suppression == sup.SuppressionMethod.NONE:
        return const.Fgw0 * J(r_star, K_frac) * omgwi
    elif suppression == sup.SuppressionMethod.NO_EXT:
        sup_fac = sup.get_suppression_factor(vw, alpha, method=suppression)
        return const.Fgw0 * J(r_star, K_frac) * omgwi * sup_fac
    elif suppression == sup.SuppressionMethod.EXT_CONSTANT:
        sup_fac = sup.get_suppression_factor(vw, alpha, method=suppression)
        return const.Fgw0 * J(r_star, K_frac) * omgwi * sup_fac
    raise ValueError(f"Invalid suppression: {suppression}")


def r_star(H_n: th.FloatOrArr, R_star: th.FloatOrArr) -> th.FloatOrArr:
    """
    $$r_* = H_n R_*$$
    :gowling_2021:`\ ` eq. 2.2
    """
    return H_n * R_star
