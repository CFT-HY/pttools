r"""
Calculate the physical gravitational wave power spectrum $\Omega_{\rm gw}(f)$
as a function of physical frequency $f$ in the Sound shell model.

Created on 10/11/21

@author: chloeg, markh
"""

import functools
import logging

import numpy as np

from pttools.bubble.boundary import Phase
from pttools.bubble.bubble import Bubble
import pttools.bubble.ke_frac_approx as K
import pttools.omgw0.suppression as sup
from pttools.ssmtools.const import NPTDEFAULT, NTDEFAULT, N_Z_LOOKUP_DEFAULT, NptType
import pttools.ssmtools as ssm
import pttools.type_hints as th
from . import const
from . import noise

logger = logging.getLogger(__name__)


class Spectrum(ssm.SSMSpectrum):
    r"""A spectrum object that includes $\Omega_{\text{gw},0}$"""
    def __init__(
            self,
            bubble: Bubble,
            y: np.ndarray = None,
            z_st_thresh: float = ssm.Z_ST_THRESH,
            nuc_type: ssm.NucType = ssm.DEFAULT_NUC_TYPE,
            nt: int = NTDEFAULT,
            n_z_lookup: int = N_Z_LOOKUP_DEFAULT,
            r_star: float = None,
            lifetime_multiplier: float = 1,
            compute: bool = True,
            Tn: float = None,
            g_star: float = None,
            gs_star: float = None
            ):
        """
        :param bubble: the Bubble object
        :param y: $z = kR*$ array
        :param z_st_thresh: for $z$ values above z_sh_tresh, use approximation rather than doing the sine transform integral.
        :param nuc_type: nucleation type
        :param nt: number of points in the t array
        :param r_star: $r_*$
        :param lifetime_multiplier: used for computing the source lifetime factor
        :param compute: whether to compute the spectrum immediately
        :param Tn: $T_n$, nucleation temperature override
        :param g_star: $g_*$, degrees of freedom override at the time of GW production
        :param gs_star: $g_{s,*}$ degrees of freedom override for entropy at the time of GW production
        """
        super().__init__(
            bubble=bubble,
            y=y,
            z_st_thresh=z_st_thresh,
            nuc_type=nuc_type,
            nt=nt,
            n_z_lookup=n_z_lookup,
            r_star=r_star,
            lifetime_multiplier=lifetime_multiplier,
            compute=compute
        )
        self.override_necessary = not self.bubble.model.temperature_is_physical
        self.Tn_manual_override = Tn is not None
        self.g_star_manual_override = g_star is not None
        self.gs_star_manual_override = gs_star is not None
        self.Tn_override = const.T_default if Tn is None else Tn
        self.g_star_override = const.G_STAR_DEFAULT if g_star is None else gs_star
        self.gs_star_override = self.g_star_override if gs_star is None else gs_star

    def f(self, z: np.ndarray = None) -> th.FloatOrArr:
        if z is None:
            z = self.y
        return f(z=z, r_star=self.r_star, f_star0=self.f_star0)

    @functools.cached_property
    def f_star0(self) -> float:
        return f_star0(
            Tn=self.Tn,
            g_star=self.g_star
        )

    def F_gw0(self, g0: float = const.G0, gs0: float = const.GS0) -> float:
        return F_gw0(
            g_star=self.g_star,
            g0=g0,
            gs0=gs0,
            gs_star=self.gs_star
        )

    @functools.cached_property
    def g_star(self) -> float:
        if self.override_necessary or self.gs_star_manual_override:
            return self.g_star_override
        return self.g_star_computed

    @functools.cached_property
    def g_star_computed(self):
        return self.bubble.model.gp(w=self.bubble.va_enthalpy_density, phase=Phase.BROKEN)

    @functools.cached_property
    def gs_star(self) -> float:
        if self.override_necessary or self.gs_star_manual_override:
            return self.gs_star_override
        return self.gs_star_computed

    @functools.cached_property
    def gs_star_computed(self) -> float:
        return self.bubble.model.gs(w=self.bubble.va_enthalpy_density, phase=Phase.BROKEN)

    def noise(self) -> np.ndarray:
        return noise.omega_noise(self.f())

    def noise_ins(self) -> np.ndarray:
        return noise.omega_ins(self.f())

    def omgw0(
            self,
            g0: float = const.G0,
            gs0: float = const.GS0,
            suppression: sup.SuppressionMethod = sup.SuppressionMethod.DEFAULT) -> np.ndarray:
        # The r_star compensates the fact that the pow_gw includes a correction factor that is J without r_star
        return self.r_star * self.F_gw0(g0=g0, gs0=gs0) * self.pow_gw * self.suppression_factor(method=suppression)

    def omgw0_peak(
            self,
            g0: float = const.G0,
            gs0: float = const.GS0,
            suppression: sup.SuppressionMethod = sup.SuppressionMethod.DEFAULT):
        omgw0 = self.omgw0(g0=g0, gs0=gs0, suppression=suppression)
        i_max = np.argmax(omgw0)
        return self.f()[i_max], omgw0[i_max]

    def signal_to_noise_ratio(self) -> float:
        return noise.signal_to_noise_ratio(f=self.f(), signal=self.omgw0(), noise=self.noise())

    def signal_to_noise_ratio_instrument(self) -> float:
        return noise.signal_to_noise_ratio(f=self.f(), signal=self.omgw0(), noise=self.noise_ins())

    def suppression_factor(self, method: sup.SuppressionMethod = sup.SuppressionMethod.DEFAULT) -> float:
        return sup.get_suppression_factor(vw=self.bubble.v_wall, alpha=self.bubble.alpha_n, method=method)

    @functools.cached_property
    def Tn(self) -> float:
        if self.override_necessary or self.Tn_manual_override:
            return self.Tn_override
        return self.bubble.Tn


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


def f_star0(Tn: th.FloatOrArr, g_star: th.FloatOrArr = 100) -> th.FloatOrArr:
    r"""
    Conversion factor between the frequencies at the time of the nucleation and frequencies today.
    $$f_{*,0} = 2.6 \cdot 10^{-6} \text{Hz} \left( \frac{T_n}{100 \text{GeV}} \right) \left( \frac{g_*}{100} \right)^{\frac{1}{6}}$$,
    :gowling_2021:`\ ` eq. 2.13
    :param Tn: Nucleation temperature
    :param g_star: Degrees of freedom at the time the GWs were produced. The default value is from the article.
    :return:
    """
    return const.fs0_ref * (Tn / 100) * (g_star / 100)**(1 / 6)


def F_gw0(
        g_star: th.FloatOrArr,
        g0: th.FloatOrArr = const.G0,
        gs0: th.FloatOrArr = const.GS0,
        gs_star: th.FloatOrArr = None,
        om_gamma0: th.FloatOrArr = const.OMEGA_RADIATION) -> th.FloatOrArr:
    r"""Power attenuation following the end of the radiation era
    $$F_{\text{gw},0} = \Omega_{\gamma,0} \left( \frac{g_{s0}}{g_{s*}} \right)^{4/9} \frac{g_*}{g_0}
    = (3.57 \pm 0.05) \cdot 10^{-5} \left( \frac{100}{g_*} \right)^{1/3}$$
    There is a typo in :gowling_2021:`\ ` eq. 2.11: the $\frac{4}{9}$ should be $\frac{4}{3}$.
    """
    if g0 is None or gs0 is None or gs_star is None or om_gamma0 is None:
        return 3.57e-5 * (100/g_star)**(1/3)
    return om_gamma0 * (gs0 / gs_star)**(4/3) * g_star / g0


def J(r_star: th.FloatOrArr, K_frac: th.FloatOrArr, nu: float = 0) -> th.FloatOrArr:
    r"""
    Pre-factor to convert power_gw_scaled to predicted spectrum
    approximation of $(H_n R_*)(H_n \tau_v)$
    updating to properly convert from flow time to source time

    $$J = H_n R_* H_n \tau_v = r_* \left(1 - \frac{1}{\sqrt{1 + 2x}}$$
    :gowling_2021:`\ ` eq. 2.8
    """
    sqrt_K = np.sqrt(K_frac)
    return r_star * (1 - (np.sqrt(1 + 2*r_star/sqrt_K)**(-1-2*nu)))


def omgw0_bag(
        freqs: np.ndarray,
        vw: float,
        alpha: float,
        r_star: float,
        T: float = const.T_default,
        npt: NptType = NPTDEFAULT,
        suppression: sup.SuppressionMethod = sup.SuppressionMethod.DEFAULT):
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
    r"""
    $$r_* = H_n R_*$$
    :gowling_2021:`\ ` eq. 2.2
    """
    return H_n * R_star
