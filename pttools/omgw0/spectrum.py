import functools
import math
import typing as tp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from pttools.bubble import Bubble
from pttools.omgw0 import const
from pttools.omgw0.factors import F_gw0
from pttools.omgw0 import freq
from pttools.omgw0 import noise
from pttools import ssm
from pttools.ssm.const import DEFAULT_Y
from pttools.ssm.suppression import DEFAULT_SUPPRESSION, Suppression, SuppressionMethod
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr
from pttools.utils import copy_docstrings, export_json

if tp.TYPE_CHECKING:
    from pttools.analysis.utils import FigAndAxes


class Spectrum(ssm.SSMSpectrum):
    r"""A spectrum object that includes the conversion to the GW power spectrum today $\Omega_{\text{gw},0}$"""
    def __init__(
            self,
            bubble: Bubble,
            # Input parameters
            beta_tilde: float | None = None,
            r_star: float | None = None,
            y: th.FloatArr1D = DEFAULT_Y,
            a_star_a_r_ratio: float = ssm.DEFAULT_A_STAR_A_R_RATIO,
            N_sh: float = ssm.DEFAULT_N_SH,
            nuc_type: ssm.NucType = ssm.DEFAULT_NUC_TYPE,
            # Suppression
            suppression: Suppression = DEFAULT_SUPPRESSION,
            suppression_method: SuppressionMethod = SuppressionMethod.DEFAULT,
            # Omega_gw_0 input parameters
            T_star: float | None = None,
            g_star: float | None = None,
            gs_star: float | None = None,
            # Accuracy settings
            nT: int = ssm.DEFAULT_N_T,
            nx_P_tilde_gw: int | None = None,
            n_z_lookup: int = ssm.DEFAULT_N_Z_LOOKUP,
            z_st_thresh: float = ssm.Z_ST_THRESH,
            # Switches
            compute: bool = True,
            low_k: bool = True,
            parallel: bool = True,
            # Labels
            label_latex: str | None = None,
            label_unicode: str | None = None):
        r"""
        :param bubble: the Bubble object
        :param beta_tilde: nucleation rate parameter $\tilde{\beta} \equiv \frac{\beta}{H_*}$
        :param r_star: Hubble-scaled mean bubble spacing $r_*$
        :param y: $z = k R_*$ array
        :param N_sh: $N_\text{sh}$, number of shock formation times
        :param nuc_type: nucleation type
        :param T_star: $T_*$, temperature at the time of GW production
        :param g_star: $g_*$, degrees of freedom override at the time of GW production
        :param gs_star: $g_{s,*}$ degrees of freedom override for entropy at the time of GW production
        :param nT: number of points in the t array
        :param n_z_lookup: number of points in the lookup arrays
        :param z_st_thresh: for $z$ values above z_sh_tresh,
            use approximation rather than doing the sine transform integral.
        :param compute: whether to compute the spectrum immediately
        :param low_k: whether to use the :giombi_2024_cs: approximation for low $k$
        :param parallel: whether to use multiple CPU cores
        """
        super().__init__(
            bubble=bubble,
            beta_tilde=beta_tilde,
            r_star=r_star,
            y=y,
            z_st_thresh=z_st_thresh,
            nuc_type=nuc_type,
            suppression=suppression,
            suppression_method=suppression_method,
            a_star_a_r_ratio=a_star_a_r_ratio,
            N_sh=N_sh,
            nT=nT,
            nx_P_tilde_gw=nx_P_tilde_gw,
            n_z_lookup=n_z_lookup,
            compute=compute,
            parallel=parallel,
            low_k=low_k,
            label_latex=label_latex,
            label_unicode=label_unicode
        )
        # This is needed for T_star, g_star and gs_star, and beta_tilde -> r_star conversion
        if not self.bubble.solved:
            self.bubble.solve()

        bubble_temp_physical = bubble.model.temperature_is_physical
        #: Temperature $T_*$ at the time of GW production
        self.T_star: float = T_star if T_star is not None \
            else bubble.T_star if bubble_temp_physical \
            else const.DEFAULT_T_STAR
        #: Degrees of freedom $g_*$ for pressure at the time the GWs were produced
        self.g_star: float = g_star if g_star is not None \
            else bubble.g_star if bubble_temp_physical \
            else const.DEFAULT_G_STAR
        #: Degrees of freedom $g_{s,*}$ for entropy at the time the GWs were produced
        self.gs_star: float = gs_star if gs_star is not None \
            else bubble.gs_star if bubble_temp_physical \
            else const.DEFAULT_G_STAR

    # =====
    # Properties
    # =====

    @functools.cached_property
    def ge_star(self) -> float:
        r"""Degrees of freedom $g_{e,*}$ for energy density at the time the GWs were produced
        $$g_{e,*} = \frac{1}{3}(4 g_s - g_p)$$
        :maki_msc:`\ ` eq. 2.108
        """
        return (4 * self.gs_star - self.g_star) / 3

    @functools.cached_property
    def e_star(self) -> float:
        r"""Energy density $e_*$ at GW formation
        $$e_* = \frac{\pi^2}{30} g_e(T_*) T_*^4$$
        :maki_msc:`\ ` eq. 2.105
        This presumes that $V(T_*, \phi_b) = 0$.
        """
        return np.pi**2 / 30 * self.ge_star * self.T_star ** 4

    @functools.cached_property
    def f_star0(self) -> float:  # pylint: disable=missing-function-docstring
        return freq.f_star0(
            T_star=self.T_star,
            g_star=self.g_star
        )

    @functools.cached_property
    def H_star(self):
        r"""Hubble rate $H_*$ at GW formation, in units of $T^2$
        $$H = \sqrt{8 \pi \frac{e_*}{3}} \frac{1}{m_\text{pl}}$$
        This is a direct consequence of the Friedmann equation
        $$H^2 + \frac{K}{a^2} = \frac{8 \pi G}{3} e$$
        with $K = 0$ and $m_\text{pl} = \frac{1}{G}$.
        Note that here $m_pl = 1$ in natural units.
        """
        return math.sqrt(8 * math.pi * self.e_star / 3)

    @functools.cached_property
    def R_star(self) -> th.FloatOrArr:
        r"""Mean bubble separation $R_*$, in units of $T^{-2}$
        $$R_* = \frac{r_*}{H_*}$$
        :gowling_2021:`\ ` eq. 2.2
        """
        return self.r_star / self.H_star

    @functools.cached_property
    def R_star_m(self) -> th.FloatOrArr:
        r"""Mean bubble separation $R_*$, in meters, presuming that $T$ is in GeV"""
        return self.R_star * const.GEV_IN_J * const.PLANCK_LENGTH

    # =====
    # Methods
    # =====

    def export(self, path: str | None = None) -> dict[str, tp.Any]:
        omgw0_peak = self.omgw0_peak()
        f = self.f()
        data = {
            **super().export(),
            # Input parameters
            "g_star": self.g_star,
            "gs_star": self.gs_star,
            "T_star": self.T_star,
            # Computed values
            "F_gw0": self.F_gw0(),
            "ge_star": self.ge_star,
            "e_star": self.e_star,
            "f_max": f.max(),
            "f_min": f.min(),
            "f_star0": self.f_star0,
            "H_star": self.H_star,
            "omgw0_peak_f": omgw0_peak[0],
            "omgw0_peak": omgw0_peak[1],
            "omgw0_total": self.omgw0_total(),
            "R_star": self.R_star,
            "R_star_m": self.R_star_m,
            "signal_to_noise_ratio": self.signal_to_noise_ratio(),
            "signal_to_noise_ratio_instrument": self.signal_to_noise_ratio_instrument()
        }
        if path is not None:
            export_json(data, path)
        return data

    def f(self, z: th.FloatArr | None = None) -> th.FloatOrArr:  # pylint: disable=missing-function-docstring
        if z is None:
            z = self.y
        return freq.f(z=z, r_star=self.r_star, f_star0=self.f_star0)

    def F_gw0(self, g0: float = const.G0, gs0: float = const.GS0) -> float:  # pylint: disable=missing-function-docstring
        return F_gw0(
            g_star=self.g_star,
            g0=g0,
            gs0=gs0,
            gs_star=self.gs_star
        )

    def noise(self) -> th.FloatArr1D:  # pylint: disable=missing-function-docstring
        return noise.omega_noise(self.f())

    def noise_ins(self) -> th.FloatArr1D:  # pylint: disable=missing-function-docstring
        return noise.omega_ins(self.f())

    def omgw0(
            self,
            g0: float = const.G0,
            gs0: float = const.GS0) -> th.FloatArr1D:
        r"""Gravitational wave power spectrum today $\Omega_{\text{gw},0}$"""
        return self.F_gw0(g0=g0, gs0=gs0) * self.pow_gw

    def omgw0_peak(
            self,
            g0: float = const.G0,
            gs0: float = const.GS0):
        r"""Peak $\Omega_{\text{gw},0}
        :param g0: Degrees of freedom today for pressure $g_0$
        :param gs0: Degrees of freedom today for entropy $g_{s,0}$
        :param sup: Suppression type
        :param sup_method: Suppression method
        """
        omgw0 = self.omgw0(g0=g0, gs0=gs0)
        i_max = np.argmax(omgw0)
        return self.f()[i_max], omgw0[i_max]

    def omgw0_total(self, omgw0: th.FloatArr1D | None = None) -> float:
        r"""Total $\Omega_{\text{gw},0} integrated over all frequencies"""
        if omgw0 is None:
            omgw0 = self.omgw0()
        return ssm.trapezoid_loglog(x=self.f(), y=omgw0)

    def signal_to_noise_ratio(self) -> float:
        """Signal-to-noise ratio for LISA, taking into account all noise sources"""
        snr, f_min, f_max = noise.signal_to_noise_ratio(f=self.f(), signal=self.omgw0(), noise=self.noise())
        return snr

    def signal_to_noise_ratio_instrument(self) -> float:
        """Signal-to-noise ratio for LISA, taking into account only the instrument noise"""
        snr, f_min, f_max = noise.signal_to_noise_ratio(f=self.f(), signal=self.omgw0(), noise=self.noise_ins())
        return snr

    def z_from_f[T: FloatOrArr](self, f: T) -> T:
        r"""Convert from frequencies $f$ back to wavenumbers $z$

        $$z(f) = \frac{f}{{f}_{\ast,0}} {r}_\ast$$
        Inverted from :gowling_2021:`\ ` eq. 2.12
        :param f: frequencies $f$ today
        :return: wavenumbers $z$
        """
        return freq.z(f=f, T_star=self.T_star, r_star=self.r_star, g_star=self.g_star)

    # -----
    # Plotting
    # -----

    def plot(
            self,
            fig: Figure | None = None,
            ax: Axes | None = None,
            path: str | None = None,
            **kwargs) -> "FigAndAxes":
        from pttools.analysis.plot_spectra import plot_spectra
        return plot_spectra([self], fig, ax, path, **kwargs)

    def plot_multi(
            self,
            fig: Figure | None = None,
            path: str | None = None,
            **kwargs) -> tuple[Figure, th.AxesArr2D]:
        from pttools.analysis.plot_spectra import plot_spectra_multi
        return plot_spectra_multi([self], fig, path, **kwargs)

    def plot_multi_flat(
            self,
            fig: Figure | None = None,
            path: str | None = None,
            **kwargs) -> tuple[Figure, th.AxesArr1D]:
        from pttools.analysis.plot_spectra import plot_spectra_multi_flat
        return plot_spectra_multi_flat([self], fig, path, **kwargs)


type SpectrumArr = NDArray[Spectrum]
type SpectrumArr2D = np.ndarray[tuple[int, int], np.dtype[Spectrum]]
type SpectrumArr3D = np.ndarray[tuple[int, int, int], np.dtype[Spectrum]]

copy_docstrings({
    Spectrum.f: freq.f,
    Spectrum.F_gw0: F_gw0,
    Spectrum.f_star0: freq.f_star0,
    Spectrum.noise: noise.omega_noise,
    Spectrum.noise_ins: noise.omega_ins,
}, without_params=True)
