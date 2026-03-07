import functools
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
from pttools.omgw0 import suppression as sup_mod
from pttools import ssm
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr
from pttools.utils.docstrings import copy_docstrings

if tp.TYPE_CHECKING:
    from pttools.analysis.utils import FigAndAxes


class Spectrum(ssm.SSMSpectrum):
    r"""A spectrum object that includes the conversion to the GW power spectrum today $\Omega_{\text{gw},0}$"""
    def __init__(
            self,
            bubble: Bubble,
            r_star: float,
            y: th.FloatArr1D | None = None,
            z_st_thresh: float = ssm.Z_ST_THRESH,
            nuc_type: ssm.NucType = ssm.DEFAULT_NUC_TYPE,
            nt: int = ssm.NTDEFAULT,
            n_z_lookup: int = ssm.N_Z_LOOKUP_DEFAULT,
            lifetime_multiplier: float = 1,
            compute: bool = True,
            parallel: bool = True,
            low_k: bool = True,
            label_latex: str | None = None,
            label_unicode: str | None = None,
            T_star: float | None = None,
            g_star: float | None = None,
            gs_star: float | None = None
            ):
        """
        :param bubble: the Bubble object
        :param r_star: Hubble-scaled mean bubble spacing $r_*$
        :param y: $z = kR*$ array
        :param z_st_thresh: for $z$ values above z_sh_tresh, use approximation rather than doing the sine transform integral.
        :param nuc_type: nucleation type
        :param nt: number of points in the t array
        :param lifetime_multiplier: used for computing the source lifetime factor
        :param compute: whether to compute the spectrum immediately
        :param T_star: $T_*$, temperature at the time of GW production
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
            compute=compute,
            parallel=parallel,
            low_k=low_k,
            label_latex=label_latex,
            label_unicode=label_unicode
        )
        # This is needed for T_star, g_star and gs_star
        if not self.bubble.solved:
            self.bubble.solve()

        bubble_temp_physical = bubble.model.temperature_is_physical
        #: Temperature $T_*$ at the time of GW production
        self.T_star: float = T_star if T_star is not None \
            else bubble.T_star if bubble_temp_physical \
            else const.T_STAR_DEFAULT
        #: Degrees of freedom $g_*$ for pressure at the time the GWs were produced
        self.g_star: float = g_star if g_star is not None \
            else bubble.g_star if bubble_temp_physical \
            else const.G_STAR_DEFAULT
        #: Degrees of freedom $g_{s,*}$ for entropy at the time the GWs were produced
        self.gs_star: float = gs_star if gs_star is not None \
            else bubble.gs_star if bubble_temp_physical \
            else const.G_STAR_DEFAULT

    # =====
    # Properties
    # =====

    @functools.cached_property
    def f_star0(self) -> float:  # pylint: disable=missing-function-docstring
        return freq.f_star0(
            T_star=self.T_star,
            g_star=self.g_star
        )

    @property
    def H_n(self):
        """Hubble constant at nucleation temperature, $H_n$

        $$H_n = H(T_n)$$
        """
        return ssm.H(T=self.T_star)

    # =====
    # Methods
    # =====

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
            gs0: float = const.GS0,
            sup: sup_mod.Suppression = sup_mod.DEFAULT,
            sup_method: sup_mod.SuppressionMethod = sup_mod.SuppressionMethod.DEFAULT) -> th.FloatArr1D:
        r"""Gravitational wave power spectrum today $\Omega_{\text{gw},0}$"""
        # The r_star compensates the fact that the pow_gw includes a correction factor that is J without r_star
        return self.r_star * self.F_gw0(g0=g0, gs0=gs0) * self.pow_gw * \
            self.suppression_factor(suppression=sup, method=sup_method)

    def omgw0_peak(
            self,
            g0: float = const.G0,
            gs0: float = const.GS0,
            sup: sup_mod.Suppression = sup_mod.DEFAULT,
            sup_method: sup_mod.SuppressionMethod = sup_mod.SuppressionMethod.DEFAULT):
        r"""Peak $\Omega_{\text{gw},0}
        :param g0: Degrees of freedom today for pressure $g_0$
        :param gs0: Degrees of freedom today for entropy $g_{s,0}$
        :param sup: Suppression type
        :param sup_method: Suppression method
        """
        omgw0 = self.omgw0(g0=g0, gs0=gs0, sup=sup, sup_method=sup_method)
        i_max = np.argmax(omgw0)
        return self.f()[i_max], omgw0[i_max]

    def omgw0_total(self, omgw0: th.FloatArr1D | None = None) -> float:
        r"""Total $\Omega_{\text{gw},0} integrated over all frequencies"""
        if omgw0 is None:
            omgw0 = self.omgw0()
        return ssm.trapezoid_loglog(x=self.f(), y=omgw0)

    def R_star(self, H_n: th.FloatOrArr | None = None) -> th.FloatOrArr:
        r"""Mean bubble separation $R_*$
        $$R_* = \frac{r_*}{H_n}$$
        :gowling_2021:`\ ` eq. 2.2
        """
        if H_n is None:
            H_n = self.H_n
        return self.r_star / H_n

    def signal_to_noise_ratio(self) -> float:
        """Signal-to-noise ratio for LISA, taking into account all noise sources"""
        snr, f_min, f_max = noise.signal_to_noise_ratio(f=self.f(), signal=self.omgw0(), noise=self.noise())
        return snr

    def signal_to_noise_ratio_instrument(self) -> float:
        """Signal-to-noise ratio for LISA, taking into account only the instrument noise"""
        snr, f_min, f_max = noise.signal_to_noise_ratio(f=self.f(), signal=self.omgw0(), noise=self.noise_ins())
        return snr

    def suppression_factor(
            self,
            suppression: sup_mod.Suppression = sup_mod.DEFAULT,
            method: sup_mod.SuppressionMethod = sup_mod.SuppressionMethod.DEFAULT) -> float:
        return suppression.suppression(v_wall=self.bubble.v_wall, alpha_n=self.bubble.alpha_n, method=method)

    def tau_nl[T: FloatOrArr](self, H_n: T) -> T:
        r"""Timescale of nonlinearities $\tau_\text{nl}$
        $$\tau_\text{nl} = \frac{R_*}{\bar{U}_f}$
        :gw_pt_ssm:`\ ` p. 6
        :notes:`\ ` p. 48
        :giombi_2024_cs:`\ ` p. 2
        """
        return self.R_star(H_n) / self.bubble.ubarf

    def z_from_f[T: FloatOrArr](self, f: T) -> T:
        r"""Convert from frequencies $f$ back to wavenumbers $z$

        $$z(f) = \frac{f}{{f}_{*,0} {r}_*$$
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
    Spectrum.noise_ins: noise.omega_ins
}, without_params=True)
