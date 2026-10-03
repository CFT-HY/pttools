"""Gravitational wave power spectrum as observed today."""

import functools
import math
import os
import typing as tp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from pttools.bubble import Bubble
from pttools.omgw0 import const, freq
from pttools.omgw0.const import H2, LISA_OBS_TIME, OMEGA_PHOTON_H2
from pttools.omgw0.export import SPECTRUM_F_FIELDS, SPECTRUM_FIELDS
from pttools.omgw0.factors import F_gw0_h2
from pttools.omgw0.noise import omega_ins_h2, omega_noise_h2, signal_to_noise_ratio
from pttools.ssm.calculators import trapezoid_loglog
from pttools.ssm.const import (
    DEFAULT_A_STAR_A_R_RATIO,
    DEFAULT_N_SH,
    DEFAULT_N_T,
    DEFAULT_N_Z_LOOKUP,
    DEFAULT_Y,
    Z_ST_THRESH,
)
from pttools.ssm.nucleation import DEFAULT_NUC_TYPE, NucType
from pttools.ssm.spectrum import SSMSpectrum
from pttools.ssm.suppression import DEFAULT_SUPPRESSION, Suppression, SuppressionMethod
import pttools.type_hints as th
from pttools.type_hints import FloatArr1D, FloatOrArr
from pttools.utils import copy_docstrings
from pttools.utils.fields import Fields

if tp.TYPE_CHECKING:
    from pttools.analysis.utils import FigAndAxes


class Spectrum(SSMSpectrum):
    r"""A spectrum object that includes the conversion to the GW power spectrum today $\Omega_{\text{gw},0}$."""

    FIELDS: tp.ClassVar[Fields] = SPECTRUM_FIELDS
    #: The exportable fields of the spectra that have been given the frequencies $f$ instead of $y$.
    #: For these, $f$ is shared by the spectra of a file instead of $y$.
    F_FIELDS: tp.ClassVar[Fields] = SPECTRUM_F_FIELDS

    def __init__(
            self,
            bubble: Bubble,
            # Input parameters
            beta_tilde: float | None = None,
            r_star: float | None = None,
            y: th.FloatArr1D | None = None,
            f: th.FloatArr1D | None = None,
            a_star_a_r_ratio: float = DEFAULT_A_STAR_A_R_RATIO,
            N_sh: float = DEFAULT_N_SH,
            nuc_type: NucType = DEFAULT_NUC_TYPE,
            # Suppression
            suppression: Suppression = DEFAULT_SUPPRESSION,
            suppression_method: SuppressionMethod = SuppressionMethod.DEFAULT,
            # Omega_gw_0 input parameters
            T_star: float | None = None,
            g_star: float | None = None,
            gs_star: float | None = None,
            # Accuracy settings
            nT: int = DEFAULT_N_T,
            nx_P_tilde_gw: int | None = None,
            n_z_lookup: int = DEFAULT_N_Z_LOOKUP,
            z_st_thresh: float = Z_ST_THRESH,
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
        :param y: $z = k R_*$ array.
            If neither $y$ nor $f$ is given, the default of :py:mod:`pttools.ssm.const` is used.
        :param f: $f$, frequencies today, which are converted to the $y$ array with :py:meth:`z_from_f`.
            Cannot be given together with $y$.
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
        if f is not None:
            if y is not None:
                raise ValueError("Either y or f can be provided, but not both.")
            if np.isnan(f).any():
                raise ValueError("f must not contain nan values.")
        # The spectrum is computed only after the y array has been obtained,
        # since converting f to y requires T_star, g_star and r_star.
        super().__init__(
            bubble=bubble,
            beta_tilde=beta_tilde,
            r_star=r_star,
            y=DEFAULT_Y if y is None else y,  # When f is set, this is a placeholder that will be replaced below.
            z_st_thresh=z_st_thresh,
            nuc_type=nuc_type,
            suppression=suppression,
            suppression_method=suppression_method,
            a_star_a_r_ratio=a_star_a_r_ratio,
            N_sh=N_sh,
            nT=nT,
            nx_P_tilde_gw=nx_P_tilde_gw,
            n_z_lookup=n_z_lookup,
            compute=False,
            low_k=low_k,
            label_latex=label_latex,
            label_unicode=label_unicode
        )
        # This is needed for T_star, g_star and gs_star, and beta_tilde -> r_star conversion
        if not self.bubble.solved:
            self.bubble.solve()

        bubble_temp_physical = bubble.model.temperature_is_physical
        self.T_star: float = T_star if T_star is not None \
            else bubble.T_star if bubble_temp_physical \
            else const.DEFAULT_T_STAR
        self.g_star: float = g_star if g_star is not None \
            else bubble.g_star if bubble_temp_physical \
            else const.DEFAULT_G_STAR
        self.gs_star: float = gs_star if gs_star is not None \
            else bubble.gs_star if bubble_temp_physical \
            else const.DEFAULT_G_STAR

        # The frequencies given as an argument are returned by f() as is, instead of converting them back from y.
        self._f_given: th.FloatArr1D | None = f
        if f is not None:
            self.y: th.FloatArr1D = self.z_from_f(f)

        if compute:
            self.compute(parallel=parallel)

    # =====
    # Properties
    # =====

    @functools.cached_property
    def ge_star(self) -> float:
        r"""Degrees of freedom $g_{e,*}$ for energy density at the time the GWs were produced
        $$g_{e,*} = \frac{1}{3}(4 g_s - g_p)$$
        :maki_msc:`\ ` eq. 2.108.
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

    @property
    def f_given(self) -> bool:
        """Whether the frequencies $f$ were given as an argument instead of the $y$ array."""
        return self._f_given is not None

    @functools.cached_property
    def _f_y(self) -> th.FloatArr1D:
        r"""Frequencies today $f(y)$ corresponding to the $y$ array of the spectrum.

        This is cached, so that :py:meth:`f`, :py:attr:`f_min` and :py:attr:`f_max` compute the frequencies only once.
        If the frequencies were given as an argument, they are returned as is.
        """
        if self._f_given is not None:
            return self._f_given
        return freq.f(z=self.y, r_star=self.r_star, f_star0=self.f_star0)

    @functools.cached_property
    def f_max(self) -> float:
        r"""$f_\text{max}$, maximum frequency today corresponding to the $y$ array of the spectrum."""
        return float(self._f_y.max())

    @functools.cached_property
    def f_min(self) -> float:
        r"""$f_\text{min}$, minimum frequency today corresponding to the $y$ array of the spectrum."""
        return float(self._f_y.min())

    @functools.cached_property
    def f_star0(self) -> float:
        return freq.f_star0(
            T_star=self.T_star,
            g_star=self.g_star
        )

    @functools.cached_property
    def H_star(self) -> float:
        r"""Hubble rate $H_*$ at GW formation, in units of $T^2$.

        $$H = \sqrt{8 \pi \frac{e_*}{3}} \frac{1}{m_{\text{pl}}}$$
        This is a direct consequence of the Friedmann equation
        $$H^2 + \frac{K}{a^2} = \frac{8 \pi G}{3} e$$
        with $K = 0$ and $m_{\text{pl}} = \frac{1}{G}$.
        Note that here $m_pl = 1$ in natural units.
        """
        return math.sqrt(8 * math.pi * self.e_star / 3)

    @functools.cached_property
    def R_star(self) -> th.FloatOrArr:
        r"""Mean bubble separation $R_*$, in units of $T^{-2}$.

        $$R_* = \frac{r_*}{H_*}$$
        :gowling_2021:`\ ` eq. 2.2
        """
        return self.r_star / self.H_star

    @functools.cached_property
    def R_star_m(self) -> th.FloatOrArr:
        r"""Mean bubble separation $R_*$, in meters, presuming that $T$ is in GeV."""
        return self.R_star * const.GEV_IN_J * const.PLANCK_LENGTH

    # =====
    # Methods
    # =====

    def f(self, z: th.FloatArr1D | None = None) -> th.FloatArr1D:
        # The docstring is copied from freq.f.
        # The frequencies of the y array are cached, and the frequencies of a custom z are not.
        if z is None:
            return self._f_y
        return freq.f(z=z, r_star=self.r_star, f_star0=self.f_star0)

    def F_gw0[T: FloatOrArr](
            self,
            g0: T = const.G0,
            gs0: th.FloatOrArr = const.GS0,
            h2: th.FloatOrArr = H2) -> T:
        r"""$F_{\text{gw},0}$, power attenuation following the end of the radiation era."""
        return self.F_gw0_h2(g0=g0, gs0=gs0) / h2  # pyrefly: ignore[bad-return]

    def F_gw0_h2[T: FloatOrArr](
            self,
            g0: T = const.G0,
            gs0: th.FloatOrArr = const.GS0,
            om_gamma0_h2: th.FloatOrArr = OMEGA_PHOTON_H2) -> T:
        return F_gw0_h2(  # pyrefly: ignore[bad-return]
            g_star=self.g_star,
            g0=g0,
            gs0=gs0,
            gs_star=self.gs_star,
            om_gamma0_h2=om_gamma0_h2
        )

    def noise(self, eb: bool = True, gb: bool = True, ins: bool = True, h2: th.FloatOrArr1D = H2) -> th.FloatArr1D:
        r"""Total LISA noise $\Omega_\text{noise}$."""
        return self.noise_h2(eb=eb, gb=gb, ins=ins) / h2

    def noise_h2(self, eb: bool = True, gb: bool = True, ins: bool = True) -> th.FloatArr1D:
        return omega_noise_h2(f=self.f(), eb=eb, gb=gb, ins=ins)

    def noise_ins(self, h2: th.FloatOrArr1D) -> th.FloatArr1D:
        r"""LISA instrument noise $\Omega_\text{ins}$."""
        return self.noise_ins_h2() / h2

    def noise_ins_h2(self) -> th.FloatArr1D:
        return omega_ins_h2(f=self.f())

    def omgw0(
            self,
            g0: float = const.G0,
            gs0: float = const.GS0,
            h2: th.FloatOrArr1D = H2) -> th.FloatArr1D:
        r"""Gravitational wave power spectrum today $\Omega_{\text{gw},0}$.

        :param g0: $g_0$, degrees of freedom today for pressure
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param h2: $h^2$, dimensionless reduced Hubble constant squared
        """
        return self.omgw0_h2(g0=g0, gs0=gs0) / h2

    def omgw0_h2(
            self,
            g0: float = const.G0,
            gs0: float = const.GS0,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2) -> th.FloatArr1D:
        r"""Gravitational wave power spectrum today $\Omega_{\text{gw},0} h^2$.

        :param g0: $g_0$, degrees of freedom today for pressure
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
        """
        return self.F_gw0_h2(g0=g0, gs0=gs0, om_gamma0_h2=om_gamma0_h2) * self.pow_gw

    def omgw0_h2_peak(
            self,
            g0: float = const.G0,
            gs0: float = const.GS0) -> tuple[float, float]:
        r"""Peak $\Omega_{\text{gw},0} h^2$.

        :param g0: $g_0$, degrees of freedom today for pressure
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        """
        omgw0_h2 = self.omgw0_h2(g0=g0, gs0=gs0)
        i_max = np.argmax(omgw0_h2)
        return self.f()[i_max], omgw0_h2[i_max]

    def omgw0_h2_total(self, omgw0_h2: th.FloatArr1D | None = None) -> float:
        r"""Total $\Omega_{\text{gw},0} h^2$ integrated over all frequencies.

        :param omgw0_h2: $\Omega_{\text{gw},0} h^2$
        """
        return trapezoid_loglog(x=self.f(), y=self.omgw0_h2() if omgw0_h2 is None else omgw0_h2)

    def omgw0_peak[T: FloatOrArr](
            self,
            g0: float = const.G0,
            gs0: float = const.GS0,
            h2: T = H2) -> tuple[float, T]:
        r"""Peak $\Omega_{\text{gw},0}$.

        :param g0: $g_0$, degrees of freedom today for pressure
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param h2: $h^2$, dimensionless reduced Hubble constant squared
        """
        f_peak, omgw0_h2_peak = self.omgw0_h2_peak(g0=g0, gs0=gs0)
        return f_peak, tp.cast(T, omgw0_h2_peak / h2)

    def omgw0_total[T: FloatOrArr](
            self,
            omgw0_h2: th.FloatArr1D | None = None,
            h2: T = H2) -> T:
        r"""Total $\Omega_{\text{gw},0}$ integrated over all frequencies.

        :param omgw0_h2: $\Omega_{\text{gw},0} h^2$
        :param h2: $h^2$, dimensionless reduced Hubble constant squared
        """
        return tp.cast(T, self.omgw0_h2_total(omgw0_h2=omgw0_h2) / h2)

    def snr(
            self,
            # Arrays
            noise: FloatArr1D | None = None,
            f_noise: FloatArr1D | None = None,
            # Scalars
            g0: float = const.G0,
            gs0: float = const.GS0,
            obs_time: float = LISA_OBS_TIME,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2,
            # Switches
            noise_eb: bool = True,
            noise_gb: bool = True,
            noise_ins: bool = True,
        ) -> tuple[float, FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D]:
        r"""Signal-to-noise ratio for LISA, taking into account all noise sources.

        :param noise: $\Omega_\text{noise} h^2$
        :param f_noise: frequencies for the noise (assumed to be the same as for the signal, if not provided)
        :param g0: $g_0$, degrees of freedom today for pressure
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param obs_time: observation time (s)
        :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
        :param noise_eb: whether to generate extragalactic compact binary noise when noise is not provided
        :param noise_gb: whether to generate galactic compact binary noise when noise is not provided
        :param noise_ins: whether to generate instrument noise when noise is not provided
        """
        f: FloatArr1D = self.f()
        omgw0_h2 = self.omgw0_h2(g0=g0, gs0=gs0, om_gamma0_h2=om_gamma0_h2)
        snr, f_noise, noise = signal_to_noise_ratio(
            f=f, signal=omgw0_h2, obs_time=obs_time,
            noise=noise, f_noise=f_noise,
            noise_eb=noise_eb, noise_gb=noise_gb, noise_ins=noise_ins
        )
        return snr, f, omgw0_h2, f_noise, noise

    def snr_ins(
            self,
            g0: float = const.G0,
            gs0: float = const.GS0,
            obs_time: float = LISA_OBS_TIME,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2
        ) -> tuple[float, FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D]:
        """Signal-to-noise ratio for LISA, taking into account only the instrument noise."""
        f: FloatArr1D = self.f()
        omgw0_h2 = self.omgw0_h2(g0=g0, gs0=gs0, om_gamma0_h2=om_gamma0_h2)
        snr, f_noise, noise = signal_to_noise_ratio(
            f=f, signal=omgw0_h2, obs_time=obs_time,
            noise_eb=False, noise_gb=False, noise_ins=True
        )
        return snr, f, omgw0_h2, f_noise, noise

    def z_from_f[T: FloatOrArr](self, f: T) -> T:
        r"""Convert from frequencies $f$ back to wavenumbers $z$.

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
            path: str | os.PathLike[str] | None = None,
            **kwargs: tp.Any) -> "FigAndAxes":
        from pttools.analysis.plot_spectra import plot_spectra  # noqa: PLC0415
        return plot_spectra([self], fig, ax, path, **kwargs)

    def plot_multi(
            self,
            fig: Figure | None = None,
            path: str | os.PathLike[str] | None = None,
            **kwargs: tp.Any) -> tuple[Figure, th.AxesArr2D]:
        from pttools.analysis.plot_spectra import plot_spectra_multi  # noqa: PLC0415
        return plot_spectra_multi([self], fig, path, **kwargs)

    def plot_multi_flat(
            self,
            fig: Figure | None = None,
            path: str | os.PathLike[str] | None = None,
            label: str | None = None,
            legend: bool = False,
            **kwargs: tp.Any) -> tuple[Figure, th.AxesArr1D]:
        from pttools.analysis.plot_spectra import plot_spectra_multi_flat  # noqa: PLC0415
        return plot_spectra_multi_flat([self], fig=fig, path=path, labels=[label], legend=legend, **kwargs)


# These are object arrays. Numpy typing has no way of expressing the element type of an object array,
# but declaring the element type here does give the correct types when the arrays are indexed.
type SpectrumArr = NDArray[Spectrum]  # pyrefly: ignore[bad-specialization]
type SpectrumArr2D = np.ndarray[tuple[int, int], np.dtype[Spectrum]]  # pyrefly: ignore[bad-specialization]
type SpectrumArr3D = np.ndarray[tuple[int, int, int], np.dtype[Spectrum]]  # pyrefly: ignore[bad-specialization]

copy_docstrings({
    Spectrum.f: freq.f,
    Spectrum.F_gw0_h2: F_gw0_h2,
    Spectrum.f_star0: freq.f_star0,
    Spectrum.noise_h2: omega_noise_h2,
    Spectrum.noise_ins_h2: omega_ins_h2,
}, without_params=True)
