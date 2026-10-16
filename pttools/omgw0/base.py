r"""Common base class of the spectra that can be converted to the GW power spectrum today $\Omega_{\text{gw},0}$.

The base class is shared by the spectra of the Sound Shell Model (:py:class:`pttools.omgw0.spectrum.Spectrum`)
and of hydrodynamic lattice simulations (:py:class:`pttools.imports.hydro.spectrum.HydroSpectrum`),
so that they can be compared with each other and with the sensitivity of LISA.
"""

import abc
import functools
import os
import typing as tp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from pttools.omgw0 import const, freq
from pttools.omgw0.const import H2, LISA_OBS_TIME, OMEGA_PHOTON_H2
from pttools.omgw0.factors import F_gw0_h2
from pttools.omgw0.noise import omega_ins_h2, omega_noise_h2, signal_to_noise_ratio
from pttools.ssm.calculators import trapezoid_loglog
from pttools.ssm.spectrum_base import SpectrumBase
import pttools.type_hints as th
from pttools.type_hints import FloatArr1D, FloatOrArr
from pttools.utils.docstrings import copy_docstring_dec

if tp.TYPE_CHECKING:
    from pttools.analysis.utils import FigAndAxes


class Omgw0Base(SpectrumBase):
    r"""Base class for the spectra that can be converted to the GW power spectrum today $\Omega_{\text{gw},0}$.

    In addition to the attributes and properties of :py:class:`pttools.ssm.spectrum_base.SpectrumBase`,
    the subclasses have to provide the attributes :py:attr:`T_star`, :py:attr:`g_star` and :py:attr:`gs_star`
    and the properties :py:attr:`H_star` and :py:attr:`R_star`.
    The frequencies today $f$ follow from $z = kR_*$, $r_*$ and $f_{\ast,0}$ (see :py:mod:`pttools.omgw0.freq`),
    and the GW power spectrum today from $\mathcal{P}_\text{gw}$ and $F_{\text{gw},0}$.
    """

    #: $T_*$, temperature at the time of GW production in GeV
    T_star: float
    #: $g_*$, degrees of freedom for pressure at the time of GW production
    g_star: float
    #: $g_{s,*}$, degrees of freedom for entropy at the time of GW production
    gs_star: float

    # =====
    # Properties
    # =====

    @property
    @abc.abstractmethod
    def H_star(self) -> float:
        r"""$H_*$, Hubble rate at the time of GW production.

        $$H_* = \sqrt{\frac{8 \pi G}{3} e_*},$$
        where $e_*$ is the energy density at the time of GW production.
        This is a direct consequence of the Friedmann equation
        $$H^2 + \frac{K}{a^2} = \frac{8 \pi G}{3} e$$
        with $K = 0$, see :py:func:`pttools.omgw0.factors.H`.
        The units depend on the spectrum:
        :py:class:`pttools.omgw0.spectrum.Spectrum` uses the units of $T^2$, where $G = 1 / m_{\text{pl}}^2 = 1$,
        and :py:class:`pttools.imports.hydro.spectrum.HydroSpectrum` uses the lattice units of the simulation.
        """

    @property
    @abc.abstractmethod
    def R_star(self) -> float:
        r"""$R_*$, mean bubble separation.

        $$R_* = \frac{r_*}{H_*}$$
        :gowling_2021:`\ ` eq. 2.2.
        The units are those of $\frac{1}{H_*}$.
        """

    @functools.cached_property
    def ge_star(self) -> float:
        r"""$g_{e,*}$, degrees of freedom for energy density at the time the GWs were produced.

        $$g_{e,*} = \frac{1}{3}(4 g_s - g_p)$$
        :maki_msc:`\ ` eq. 2.108.
        """
        return (4 * self.gs_star - self.g_star) / 3

    @functools.cached_property
    def _f_y(self) -> th.FloatArr1D:
        r"""Frequencies today $f(y)$ corresponding to the $y$ array of the spectrum.

        This is cached, so that :py:meth:`f`, :py:attr:`f_min` and :py:attr:`f_max` compute the frequencies only once.
        """
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
    @copy_docstring_dec(freq.f_star0, without_params=True)
    def f_star0(self) -> float:
        return freq.f_star0(
            T_star=self.T_star,
            ge_star=self.ge_star,
            gs_star=self.gs_star
        )

    # =====
    # Methods
    # =====

    @copy_docstring_dec(freq.f, without_params=True)
    def f(self, z: th.FloatArr1D | None = None) -> th.FloatArr1D:
        # The frequencies of the y array are cached, and the frequencies of a custom z are not.
        if z is None:
            return self._f_y
        return freq.f(z=z, r_star=self.r_star, f_star0=self.f_star0)

    def F_gw0[T: FloatOrArr](
            self,
            ge0_photon: T = const.GE0_PHOTON,
            gs0: th.FloatOrArr = const.GS0,
            h2: th.FloatOrArr = H2) -> T:
        r"""$F_{\text{gw},0}$, power attenuation following the end of the radiation era."""
        return self.F_gw0_h2(ge0_photon=ge0_photon, gs0=gs0) / h2  # pyrefly: ignore[bad-return]

    @copy_docstring_dec(F_gw0_h2, without_params=True)
    def F_gw0_h2[T: FloatOrArr](
            self,
            ge0_photon: T = const.GE0_PHOTON,
            gs0: th.FloatOrArr = const.GS0,
            om_gamma0_h2: th.FloatOrArr = OMEGA_PHOTON_H2) -> T:
        return F_gw0_h2(  # pyrefly: ignore[bad-return]
            ge_star=self.ge_star,
            ge0_photon=ge0_photon,
            gs0=gs0,
            gs_star=self.gs_star,
            om_gamma0_h2=om_gamma0_h2
        )

    def noise(self, eb: bool = True, gb: bool = True, ins: bool = True, h2: th.FloatOrArr1D = H2) -> th.FloatArr1D:
        r"""Total LISA noise $\Omega_\text{noise}$."""
        return self.noise_h2(eb=eb, gb=gb, ins=ins) / h2

    @copy_docstring_dec(omega_noise_h2, without_params=True)
    def noise_h2(self, eb: bool = True, gb: bool = True, ins: bool = True) -> th.FloatArr1D:
        return omega_noise_h2(f=self.f(), eb=eb, gb=gb, ins=ins)

    def noise_ins(self, h2: th.FloatOrArr1D) -> th.FloatArr1D:
        r"""LISA instrument noise $\Omega_\text{ins}$."""
        return self.noise_ins_h2() / h2

    @copy_docstring_dec(omega_ins_h2, without_params=True)
    def noise_ins_h2(self) -> th.FloatArr1D:
        return omega_ins_h2(f=self.f())

    def omgw0(
            self,
            ge0_photon: float = const.GE0_PHOTON,
            gs0: float = const.GS0,
            h2: th.FloatOrArr1D = H2) -> th.FloatArr1D:
        r"""Gravitational wave power spectrum today $\Omega_{\text{gw},0}$.

        :param ge0_photon: $g_{e\gamma 0}$, degrees of freedom for energy density of photons today
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param h2: $h^2$, dimensionless reduced Hubble constant squared
        """
        return self.omgw0_h2(ge0_photon=ge0_photon, gs0=gs0) / h2

    def omgw0_h2(
            self,
            ge0_photon: float = const.GE0_PHOTON,
            gs0: float = const.GS0,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2) -> th.FloatArr1D:
        r"""Gravitational wave power spectrum today $\Omega_{\text{gw},0} h^2$.

        :param ge0_photon: $g_{e\gamma 0}$, degrees of freedom for energy density of photons today
        :param gs0: $g_{s,0}$, degrees of freedom for entropy today
        :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
        """
        return self.F_gw0_h2(ge0_photon=ge0_photon, gs0=gs0, om_gamma0_h2=om_gamma0_h2) * self.pow_gw

    def omgw0_h2_peak(
            self,
            ge0_photon: float = const.GE0_PHOTON,
            gs0: float = const.GS0,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2) -> tuple[float, float]:
        r"""Peak $\Omega_{\text{gw},0} h^2$.

        :param ge0_photon: $g_{e\gamma 0}$, degrees of freedom for energy density of photons today
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
        """
        omgw0_h2 = self.omgw0_h2(ge0_photon=ge0_photon, gs0=gs0, om_gamma0_h2=om_gamma0_h2)
        i_max = np.argmax(omgw0_h2)
        return self.f()[i_max], omgw0_h2[i_max]

    def omgw0_h2_total(self, omgw0_h2: th.FloatArr1D | None = None) -> float:
        r"""Total $\Omega_{\text{gw},0} h^2$ integrated over all frequencies.

        :param omgw0_h2: $\Omega_{\text{gw},0} h^2$
        """
        return trapezoid_loglog(
            x=self.f(),
            # This doesn't take arguments, as the entire omgw0_h2 can be overridden with an argument.
            y=self.omgw0_h2() if omgw0_h2 is None else omgw0_h2
        )

    def omgw0_peak[T: FloatOrArr](
            self,
            ge0_photon: float = const.GE0_PHOTON,
            gs0: float = const.GS0,
            h2: T = H2,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2) -> tuple[float, T]:
        r"""Peak $\Omega_{\text{gw},0}$.

        :param ge0_photon: $g_{e\gamma 0}$, degrees of freedom for energy density of photons today
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param h2: $h^2$, dimensionless reduced Hubble constant squared
        :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
        """
        f_peak, omgw0_h2_peak = self.omgw0_h2_peak(ge0_photon=ge0_photon, gs0=gs0, om_gamma0_h2=om_gamma0_h2)
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
            ge0_photon: float = const.GE0_PHOTON,
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
        :param ge0_photon: $g_{e\gamma 0}$, degrees of freedom for energy density of photons today
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param obs_time: observation time (s)
        :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
        :param noise_eb: whether to generate extragalactic compact binary noise when noise is not provided
        :param noise_gb: whether to generate galactic compact binary noise when noise is not provided
        :param noise_ins: whether to generate instrument noise when noise is not provided
        :return: the signal-to-noise ratio, the frequencies of the signal, $\Omega_{\text{gw},0} h^2$,
            the frequencies of the noise and $\Omega_\text{noise} h^2$
        """
        f: FloatArr1D = self.f()
        omgw0_h2 = self.omgw0_h2(ge0_photon=ge0_photon, gs0=gs0, om_gamma0_h2=om_gamma0_h2)
        snr, f_noise, noise = signal_to_noise_ratio(
            f=f, signal=omgw0_h2, obs_time=obs_time,
            noise=noise, f_noise=f_noise,
            noise_eb=noise_eb, noise_gb=noise_gb, noise_ins=noise_ins
        )
        return snr, f, omgw0_h2, f_noise, noise

    def snr_ins(
            self,
            ge0_photon: float = const.GE0_PHOTON,
            gs0: float = const.GS0,
            obs_time: float = LISA_OBS_TIME,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2
        ) -> tuple[float, FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D]:
        """Signal-to-noise ratio for LISA, taking into account only the instrument noise."""
        return self.snr(
            ge0_photon=ge0_photon, gs0=gs0, obs_time=obs_time, om_gamma0_h2=om_gamma0_h2,
            noise_eb=False, noise_gb=False, noise_ins=True
        )

    @copy_docstring_dec(freq.z, without_params=True)
    def z_from_f[T: FloatOrArr](self, f: T) -> T:
        return freq.z(f=f, r_star=self.r_star, f_star0=self.f_star0)

    # -----
    # Plotting
    # -----

    def plot(
            self,
            fig: Figure | None = None,
            ax: Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            **kwargs: tp.Any) -> "FigAndAxes":
        r"""Plot the GW power spectrum today $\Omega_{\text{gw},0} h^2(f)$ together with the LISA noise.

        :param fig: the figure to plot on
        :param ax: the axes to plot on
        :param path: if given, save the figure to this path
        :param kwargs: additional arguments for :py:func:`pttools.analysis.plot_spectra.plot_spectra`
        :return: the figure and the axes
        """
        from pttools.analysis.plot_spectra import plot_spectra  # noqa: PLC0415
        return plot_spectra([self], fig, ax, path, **kwargs)
