r"""Common interface of the power spectra from PTtools and from external sources.

The quantities of this interface can be computed both by the Sound Shell Model
(:py:class:`pttools.ssm.spectrum.SSMSpectrum`) and from the output of hydrodynamic lattice simulations
(:py:class:`pttools.lattice.spectrum.LatticeSpectrum`).
They are defined in flat space, without the expansion of the universe,
so that the spectra can be compared directly with each other.
"""

import abc
import os
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.type_hints import FloatArr1D

if tp.TYPE_CHECKING:
    from pttools.analysis.utils import FigAndAxes


class SpectrumBase(abc.ABC):
    r"""Base class for the power spectra that can be compared with each other.

    The subclasses have to provide the attributes ``y``, ``r_star``, ``label_latex`` and ``label_unicode``
    and the properties :py:attr:`pow_U_par`, :py:attr:`pow_gw` and :py:attr:`pow_gw_rate`.
    The derived quantities are then computed from these
    by integrating over $\ln z$ with the trapezoidal rule.
    The power spectra are normalised as in :correia_2025:`\ ` eq. 15, so that
    $$\int \frac{dz}{z} \mathcal{P}_X(z) = \langle X^2 \rangle.$$
    """

    #: $z = kR_*$, wavenumbers scaled by the mean bubble spacing
    y: FloatArr1D
    # This is an attribute instead of an abstract property,
    # as pttools.ssm.spectrum.SSMSpectrum stores it as an instance attribute.
    #: $r_* = H_* R_*$, Hubble-scaled mean bubble spacing
    r_star: float
    #: Label of the spectrum in LaTeX format
    label_latex: str
    #: Label of the spectrum in Unicode format
    label_unicode: str

    # -----
    # Spectra
    # -----

    @property
    @abc.abstractmethod
    def pow_U_par(self) -> FloatArr1D:
        r"""$\mathcal{P}_{U_{\parallel}}(z)$, power spectrum of the enthalpy-weighted compressional fluid velocity.

        The enthalpy-weighted fluid 4-velocity is
        $$U^i = \sqrt{\frac{w}{\bar{w}}} \gamma v^i$$
        :correia_2025:`\ ` eq. 20.
        The spectrum is normalised so that $\int \frac{dz}{z} \mathcal{P}_{U_{\parallel}} = \bar{U}_\parallel^2$.
        """

    @property
    @abc.abstractmethod
    def pow_gw(self) -> FloatArr1D:
        r"""$\mathcal{P}_\text{gw}(z)$, GW power spectrum.

        This is the GW energy density per logarithmic wavenumber
        as a fraction of the background energy density, normalised so that
        $$\int \frac{dz}{z} \mathcal{P}_\text{gw}(z) = \Omega_\text{gw}.$$
        The time at which the spectrum is evaluated depends on the spectrum.
        Unlike the other quantities of this interface, it therefore depends on the history of the source,
        and the spectra of different sources cannot in general be compared directly.

        - :py:class:`pttools.ssm.spectrum.SSMSpectrum`: the spectrum after the source has turned off,
          including the effects of the expansion of the universe and of the finite lifetime of the source.
        - :py:class:`pttools.lattice.spectrum.LatticeSpectrum`: the spectrum at the time step of the spectrum
          in flat space, as a fraction of the initial mean energy density $\bar{\epsilon}_0$,
          :correia_2025:`\ ` eq. 25.
        """

    @property
    @abc.abstractmethod
    def pow_gw_rate(self) -> FloatArr1D:
        r"""$\frac{1}{H_n R_*} \frac{d\mathcal{P}_\text{gw}}{d(H_n t)}$, growth rate of the GW power spectrum.

        This is the rate at which the GW power spectrum grows in flat space
        per Hubble time and per Hubble-scaled mean bubble spacing $H_n R_*$,
        as in :correia_2025:`\ ` fig. 5 and :gw_pt_ssm:`\ ` eq. 3.45.
        """

    # -----
    # Derived quantities
    # -----

    def _integrate_ln_y(self, integrand: FloatArr1D) -> float:
        r"""Integrate over $\ln z$ using only the finite values of the integrand."""
        finite = np.isfinite(integrand)
        return float(np.trapezoid(integrand[finite], np.log(self.y[finite])))

    @property
    def ubar_par(self) -> float:
        r"""$\bar{U}_\parallel$, RMS enthalpy-weighted compressional fluid velocity.

        $$\bar{U}_\parallel^2 = \int \frac{dz}{z} \mathcal{P}_{U_{\parallel}}(z)$$
        """
        return np.sqrt(self._integrate_ln_y(self.pow_U_par))

    @property
    def integral_scale_par(self) -> float:
        r"""$\xi(U_{\parallel}) / R_*$, integral scale of the compressional velocity in units of $R_*$.

        $$\frac{\xi}{R_*} = \frac{1}{\bar{U}_\parallel^2} \int \frac{dz}{z} \frac{1}{z} \mathcal{P}_{U_{\parallel}}(z)$$
        :correia_2025:`\ ` eq. 39.
        """
        pow_U = self.pow_U_par
        return self._integrate_ln_y(pow_U / self.y) / self._integrate_ln_y(pow_U)

    @property
    def omega_gw_rate(self) -> float:
        r"""$\frac{1}{H_n R_*} \frac{d\Omega_\text{gw}}{d(H_n t)}$, growth rate of the GW energy density fraction.

        $$\frac{1}{H_n R_*} \frac{d\Omega_\text{gw}}{d(H_n t)}
        = \int \frac{dz}{z} \frac{1}{H_n R_*} \frac{d\mathcal{P}_\text{gw}}{d(H_n t)}$$
        """
        return self._integrate_ln_y(self.pow_gw_rate)

    # -----
    # Plotting
    # -----

    def plot(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            **kwargs: tp.Any) -> "FigAndAxes":
        r"""Plot the GW power spectrum $\mathcal{P}_\text{gw}(kR_*)$, see :py:meth:`plot_gw`."""
        return self.plot_gw(fig, ax, path, **kwargs)

    def plot_gw(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            **kwargs: tp.Any) -> "FigAndAxes":
        r"""Plot the GW power spectrum $\mathcal{P}_\text{gw}(kR_*)$.

        :param fig: figure to plot in
        :param ax: axes to plot in
        :param path: path to save the figure to
        :param kwargs: arguments for :py:func:`pttools.analysis.plot_spectra.plot_spectra_gw`
        :return: the figure and the axes
        """
        from pttools.analysis.plot_spectra import plot_spectra_gw  # noqa: PLC0415
        return plot_spectra_gw([self], ax=ax, fig=fig, path=path, **kwargs)

    def plot_pow_U_par(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            scale: float = 1.,
            **kwargs: tp.Any) -> "FigAndAxes":
        r"""Plot the compressional velocity power spectrum $\mathcal{P}_{U_{\parallel}}(z)$.

        :param fig: figure to plot in
        :param ax: axes to plot in
        :param path: path to save the figure to
        :param scale: factor by which the spectrum is multiplied
        :param kwargs: arguments for :py:meth:`matplotlib.axes.Axes.plot`, e.g. ``label``
        :return: the figure and the axes
        """
        from pttools.analysis.plot_spectra import plot_spectra_pow_U_par  # noqa: PLC0415
        return plot_spectra_pow_U_par([self], ax=ax, fig=fig, path=path, scale=scale, **kwargs)

    def plot_pow_gw_rate(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            scale: float = 1.,
            **kwargs: tp.Any) -> "FigAndAxes":
        r"""Plot the growth rate of the GW power spectrum.

        :param fig: figure to plot in
        :param ax: axes to plot in
        :param path: path to save the figure to
        :param scale: factor by which the spectrum is multiplied, e.g. a suppression factor
        :param kwargs: arguments for :py:meth:`matplotlib.axes.Axes.plot`, e.g. ``label``
        :return: the figure and the axes
        """
        from pttools.analysis.plot_spectra import plot_spectra_pow_gw_rate  # noqa: PLC0415
        return plot_spectra_pow_gw_rate([self], ax=ax, fig=fig, path=path, scale=scale, **kwargs)
