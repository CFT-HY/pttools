r"""Common interface of the power spectra from PTtools and from external sources.

The quantities of this interface can be computed both by the Sound Shell Model
(:py:class:`pttools.ssm.spectrum.SSMSpectrum`) and from the output of hydrodynamic lattice simulations
(:py:class:`pttools.imports.hydro.spectrum.HydroSpectrum`).
They are defined in flat space, without the expansion of the universe,
so that the spectra can be compared directly with each other.
"""

import os
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.type_hints import FloatArr1D

if tp.TYPE_CHECKING:
    from pttools.analysis.utils import FigAndAxes


class SpectrumBase:
    r"""Base class for the power spectra that can be compared with each other.

    The subclasses have to provide the attributes ``y``, ``label_latex`` and ``label_unicode``
    and the properties :py:attr:`pow_U_par` and :py:attr:`pow_gw_rate`.
    The derived quantities are then computed from these
    by integrating over $\ln z$ with the trapezoidal rule.
    The power spectra are normalised as in :correia_2025:`\ ` eq. 15, so that
    $$\int \frac{dz}{z} \mathcal{P}_X(z) = \langle X^2 \rangle.$$
    """

    #: $z = kR_*$, wavenumbers scaled by the mean bubble spacing
    y: FloatArr1D
    #: Label of the spectrum in LaTeX format
    label_latex: str
    #: Label of the spectrum in Unicode format
    label_unicode: str

    # -----
    # Spectra
    # -----

    @property
    def pow_U_par(self) -> FloatArr1D:
        r"""$\mathcal{P}_{U_{\parallel}}(z)$, power spectrum of the enthalpy-weighted compressional fluid velocity.

        The enthalpy-weighted fluid 4-velocity is
        $$U^i = \sqrt{\frac{w}{\bar{w}}} \gamma v^i$$
        :correia_2025:`\ ` eq. 20.
        The spectrum is normalised so that $\int \frac{dz}{z} \mathcal{P}_{U_{\parallel}} = \bar{U}_\parallel^2$.
        """
        raise NotImplementedError

    @property
    def pow_gw_rate(self) -> FloatArr1D:
        r"""$\frac{1}{H_n R_*} \frac{d\mathcal{P}_\text{gw}}{d(H_n t)}$, growth rate of the GW power spectrum.

        This is the rate at which the GW power spectrum grows in flat space
        per Hubble time and per Hubble-scaled mean bubble spacing $H_n R_*$,
        as in :correia_2025:`\ ` fig. 5 and :gw_pt_ssm:`\ ` eq. 3.45.
        """
        raise NotImplementedError

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
        :param kwargs: arguments for :py:meth:`matplotlib.axes.Axes.plot`
        :return: the figure and the axes
        """
        return self._plot(self.pow_U_par, r"$\mathcal{P}_{U_{\parallel}}(kR_*)$", fig, ax, path, scale, **kwargs)

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
        :param kwargs: arguments for :py:meth:`matplotlib.axes.Axes.plot`
        :return: the figure and the axes
        """
        return self._plot(
            self.pow_gw_rate, r"$(H_n R_*)^{-1} \, d\mathcal{P}_\text{gw} / d(H_n t)$", fig, ax, path, scale, **kwargs)

    def _plot(
            self,
            data: FloatArr1D,
            ylabel: str,
            fig: plt.Figure | None,
            ax: plt.Axes | None,
            path: str | os.PathLike[str] | None,
            scale: float,
            **kwargs: tp.Any) -> "FigAndAxes":
        from pttools.analysis.utils import create_fig_ax  # noqa: PLC0415
        fig, ax = create_fig_ax(fig, ax)
        kwargs.setdefault("label", self.label_latex)
        ax.plot(self.y, scale * data, **kwargs)
        ax.set_xlabel(r"$z = kR_*$")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if path is not None:
            fig.savefig(path)
        return fig, ax
