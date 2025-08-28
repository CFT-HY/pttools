"""Utilties for plotting the spectrum of a single bubble"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.utils import FigAndAxes, create_fig_ax, legend
from pttools.analysis.plot_bubble import plot_bubble_v
from pttools.ssmtools.spectrum import SSMSpectrum
from pttools.omgw0 import Spectrum

SPEC_DEN_V_LABEL = r"$\mathcal{P}_{v}(kR_*)$"
SPEC_DEN_GW_LABEL = r"$\mathcal{P}_{gw}(kR_*)$"
POW_V_LABEL = r"$\mathcal{P}_{\tilde{v}}(kR_*)$"
POW_GW_LABEL = r"$\mathcal{P}_{\tilde{gw}}(kR_*)$"
OMGW0_LABEL = r"$\Omega_{gw,0}$"


def plot_spectrum(
        spectrum: SSMSpectrum,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        path: str | None = None,
        **kwargs) -> FigAndAxes:
    rf"""Plot the GW spectrum {POW_GW_LABEL} of a bubble"""
    fig, ax = create_fig_ax(fig, ax)
    ax.plot(spectrum.y, spectrum.pow_gw, **kwargs)
    ax.set_ylabel(POW_GW_LABEL)
    ax.set_title("GW power spectrum")
    return plot_spectrum_common(spectrum, fig, ax, path)


def plot_spectrum_common(
        spectrum: SSMSpectrum,
        fig: plt.Figure,
        ax: plt.Axes,
        path: str | None = None,
        x_is_z: bool = True) -> FigAndAxes:
    """Common steps for plotting a spectrum"""
    if x_is_z:
        ax.set_xlabel("$z$")
        ax.set_xlim(np.min(spectrum.y), np.max(spectrum.y))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    legend(ax)
    if path is not None:
        fig.savefig(path)
    return fig, ax


def plot_spectrum_multi(
        spectrum: Spectrum,
        fig: plt.Figure | None = None,
        path: str | None = None,
        **kwargs) -> plt.Figure:
    """Plot multiple types of spectra for a bubble"""
    if fig is None:
        fig = plt.figure(figsize=(7, 5))
    axs = fig.subplots(2, 2)
    plot_bubble_v(spectrum.bubble, fig, axs[0, 0], **kwargs)
    # plot_spectrum_spec_den_v(spectrum, fig, axs[0, 0], **kwargs)
    # plot_spectrum_spec_den_gw(spectrum, fig, axs[1, 0], **kwargs)
    plot_spectrum_v(spectrum, fig, axs[0, 1], **kwargs)
    plot_spectrum(spectrum, fig, axs[1, 0], **kwargs)
    plot_spectrum_omgw0(spectrum, fig, axs[1, 1], **kwargs)
    fig.tight_layout()

    arrowprops = {"width": 7}
    axs[0, 0].annotate("", xytext=(0.48, 0.68), xy=(0.54, 0.68), xycoords="figure fraction", arrowprops=arrowprops)
    axs[0, 0].annotate("", xytext=(0.58, 0.58), xy=(0.48, 0.48), xycoords="figure fraction", arrowprops=arrowprops)
    axs[0, 0].annotate("", xytext=(0.48, 0.4), xy=(0.54, 0.4), xycoords="figure fraction", arrowprops=arrowprops)

    if path is not None:
        fig.savefig(path)
    return fig


def plot_spectrum_multi_flat(
    spectrum: Spectrum,
    fig: plt.Figure | None = None,
    path: str | None = None,
    **kwargs) -> plt.Figure:
    if fig is None:
        fig = plt.figure(figsize=(14, 4))
    axs = fig.subplots(1, 4)
    plot_bubble_v(spectrum.bubble, fig, axs[0], **kwargs)
    # plot_spectrum_spec_den_v(spectrum, fig, axs[0, 0], **kwargs)
    # plot_spectrum_spec_den_gw(spectrum, fig, axs[1, 0], **kwargs)
    plot_spectrum_v(spectrum, fig, axs[1], **kwargs)
    plot_spectrum(spectrum, fig, axs[2], **kwargs)
    plot_spectrum_omgw0(spectrum, fig, axs[3], **kwargs)
    fig.tight_layout()

    arrowprops = {"width": 7}
    y = 0.4
    length = 0.03
    x1 = 0.24
    x2 = 0.49
    x3 = 0.75
    axs[0].annotate("", xytext=(0.24, y), xy=(x1 + length, y), xycoords="figure fraction", arrowprops=arrowprops)
    axs[0].annotate("", xytext=(0.49, y), xy=(x2 + length, y), xycoords="figure fraction", arrowprops=arrowprops)
    axs[0].annotate("", xytext=(0.75, y), xy=(x3 + length, y), xycoords="figure fraction", arrowprops=arrowprops)

    if path is not None:
        fig.savefig(path)
    return fig


def plot_spectrum_omgw0(
        spectrum: Spectrum,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        path: str | None = None,
        **kwargs) -> FigAndAxes:
    fig, ax = create_fig_ax(fig, ax)
    f = spectrum.f()
    ax.plot(f, spectrum.omgw0(), **kwargs)
    ax.set_ylabel(OMGW0_LABEL)
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_xlim(f.min(), f.max())
    ax.set_title(r"Power spectrum today")
    return plot_spectrum_common(spectrum, fig, ax, path, x_is_z=False)


def plot_spectrum_v(
        spectrum: SSMSpectrum,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        path: str | None = None,
        **kwargs) -> FigAndAxes:
    rf"""Plot the velocity power spectrum {POW_V_LABEL} of a bubble"""
    fig, ax = create_fig_ax(fig, ax)
    ax.plot(spectrum.y, spectrum.pow_v, **kwargs)
    ax.set_ylabel(POW_V_LABEL)
    ax.set_title(r"Power spectrum of the velocity field")
    return plot_spectrum_common(spectrum, fig, ax, path)


def plot_spectrum_spec_den_gw(
        spectrum: SSMSpectrum,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        path: str | None = None,
        **kwargs) -> FigAndAxes:
    rf"""Plot the spectral density of GWs {SPEC_DEN_GW_LABEL} for a bubble"""
    fig, ax = create_fig_ax(fig, ax)
    ax.plot(spectrum.y, spectrum.spec_den_gw, **kwargs)
    ax.set_ylabel(SPEC_DEN_GW_LABEL)
    return plot_spectrum_common(spectrum, fig, ax, path)


def plot_spectrum_spec_den_v(
        spectrum: SSMSpectrum,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        path: str | None = None,
        **kwargs) -> FigAndAxes:
    rf"""Plot the spectral density of velocity {SPEC_DEN_V_LABEL} for a bubble"""
    fig, ax = create_fig_ax(fig, ax)
    ax.plot(spectrum.y, spectrum.spec_den_v, **kwargs)
    ax.set_ylabel(SPEC_DEN_V_LABEL)
    return plot_spectrum_common(spectrum, fig, ax, path)
