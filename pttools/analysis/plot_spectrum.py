import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.utils import A4_PAPER_SIZE, FigAndAxes, create_fig_ax
from pttools.ssmtools.spectrum import SSMSpectrum


# TODO: set proper labels for these

def plot_spectrum(
        spectrum: SSMSpectrum,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        path: str = None,
        **kwargs) -> FigAndAxes:
    fig, ax = create_fig_ax(fig, ax)
    ax.plot(spectrum.y, spectrum.pow_gw, **kwargs)
    ax.set_ylabel("pow_gw")
    return plot_spectrum_common(spectrum, fig, ax, path)


def plot_spectrum_common(spectrum: SSMSpectrum, fig: plt.Figure, ax: plt.Axes, path: str = None) -> FigAndAxes:
    ax.set_xlabel("$z$")
    ax.set_xlim(np.min(spectrum.y), np.max(spectrum.y))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    if path is not None:
        fig.savefig(path)
    return fig, ax


def plot_spectrum_multi(spectrum: SSMSpectrum, fig: plt.Figure = None, path: str = None, **kwargs) -> plt.Figure:
    if fig is None:
        fig = plt.figure(figsize=A4_PAPER_SIZE)
    axs = fig.subplots(2, 2)
    plot_spectrum_spec_den_v(spectrum, fig, axs[0, 0], **kwargs)
    plot_spectrum_spec_den_gw(spectrum, fig, axs[1, 0], **kwargs)
    plot_spectrum_v(spectrum, fig, axs[0, 1], **kwargs)
    plot_spectrum(spectrum, fig, axs[1, 1], **kwargs)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig


def plot_spectrum_v(
        spectrum: SSMSpectrum,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        path: str = None,
        **kwargs) -> FigAndAxes:
    fig, ax = create_fig_ax(fig, ax)
    ax.plot(spectrum.y, spectrum.pow_v, **kwargs)
    ax.set_ylabel("pow_v")
    return plot_spectrum_common(spectrum, fig, ax, path)


def plot_spectrum_spec_den_gw(
        spectrum: SSMSpectrum,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        path: str = None,
        **kwargs) -> FigAndAxes:
    fig, ax = create_fig_ax(fig, ax)
    ax.plot(spectrum.y, spectrum.spec_den_gw, **kwargs)
    ax.set_ylabel("spec_den_gw")
    return plot_spectrum_common(spectrum, fig, ax, path)


def plot_spectrum_spec_den_v(
        spectrum: SSMSpectrum,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        path: str = None,
        **kwargs) -> FigAndAxes:
    fig, ax = create_fig_ax(fig, ax)
    ax.plot(spectrum.y, spectrum.spec_den_v, **kwargs)
    ax.set_ylabel("spec_den_v")
    return plot_spectrum_common(spectrum, fig, ax, path)
