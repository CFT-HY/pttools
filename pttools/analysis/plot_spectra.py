"""Utilities for plotting the spectra of multiple bubbles"""

import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.plot_bubbles import plot_bubbles_v
from pttools.analysis.utils import FigAndAxes, create_fig_ax, legend
from pttools.ssm.spectrum import SSMSpectrum
from pttools.omgw0 import Spectrum, omega_noise

F_LABEL = r"$f$ (Hz)"
SPEC_DEN_V_LABEL = r"$\mathcal{P}_{v}(kR_*)$"
SPEC_DEN_GW_LABEL = r"$\mathcal{P}_{gw}(kR_*)$"
POW_V_LABEL = r"$\mathcal{P}_{\tilde{v}}(kR_*)$"
POW_GW_LABEL = r"$\mathcal{P}_{\tilde{gw}}(kR_*)$"
OMGW0_LABEL = r"$\Omega_{gw,0}$"
Z_LABEL = r"$z = kR_*$"


# -----
# Common plotting functions
# -----

def plot_spectra_common(
        spectra: tp.Collection[SSMSpectrum],
        fig: plt.Figure,
        ax: plt.Axes,
        path: str | None = None,
        set_x: bool = True) -> FigAndAxes:
    """Common steps for plotting spectra"""
    if set_x:
        ax.set_xlabel(Z_LABEL)
        ax.set_xscale("log")
        ax.set_xlim(
            np.nanmin([np.min(spectrum.y) for spectrum in spectra]),
            np.nanmax([np.max(spectrum.y) for spectrum in spectra])
        )
    ax.set_yscale("log")
    ax.grid()
    if len(spectra) > 1:
        legend(ax, loc="lower left")
    if path is not None:
        fig.savefig(path)
    return fig, ax


def plot_spectra_multi(
        spectra: tp.Collection[Spectrum],
        fig: plt.Figure | None = None,
        path: str | None = None,
        **kwargs) -> tuple[plt.Figure, np.ndarray[tuple[int, int], plt.Axes]]:
    """Plot multiple types of spectra"""
    fig, axs = plot_spectra_multi_common(spectra, fig, figsize=(7, 5), nrows=2, ncols=2, **kwargs)

    arrowprops = {"width": 7}
    x_left = 0.48
    x_right = 0.54
    y_top = 0.68
    y_bottom = 0.4
    axs[0, 0].annotate("", xytext=(x_left, y_top), xy=(x_right, y_top), xycoords="figure fraction", arrowprops=arrowprops)
    axs[0, 0].annotate("", xytext=(0.56, 0.56), xy=(x_left, 0.48), xycoords="figure fraction", arrowprops=arrowprops)
    axs[0, 0].annotate("", xytext=(x_left, y_bottom), xy=(x_right, y_bottom), xycoords="figure fraction", arrowprops=arrowprops)

    if path is not None:
        fig.savefig(path)
    return fig, axs


def plot_spectra_multi_common(
        spectra: tp.Collection[Spectrum],
        fig: plt.Figure,
        figsize: tuple[float, float],
        nrows: int,
        ncols: int,
        **kwargs):
    if fig is None:
        fig = plt.figure(figsize=figsize)
    axs = fig.subplots(nrows, ncols)
    flat = axs.flat
    plot_bubbles_v([spectrum.bubble for spectrum in spectra], fig, flat[0], **kwargs)
    plot_spectra_v(spectra, fig=fig, ax=flat[1], **kwargs)
    plot_spectra_gw(spectra, fig=fig, ax=flat[2], **kwargs)
    plot_spectra(spectra, fig=fig, ax=flat[3], **kwargs)
    flat[0].set_title("Fluid velocity profile")
    flat[1].set_title("Power spectrum of the velocity field")
    flat[2].set_title("GW power spectrum")
    flat[3].set_title("GW power spectrum today")
    fig.tight_layout()
    return fig, axs


def plot_spectra_multi_flat(
        spectra: tp.Collection[Spectrum],
        fig: plt.Figure,
        path: str | None = None,
        **kwargs):
    fig, axs = plot_spectra_multi_common(spectra, fig, figsize=(14, 4), nrows=1, ncols=4, **kwargs)
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
    return fig, axs


# -----
# Individual plotting functions
# -----

def plot_spectra(
        spectra: tp.Collection[Spectrum],
        ax: plt.Axes | None = None,
        fig: plt.Figure | None = None,
        path: str | None = None,
        **kwargs) -> FigAndAxes:
    f"""Plot the GW spectra today {OMGW0_LABEL}"""
    fig, ax = create_fig_ax(fig, ax)
    for spectrum in spectra:
        snr = spectrum.signal_to_noise_ratio()
        ax.plot(spectrum.f(), spectrum.omgw0(), label=f"{spectrum.label_latex[:-1]}, SNR={snr:.2f}$", **kwargs)
    f_min = np.nanmin([np.nanmin(spectrum.f()) for spectrum in spectra])
    f_max = np.nanmax([np.nanmax(spectrum.f()) for spectrum in spectra])
    f_noise: np.ndarray[tuple[int], np.float64] = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    ax.plot(f_noise, omega_noise(f_noise), label=r"LISA noise")
    ax.set_xlabel(F_LABEL)
    ax.set_xscale("log")
    ax.set_xlim(f_min, f_max)
    ax.set_ylabel(OMGW0_LABEL)
    return plot_spectra_common(spectra, fig, ax, path, set_x=False)


def plot_spectra_gw(
        spectra: tp.Collection[SSMSpectrum],
        ax: plt.Axes | None = None,
        fig: plt.Figure | None = None,
        path: str | None = None,
        **kwargs) -> FigAndAxes:
    f"""Plot the GW power spectra {POW_GW_LABEL}"""
    fig, ax = create_fig_ax(fig, ax)
    for spectrum in spectra:
        ax.plot(spectrum.y, spectrum.pow_gw, label=spectrum.label_latex, **kwargs)
    ax.set_ylabel(POW_GW_LABEL)
    return plot_spectra_common(spectra, fig, ax, path, **kwargs)


def plot_spectra_v(
        spectra: tp.Collection[SSMSpectrum],
        ax: plt.Axes | None = None,
        fig: plt.Figure | None = None,
        path: str | None = None,
        **kwargs) -> FigAndAxes:
    f"""Plot the velocity power spectra {POW_V_LABEL}"""
    fig, ax = create_fig_ax(fig, ax)
    for spectrum in spectra:
        ax.plot(spectrum.y, spectrum.pow_v, label=spectrum.label_latex, **kwargs)
    ax.set_ylabel(POW_V_LABEL)
    return plot_spectra_common(spectra, fig, ax, path)


def plot_spectra_spec_den_gw(
        spectra: tp.Collection[SSMSpectrum],
        ax: plt.Axes | None = None,
        fig: plt.Figure | None = None,
        path: str | None = None,
        **kwargs) -> FigAndAxes:
    f"""Plot the GW spectral densities {SPEC_DEN_GW_LABEL}"""
    fig, ax = create_fig_ax(fig, ax)
    for spectrum in spectra:
        ax.plot(spectrum.y, spectrum.spec_den_gw, label=spectrum.label_latex, **kwargs)
    ax.set_ylabel(SPEC_DEN_GW_LABEL)
    return plot_spectra_common(spectra, fig, ax, path)


def plot_spectra_spec_den_v(
        spectra: tp.Collection[SSMSpectrum],
        ax: plt.Axes | None = None,
        fig: plt.Figure | None = None,
        path: str | None = None,
        **kwargs) -> FigAndAxes:
    f"""Plot the velocity spectral densities {SPEC_DEN_V_LABEL}"""
    fig, ax = create_fig_ax(fig, ax)
    for spectrum in spectra:
        ax.plot(spectrum.y, spectrum.spec_den_v, label=spectrum.label_latex, **kwargs)
    ax.set_ylabel(SPEC_DEN_V_LABEL)
    return plot_spectra_common(spectra, fig, ax, path)
