import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.utils import A4_PAPER_SIZE, FigAndAxes, create_fig_ax
from pttools.ssmtools.spectrum import SSMSpectrum
from pttools.omgw0 import Spectrum, omega_noise


def plot_spectra_common(spectra: tp.List[SSMSpectrum], fig: plt.Figure, ax: plt.Axes, path: str = None, set_x: bool = True) -> FigAndAxes:
    if set_x:
        ax.set_xlabel("$z$")
        ax.set_xscale("log")
        ax.set_xlim(
            np.nanmin([np.min(spectrum.y) for spectrum in spectra]),
            np.nanmax([np.max(spectrum.y) for spectrum in spectra])
        )
    ax.set_yscale("log")
    ax.grid()
    if ax.get_legend_handles_labels() != ([], []):
        ax.legend(loc="lower left")
    if path is not None:
        fig.savefig(path)
    return fig, ax


def plot_spectra_omgw0(spectra: tp.List[Spectrum], ax: plt.Axes = None, fig: plt.Figure = None, path: str = None) -> FigAndAxes:
    fig, ax = create_fig_ax(fig, ax)
    for spectrum in spectra:
        snr = spectrum.signal_to_noise_ratio()
        ax.plot(spectrum.f(), spectrum.omgw0(), label=f"{spectrum.label_latex[:-1]}, SNR={snr:.2f}$")
    f_min = np.nanmin([np.nanmin(spectrum.f()) for spectrum in spectra])
    f_max = np.nanmax([np.nanmax(spectrum.f()) for spectrum in spectra])
    f_noise = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    ax.plot(f_noise, omega_noise(f_noise), label=r"LISA noise")
    ax.set_xlabel("$f$ (Hz)")
    ax.set_xscale("log")
    ax.set_xlim(f_min, f_max)
    ax.set_ylabel(r"$\Omega_{gw,0}$")
    return plot_spectra_common(spectra, fig, ax, path, set_x=False)


def plot_spectra_spec_den_v(spectra: tp.List[Spectrum], ax: plt.Axes = None, fig: plt.Figure = None, path: str = None) -> FigAndAxes:
    fig, ax = create_fig_ax(fig, ax)
    for spectrum in spectra:
        ax.plot(spectrum.f(), spectrum.spec_den_v)
    return plot_spectra_common(spectra, fig, ax, path)


def plot_spectra(spectra: tp.List[SSMSpectrum], fig: plt.Figure = None, path: str = None) -> plt.Figure:
    # Todo: fix the labels here
    if fig is None:
        fig = plt.figure(figsize=A4_PAPER_SIZE)

    axs = fig.subplots(2, 2)
    ax_spec_den_v = axs[0, 0]
    ax_pow_v = axs[0, 1]
    ax_spec_den_gw = axs[1, 0]
    ax_pow_gw = axs[1, 1]

    for spectrum in spectra:
        label = rf"{spectrum.bubble.model.label_latex}, $v_w={spectrum.bubble.v_wall}, \alpha_n={spectrum.bubble.alpha_n}$"
        ax_spec_den_v.plot(spectrum.y, spectrum.spec_den_v, label=label)
        ax_spec_den_gw.plot(spectrum.y, spectrum.spec_den_gw, label=label)
        ax_pow_v.plot(spectrum.y, spectrum.pow_v, label=label)
        ax_pow_gw.plot(spectrum.y, spectrum.pow_gw, label=label)

    for ax in axs.flatten():
        ax.set_xlabel("$kR_*$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.legend(fontsize=8)

    # Todo: check that the labels for gw are correct
    ax_spec_den_v.set_ylabel(r"$\mathcal{P}_{v}(kR_*)$")
    ax_spec_den_gw.set_ylabel(r"$\mathcal{P}_{gw}(kR_*)$")
    ax_pow_v.set_ylabel(r"$\mathcal{P}_{\tilde{v}}(kR_*)$")
    ax_pow_gw.set_ylabel(r"$\mathcal{P}_{\tilde{gw}}(kR_*)$")

    if path is not None:
        fig.savefig(path)
    return fig
