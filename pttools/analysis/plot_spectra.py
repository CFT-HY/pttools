import typing as tp

import matplotlib.pyplot as plt

from pttools.ssmtools.spectrum import Spectrum


def plot_spectra(spectra: tp.List[Spectrum], fig: plt.Figure = None):
    if fig is None:
        fig = plt.figure(figsize=(11.7, 8.3))

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

    return fig
