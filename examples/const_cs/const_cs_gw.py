"""
GW spectra for ConstCSModel
===========================

Plot GW spectra for various ConstCSModels

These figures and this table are used in Mika's M.Sc. thesis.
"""

import io
import logging
import os.path
import time
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from examples import utils
from pttools.bubble import lorentz
from pttools.bubble.shock import shock_curve
from pttools.models import ConstCSModel, Model
from pttools.omgw0 import Spectrum, omega_ins
from pttools.analysis.parallel import create_spectra
# from pttools.analysis.utils import A3_PAPER_SIZE, A4_PAPER_SIZE
from pttools.speedup.options import IS_READ_THE_DOCS

logger = logging.getLogger(__name__)


def gw_lines(axs: tp.Iterable[plt.Axes]):
    pow_low = 9
    k_low = np.logspace(-1, -0.2, 10)
    p_low = k_low ** pow_low * 10 ** (-3.5)
    for ax in axs:
        ax.plot(k_low, p_low, color="k")
        ax.text(0.25, 10 ** (-6), f"$k^{pow_low}$")

    pow_high = -3
    k_high = np.logspace(1, 3, 10)
    p_high = k_high ** pow_high * 10 ** (-3.3)
    for ax in axs:
        ax.plot(k_high, p_high, color="k")
        ax.text(5.3, 10 ** (-7.5), f"$k^{{{pow_high}}}$")


def mu_curves(axs: tp.Iterable[plt.Axes], csb2s: tp.Iterable[float], ls: str = ":", c: str = "k"):
    for i_csb2, csb2 in enumerate(csb2s):
        csb = np.sqrt(csb2)
        xi_mu: np.ndarray = np.linspace(csb, 1, 20)
        v_mu = lorentz(xi=xi_mu, v=csb)
        for ax in axs:
            if i_csb2:
                ax.plot(xi_mu, v_mu, ls=ls, c=c)
            else:
                ax.plot(xi_mu, v_mu, ls=ls, c=c, label=r"$\mu(\xi, c_{s,b})$")


def plot_spectrum(
        spectrum: Spectrum,
        ax_v: plt.Axes, ax_gw: plt.Axes, ax_omgw0: plt.Axes,
        ls_v: str, label: str, label_omgw0: str):
    ax_v.plot(spectrum.bubble.xi, spectrum.bubble.v, label=label, ls=ls_v)
    ax_gw.plot(spectrum.y, spectrum.pow_gw, label=label)
    ax_omgw0.plot(spectrum.f(), spectrum.omgw0(), label=label_omgw0)


def snr_table(snrs: np.ndarray, models: tp.Iterable[Model], v_walls: np.ndarray, alpha_ns: np.ndarray) -> str:
    """Save the signal-to-noise ratios in a LaTeX table"""
    file: io.StringIO
    with io.StringIO() as file:
        file.writelines([
            "\\begin{table}\n",
            "\\centering\n",
            "\\caption{Signal-to-noise ratios of the gravitational wave power spectra of fig \\ref{fig:omgw0}}\n",
            "\\begin{tabular}{l|l|l|l}\n",
            "Model & \\multicolumn{3}{l}{$\\v_\\text{wall}$} \\\\\n"
            "& " + " & ".join([f"{v_wall:.2f}" for v_wall in v_walls]) + "\n",
            "\\hline \\\\\n"
        ])
        for i_alpha_n, alpha_n in enumerate(alpha_ns):
            for i_model, model in enumerate(models):
                file.write(
                    model.label_latex_params + " & " + \
                    " & ".join([f"{snr:.1f}" for snr in snrs[i_alpha_n, :, i_model]]))
                if i_model < len(models) - 1:
                    file.write(" \\\\\n")
            if i_alpha_n < len(alpha_ns) - 1:
                file.write(" \\hline \\\\\n")
            else:
                file.write(" \\\\\n")
        file.writelines([
            "\\end{tabular}\n",
            "\\label{table:const_cs_gw_snr}\n",
            "\\end{table}\n"
        ])
        return file.getvalue()


def setup_axes(
        spectrum: Spectrum,
        ax_v: plt.Axes, ax_gw: plt.Axes, ax_omgw0: plt.Axes,
        y_min: float, y_max: float,
        f_min: float, f_max: float):
    title = rf"$\alpha_n={spectrum.bubble.alpha_n}, v_\text{{wall}}={spectrum.bubble.v_wall}$"
    title_omgw0 = title[:-1] + rf", r_*={spectrum.r_star}, T_n={spectrum.Tn} \text{{GeV}}$"

    ax_v.set_xlim(0.25, 0.95)
    ax_v.set_ylim(0, 0.6)
    ax_v.legend(loc="upper left")
    ax_v.set_xlabel(r"$\xi$")
    ax_v.set_ylabel(r"$v(\xi)$")
    ax_v.grid()
    ax_v.set_title(title)

    for ax in (ax_gw, ax_omgw0):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()

    ax_gw.set_ylim(y_min, y_max)
    ax_gw.set_ylim(10e-12, 10e-4)
    ax_gw.legend(loc="lower center")
    ax_gw.set_xlabel("$z = kR*$")
    ax_gw.set_ylabel(r"$\mathcal{P}_{\text{gw}}(z)$")
    ax_gw.set_title(title)

    ax_omgw0.set_xlim(f_min, f_max)
    ax_omgw0.set_ylim(1e-19, 1e-7)
    ax_omgw0.legend(loc="lower left")
    ax_omgw0.set_xlabel(r"$f(\text{Hz})$")
    ax_omgw0.set_ylabel(r"$\Omega$")
    ax_omgw0.set_title(title_omgw0)

    return title, title_omgw0


def main():
    start_time = time.perf_counter()
    a_s = 5
    a_b = 1
    V_s = 1
    r_star = 0.1
    Tn = 200
    # v_walls: np.ndarray = np.array([0.4, 0.7, 0.8])
    # v_walls: np.ndarray = np.array([0.4, 0.67, 0.84])
    v_walls: np.ndarray[tuple[int], np.float64] = np.array([0.3, 0.68, 0.9])
    alpha_ns: np.ndarray[tuple[int], np.float64] = np.array([0.1, 0.2])
    alpha_n_min = np.min(alpha_ns)

    allow_invalid = False
    models = [
        ConstCSModel(css2=1/3, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
        ConstCSModel(css2=1/3, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
        ConstCSModel(css2=1/4, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
        ConstCSModel(css2=1/4, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
    ]
    lss = ["solid", "dashed", "dotted", "dashdot"]
    # css2s = {model.css2 for model in models}
    csb2s = {model.csb2 for model in models}
    alpha_n_mins = np.array([model.alpha_n_min for model in models])
    if np.any(alpha_n_mins > alpha_n_min):
        msg = f"A model has alpha_n_min > {alpha_n_min}. Please adjust the models. Currently alpha_n_mins={alpha_n_mins}",
        logger.error(msg)
        raise ValueError(msg)

    spectra: np.ndarray[tuple[int, int, int], Spectrum] = np.zeros((len(models), alpha_ns.size, v_walls.size), dtype=object)
    # z = np.logspace(-1, 3, 5000)

    for i_model, model in enumerate(models):
        spectra[i_model, :, :] = create_spectra(
            model=model, v_walls=v_walls, alpha_ns=alpha_ns,
            # spectrum_kwargs={"source_duration": 1},
            spectrum_kwargs={
                "r_star": r_star,
                # "z": z
                "Tn": Tn,
                # "g_star": 100,
                # "gs_star": 100
            },
            # bubble_kwargs={"allow_invalid": False}, allow_bubble_failure=True,
            # This fixes a BrokenProcessPool error on Read the Docs
            single_thread=IS_READ_THE_DOCS
        )

    figsize = (12, 10)
    figsize2 = (12, 5)
    figs: np.ndarray[tuple[int], plt.Figure] = np.array([plt.figure(figsize=figsize) for _ in range(3)])
    figs2: np.ndarray[tuple[int, int], plt.Figure] = np.array([[plt.figure(figsize=figsize2) for _ in alpha_ns] for _ in range(3)])
    axs: np.ndarray[tuple[int, int, int], plt.Axes] = np.stack([fig.subplots(alpha_ns.size, v_walls.size) for fig in figs])
    axs2: np.ndarray[tuple[int, int, int], plt.Axes] = np.stack([np.stack([fig.subplots(1, v_walls.size) for fig in figs2_row]) for figs2_row in figs2])

    snrs = np.zeros((len(alpha_ns), len(v_walls), len(models)))
    for i_alpha_n, alpha_n in enumerate(alpha_ns):
        for i_v_wall, v_wall in enumerate(v_walls):
            ax_v: plt.Axes = axs[0, i_alpha_n, i_v_wall]
            ax_gw: plt.Axes = axs[1, i_alpha_n, i_v_wall]
            ax_omgw0: plt.Axes = axs[2, i_alpha_n, i_v_wall]
            ax_v2: plt.Axes = axs2[0, i_alpha_n, i_v_wall]
            ax_gw2: plt.Axes = axs2[1, i_alpha_n, i_v_wall]
            ax_omgw02: plt.Axes = axs2[2, i_alpha_n, i_v_wall]
            for i_model, model in enumerate(models):
                spectrum: Spectrum = spectra[i_model, i_alpha_n, i_v_wall]
                if spectrum is not None:
                    label = model.label_latex_params
                    snr = spectrum.signal_to_noise_ratio_instrument()
                    snrs[i_alpha_n, i_v_wall, i_model] = snr
                    label_omgw0 = f"{label[:-1]}, SNR={snr:.1f}$"
                    ls = lss[i_model]
                    plot_spectrum(spectrum, ax_v, ax_gw, ax_omgw0, ls, label, label_omgw0)
                    plot_spectrum(spectrum, ax_v2, ax_gw2, ax_omgw02, ls, label, label_omgw0)

    table = snr_table(snrs, models, v_walls, alpha_ns)

    # Shock surfaces
    n_xi = 20
    ls = "--"
    for i_model, model in enumerate(models):
        xi_arr: np.ndarray = np.linspace(model.css, 0.99, n_xi)
        for i_alpha_n, alpha_n in enumerate(alpha_ns):
            vm_arr = shock_curve(model, alpha_n, xi_arr)
            for i_v_wall, v_wall in enumerate(v_walls):
                ax: plt.Axes = axs[0, i_alpha_n, i_v_wall]
                ax2: plt.Axes = axs2[0, i_alpha_n, i_v_wall]
                if i_model:
                    ax.plot(xi_arr, vm_arr, color="k", ls=ls)
                    ax2.plot(xi_arr, vm_arr, color="k", ls=ls)
                else:
                    ax.plot(xi_arr, vm_arr, color="k", ls=ls, label=r"$v_{sh}(\xi, c_{s,s})$")
                    ax2.plot(xi_arr, vm_arr, color="k", ls=ls, label=r"$v_{sh}(\xi, c_{s,s})$")

    mu_curves(np.concatenate((axs[0].flat, axs2[0].flat)), csb2s)

    # Noise curves
    y_min = np.min([spectrum.y[0] for spectrum in spectra.flat])
    y_max = np.min([spectrum.y[-1] for spectrum in spectra.flat])
    f_min = np.min([spectrum.f(z=spectrum.y[0]) for spectrum in spectra.flat])
    f_max = np.max([spectrum.f(z=spectrum.y[-1]) for spectrum in spectra.flat])
    f: np.ndarray = np.logspace(np.log10(f_min), np.log10(f_max), num=50)
    for i_alpha_n, alpha_n in enumerate(alpha_ns):
        for i_v_wall, v_wall in enumerate(v_walls):
            om_ins = omega_ins(f)
            ax: plt.Axes = axs[2, i_alpha_n, i_v_wall]
            ax2: plt.Axes = axs2[2, i_alpha_n, i_v_wall]
            ax.plot(f, om_ins, label="LISA instrument noise")
            ax2.plot(f, om_ins, label="LISA instrument noise")

    gw_lines(np.concatenate((axs[1].flat, axs2[1].flat)))

    # This must be after all the curves so that they are included in the legends.
    for i_alpha_n in range(alpha_ns.size):
        for i_v_wall in range(v_walls.size):
            setup_axes(
                spectrum=spectra[0, i_alpha_n, i_v_wall],
                ax_v=axs[0, i_alpha_n, i_v_wall],
                ax_gw=axs[1, i_alpha_n, i_v_wall],
                ax_omgw0=axs[2, i_alpha_n, i_v_wall],
                y_min=y_min, y_max=y_max,
                f_min=f_min, f_max=f_max
            )
            setup_axes(
                spectrum=spectra[0, i_alpha_n, i_v_wall],
                ax_v=axs2[0, i_alpha_n, i_v_wall],
                ax_gw=axs2[1, i_alpha_n, i_v_wall],
                ax_omgw0=axs2[2, i_alpha_n, i_v_wall],
                y_min=y_min, y_max=y_max,
                f_min=f_min, f_max=f_max
            )

    for fig in figs:
        fig.tight_layout()
    for fig in figs2.flat:
        fig.tight_layout()

    msg = f"Generating the figures took {time.perf_counter() - start_time} s"
    logger.debug(msg)
    print(msg)
    return figs, figs2, table


if __name__ == "__main__":
    figs, figs2, table2 = main()
    utils.save(figs[0], "const_cs_gw_v")
    utils.save(figs[1], "const_cs_gw")
    utils.save(figs[2], "const_cs_gw_omgw0")
    utils.save(figs2[0][0], "const_cs_gw_v_1")
    utils.save(figs2[0][1], "const_cs_gw_v_2")
    utils.save(figs2[1][0], "const_cs_gw_1")
    utils.save(figs2[1][1], "const_cs_gw_2")
    utils.save(figs2[2][0], "const_cs_gw_omgw0_1")
    utils.save(figs2[2][1], "const_cs_gw_omgw0_2")
    with open(os.path.join(utils.FIG_DIR, "const_cs_gw_snr.tex"), "w") as table_file:
        table_file.write(table2)
    plt.show()
