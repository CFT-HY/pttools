"""
GW spectra for ConstCSModel
===========================

Plot GW spectra for various ConstCSModels
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from examples import utils
from pttools.bubble import lorentz
from pttools.bubble.shock import shock_curve
from pttools.models import ConstCSModel
from pttools.omgw0 import Spectrum, omega_noise, omega_ins
from pttools.analysis.parallel import create_spectra
from pttools.analysis.utils import A3_PAPER_SIZE

logger = logging.getLogger(__name__)


def main():
    a_s = 5
    a_b = 1
    V_s = 1
    r_star = 0.1
    v_walls: np.ndarray = np.array([0.44, 0.56, 0.92])
    alpha_ns: np.ndarray = np.array([0.07, 0.2])
    alpha_n_min = np.min(alpha_ns)

    allow_invalid = False
    models = [
        ConstCSModel(css2=1/3, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
        ConstCSModel(css2=1/3, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
        ConstCSModel(css2=1/4, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
        ConstCSModel(css2=1/4, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
    ]
    # css2s = {model.css2 for model in models}
    csb2s = {model.csb2 for model in models}
    alpha_n_mins = np.array([model.alpha_n_min for model in models])
    if np.any(alpha_n_mins > alpha_n_min):
        msg = f"A model has alpha_n_min > {alpha_n_min}. Please adjust the models. Currently alpha_n_mins={alpha_n_mins}",
        logger.error(msg)
        raise ValueError(msg)

    spectra: np.ndarray = np.zeros((len(models), alpha_ns.size, v_walls.size), dtype=object)
    # z = np.logspace(-1, 3, 5000)
    for i_model, model in enumerate(models):
        spectra[i_model, :, :] = create_spectra(
            model=model, v_walls=v_walls, alpha_ns=alpha_ns,
            # spectrum_kwargs={"source_duration": 1},
            spectrum_kwargs={
                "r_star": r_star
                # "z": z
            }
            # bubble_kwargs={"allow_invalid": False}, allow_bubble_failure=True
        )

    fig: plt.Figure = plt.figure(figsize=A3_PAPER_SIZE)
    fig2: plt.Figure = plt.figure(figsize=A3_PAPER_SIZE)
    fig3: plt.Figure = plt.figure(figsize=A3_PAPER_SIZE)
    axs: np.ndarray = fig.subplots(alpha_ns.size, v_walls.size)
    axs2: np.ndarray = fig2.subplots(alpha_ns.size, v_walls.size)
    axs3: np.ndarray = fig3.subplots(alpha_ns.size, v_walls.size)
    for i_alpha_n, alpha_n in enumerate(alpha_ns):
        for i_v_wall, v_wall in enumerate(v_walls):
            ax: plt.Axes = axs[i_alpha_n, i_v_wall]
            ax2: plt.Axes = axs2[i_alpha_n, i_v_wall]
            ax3: plt.Axes = axs3[i_alpha_n, i_v_wall]
            for i_model, model in enumerate(models):
                spectrum: Spectrum = spectra[i_model, i_alpha_n, i_v_wall]
                if spectrum is not None:
                    ax.plot(spectrum.y, spectrum.pow_gw, label=model.label_latex)
                    ax2.plot(spectrum.bubble.xi, spectrum.bubble.v, label=model.label_latex)
                    ax3.plot(spectrum.f(), spectrum.omgw0(), label=model.label_latex)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("$z = kR*$")
            ax.set_ylabel(r"$\mathcal{P}_{\text{gw}}(z)$")
            ax.grid()
            title = rf"$\alpha_n={alpha_n}, v_\text{{wall}}={v_wall}$"
            ax.set_title(title)

            ax2.set_xlabel(r"$\xi$")
            ax2.set_ylabel(r"$v(\xi)$")
            ax2.grid()
            ax2.set_title(title)

            ax3.set_xscale("log")
            ax3.set_yscale("log")
            ax3.set_xlabel(r"$f(\text{Hz})$")
            ax3.set_ylabel(r"$\Omega$")
            ax3.grid()
            ax3.set_title(title + rf", r_*={r_star}")

    # Mu curves
    for csb2 in csb2s:
        csb = np.sqrt(csb2)
        xi_mu: np.ndarray = np.linspace(csb, 1, 20)
        v_mu = lorentz(xi=xi_mu, v=csb)
        for ax in axs2.flat:
            ax.plot(xi_mu, v_mu, ls=":", c="k")

    for ax in axs2.flat:
        ax.set_xlim(0.4, 0.7)
        ax.set_ylim(0, 0.4)

    # Shock surfaces
    n_xi = 20
    for model in models:
        xi_arr: np.ndarray = np.linspace(model.css, 0.99, n_xi)
        for i_alpha_n, alpha_n in enumerate(alpha_ns):
            vm_arr = shock_curve(model, alpha_n, xi_arr)
            for i_v_wall, v_wall in enumerate(v_walls):
                ax = axs2[i_alpha_n, i_v_wall]
                ax.plot(xi_arr, vm_arr, color="k")

    # Noise curves
    f_min = np.min([spectrum.f(z=spectrum.y[0]) for spectrum in spectra.flat])
    f_max = np.max([spectrum.f(z=spectrum.y[-1]) for spectrum in spectra.flat])
    f: np.ndarray = np.logspace(np.log10(f_min), np.log10(f_max), num=50)
    for i_alpha_n, alpha_n in enumerate(alpha_ns):
        for i_v_wall, v_wall in enumerate(v_walls):
            ax = axs3[i_alpha_n, i_v_wall]
            ax.plot(f, omega_noise(f), label="LISA overall noise")
            ax.plot(f, omega_ins(f), label="LISA instrument noise")

    # Lines
    pow_low = 9
    k_low = np.logspace(-1, -0.2, 10)
    p_low = k_low**pow_low * 10**(-3.5)
    for ax in axs.flat:
        ax.plot(k_low, p_low, color="k")
        ax.text(0.25, 10**(-6), f"$k^{pow_low}$")

    pow_high = -3
    k_high = np.logspace(1, 3, 10)
    p_high = k_high**pow_high * 10**(-3)
    for ax in axs.flat:
        ax.plot(k_high, p_high, color="k")
        ax.text(5, 10**(-6.5), f"$k^{{{pow_high}}}$")

    for ax in axs.flat:
        ax.legend()
    for ax in axs2.flat:
        ax.legend()

    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    utils.save_and_show(fig, "const_cs_gw.png")
    utils.save_and_show(fig2, "const_cs_gw_v.png")
    utils.save_and_show(fig3, "const_cs_gw_omgw0.png")


if __name__ == "__main__":
    fig = main()
