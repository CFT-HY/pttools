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
from pttools.bubble.shock import solve_shock
from pttools.models import ConstCSModel
from pttools.ssmtools import Spectrum
from pttools.analysis.parallel import create_spectra
from pttools.analysis.utils import A4_PAPER_SIZE

logger = logging.getLogger(__name__)


def main():
    a_s = 5
    a_b = 1
    V_s = 1
    v_walls: np.ndarray = np.array([0.44, 0.56])
    alpha_ns: np.ndarray = np.array([0.07, 0.2])
    alpha_n_min = np.min(alpha_ns)

    allow_invalid = False
    models = [
        ConstCSModel(css2=1/3, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
        ConstCSModel(css2=1/3, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
        ConstCSModel(css2=1/4, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
        ConstCSModel(css2=1/4, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min, allow_invalid=allow_invalid),
    ]
    css2s = {model.css2 for model in models}
    csb2s = {model.csb2 for model in models}
    alpha_n_mins = np.array([model.alpha_n_min for model in models])
    if np.any(alpha_n_mins > alpha_n_min):
        msg = f"A model has alpha_n_min > {alpha_n_min}. Please adjust the models. Currently alpha_n_mins={alpha_n_mins}",
        logger.error(msg)
        raise ValueError(msg)

    spectra: np.ndarray = np.zeros((len(models), alpha_ns.size, v_walls.size), dtype=object)
    for i_model, model in enumerate(models):
        spectra[i_model, :, :] = create_spectra(
            model=model, v_walls=v_walls, alpha_ns=alpha_ns,
            # bubble_kwargs={"allow_invalid": False}, allow_bubble_failure=True
        )

    fig: plt.Figure = plt.figure(figsize=A4_PAPER_SIZE)
    fig2: plt.Figure = plt.figure(figsize=A4_PAPER_SIZE)
    axs: np.ndarray = fig.subplots(2, 2)
    axs2: np.ndarray = fig2.subplots(2, 2)
    for i_alpha_n, alpha_n in enumerate(alpha_ns):
        for i_v_wall, v_wall in enumerate(v_walls):
            ax: plt.Axes = axs[i_v_wall, i_alpha_n]
            ax2: plt.Axes = axs2[i_v_wall, i_alpha_n]
            for i_model, model in enumerate(models):
                spectrum: Spectrum = spectra[i_model, i_alpha_n, i_v_wall]
                if spectrum is not None:
                    ax.plot(spectrum.z, spectrum.pow_gw, label=model.label_latex)
                    ax2.plot(spectrum.bubble.xi, spectrum.bubble.v, label=model.label_latex)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("$z = kR*$")
            ax.set_ylabel(r"$\mathcal{P}_{\text{gw}}(z)$")
            title = rf"$\alpha_n={alpha_n}, v_\text{{wall}}={v_wall}$"
            ax.set_title(title)

            ax2.set_xlabel(r"$\xi$")
            ax2.set_ylabel(r"$v(\xi)$")
            ax2.set_title(title)

    # Mu curves
    for csb2 in csb2s:
        csb = np.sqrt(csb2)
        xi_mu: np.ndarray = np.linspace(csb, 1, 20)
        v_mu = lorentz(xi=xi_mu, v=csb)
        for ax in axs2.flat:
            ax.plot(xi_mu, v_mu, ls=":", c="k")

    for ax in axs2.flat:
        ax.set_xlim(0.3, 0.7)
        ax.set_ylim(0, 0.4)

    # Shock surfaces
    n_xi = 20
    for model in models:
        xi_arr: np.ndarray = np.linspace(model.css, 0.99, n_xi)
        for i_alpha_n, alpha_n in enumerate(alpha_ns):
            vm_arr: np.ndarray = np.zeros_like(xi_arr)
            for i_xi, xi in enumerate(xi_arr):
                wn = model.w_n(alpha_n=alpha_n)
                vm_tilde, wm = solve_shock(model, v1_tilde=xi, w1=wn, backwards=True, warn_if_barely_exists=False)
                vm_arr[i_xi] = lorentz(xi, vm_tilde)

            filter_arr = np.logical_or(vm_arr > 1, vm_arr <= 0)
            # Do not filter the first point
            filter_arr[0] = 0
            vm_arr[filter_arr] = np.nan
            for i_v_wall, v_wall in enumerate(v_walls):
                ax = axs2[i_v_wall, i_alpha_n]
                ax.plot(xi_arr, vm_arr, color="k")

    for ax in axs.flat:
        ax.legend()
    for ax in axs2.flat:
        ax.legend()

    fig.tight_layout()
    fig2.tight_layout()
    utils.save_and_show(fig, "const_cs_gw.png")
    utils.save_and_show(fig2, "const_cs_gw_v.png")


if __name__ == "__main__":
    fig = main()
