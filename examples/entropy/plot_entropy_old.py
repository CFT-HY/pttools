"""
Entropy (old reference)
=======================

Created on Fri Jul  2 18:20:37 2021

@author: hindmars

Requires input data as an npz file.
"""

import logging
import os.path
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from examples.utils import FIG_DIR
from pttools.analysis.plot_entropy_grid import plot_entropy_data
from pttools.bubble.chapman_jouguet import v_chapman_jouguet_bag
from pttools.bubble.alpha import alpha_n_max_bag, alpha_n_max_detonation_bag

logger = logging.getLogger(__name__)


def load(n_alpha: int = 10, n_vw: int = 10, g_bro: int = 120, g_sym: int = 123) \
        -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    file_name = f"s_change_gbro{g_bro:3.0f}_g_sym{g_sym:3.0f}_nalpha_{n_alpha}_nvw_{n_vw}.npz"
    d = np.load(file_name)

    ds_arr = d["arr_0"]
    vw_arr = d["arr_1"]
    alpha_arr = d["arr_2"]

    return ds_arr, vw_arr, alpha_arr


def main(n_alpha: int = 10, n_vw: int = 10, g_bro: int = 120, g_sym: int = 123, path: str = None):
    # g_bro = eos.G_BRO_DEFAULT*0.5
    # g_sym = eos.G_SYM_DEFAULT

    try:
        ds_arr, vw_arr, alpha_arr = load(n_alpha, n_vw, g_bro, g_sym)
    except FileNotFoundError as e:
        msg = f"The entropy data file was not found: {e}"
        logger.error(msg)
        return

    fig, ax = plot_entropy_data(
        ds_arr,
        v_walls=vw_arr,
        alpha_ns=alpha_arr,
        min_level=-0.3,
        max_level=0.4,
        diff_level=0.05
    )

    # ax.plot(b.min_speed_deton(alpha_arr), alpha_arr, 'k--', label=r'$v_{\rm J}$')
    ax.plot(v_chapman_jouguet_bag(alpha_arr), alpha_arr, 'k--', label=r'$v_{\rm J}$')
    # ax.plot(vw_arr, b.alpha_n_max(vw_arr), 'k', label=r'$\alpha_{\rm max}$', linewidth=2)

    ax.plot(vw_arr, alpha_n_max_bag(vw_arr), label=r"$\alpha_{n,max}$")
    ax.plot(vw_arr, alpha_n_max_detonation_bag(vw_arr), label=r"$\alpha_{n,max,det}$")

    ax.legend()

    if path is not None:
        fig.savefig(path)
    return fig


if __name__ == "__main__":
    main(path=os.path.join(FIG_DIR, "entropy_old.png"))
    plt.show()
