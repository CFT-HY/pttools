import os

import matplotlib.pyplot as plt

from pttools.analysis.utils import FigAndAxes, create_fig_ax, save_fig_multi
from pttools.speedup.threads import DEFAULT_VARYING_NUMBA_THREADS, time_with_varying_numba_threads
import pttools.type_hints as th


def plot_threads(
        name: str,
        n_threads: th.IntArr1D,
        times: th.FloatArr1D,
        n_iterations: int,
        fig: plt.Figure | None = None) -> FigAndAxes:
    """Plot the execution times for different thread counts"""
    if fig is None:
        fig = plt.figure()
    ax1, ax2 = fig.subplots(1, 2)

    for ax in (ax1, ax2):
        ax.plot(n_threads, times / n_iterations)
        ax.set_xlabel("# of threads")
        ax.set_ylabel("Wall time per run (s)")

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    fig.suptitle(name)
    fig.tight_layout()
    return fig, ax


def time_and_plot_threads(
        name: str,
        filename: str,
        path: str,
        stmt: str,
        setup: str,
        n_iterations: int,
        n_threads: th.IntArr1D = DEFAULT_VARYING_NUMBA_THREADS) -> FigAndAxes:
    """Plot and save to a file the execution times for different thread counts"""
    path2 = os.path.join(path, filename)
    with open(f"{path2}.txt", "w") as file:
        n_threads, times = time_with_varying_numba_threads(
            name=name, stmt=stmt, setup=setup, n_iterations=n_iterations, n_threads=n_threads, file=file
        )
    fig, ax = plot_threads(name=name, n_threads=n_threads, times=times, n_iterations=n_iterations)
    save_fig_multi(fig, path2)
    return fig, ax
