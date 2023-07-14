r"""Script for generating a figure of the three different types of relativistic combustion

Original version was developed by Daniel Cutting for the figure 14 of :notes:`\ `.
"""

import typing as tp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
from matplotlib.image import AxesImage
import numpy as np
from scipy.interpolate import interp1d

from pttools import bubble

_VIRIDIS_BIG = plt.colormaps["autumn_r"]
_NEW_COLORS = _VIRIDIS_BIG(np.linspace(0, 1, 256))
_NEW_COLORS[0] = matplotlib.colors.to_rgba("white", alpha=0)
COLORMAP = matplotlib.colors.ListedColormap(_NEW_COLORS)


def setup_matplotlib(**kwargs):
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 28,
        "legend.fontsize": 14,
        "lines.linewidth": 1.75,
        **kwargs
    })


def plot_bubble(ax: plt.Axes, label: str, v_wall: float, alpha: float, n_xi: int) -> AxesImage:
    v_f, enthalp, xi = bubble.fluid_shell_bag(v_wall, alpha, n_xi)
    n_wall = bubble.find_v_index(xi, v_wall)
    v_fluid = interp1d(xi, v_f, fill_value=0, bounds_error=False)

    xi_wall = xi[n_wall]
    xvalues = np.linspace(-1.5 * xi_wall, 1.5 * xi_wall, num=4000)
    yvalues = np.linspace(-1.5 * xi_wall, 1.5 * xi_wall, num=4000)
    xxgrid, yygrid = np.meshgrid(xvalues, yvalues)

    fluid_grid = v_fluid(np.sqrt(xxgrid * xxgrid + yygrid * yygrid))
    fluid_grid = fluid_grid / np.max(fluid_grid)
    arrow_width = 0.03 * xi_wall

    cs = ax.imshow(
        fluid_grid,
        cmap=COLORMAP,
        extent=(-1.5 * xi_wall, 1.5 * xi_wall, -1.5 * xi_wall, 1.5 * xi_wall),
        interpolation="bilinear")
    circle = plt.Circle((0, 0), xi_wall, color="k", linewidth=4, fill=None)
    ax.arrow(
        0.75 * xi_wall, 0, 0.5 * xi_wall, 0,
        shape="full", width=arrow_width, edgecolor="k", facecolor="k")
    ax.arrow(
        0. * xi_wall, 0.75 * xi_wall, 0, 0.5 * xi_wall,
        shape="full", width=arrow_width, edgecolor="k", facecolor="k")
    ax.arrow(
        -0.75 * xi_wall, 0, -0.5 * xi_wall, 0,
        shape="full", width=arrow_width, edgecolor="k", facecolor="k")
    ax.arrow(
        0. * xi_wall, -0.75 * xi_wall, 0, -0.5 * xi_wall,
        shape="full", width=arrow_width, edgecolor="k", facecolor="k")
    ax.arrow(
        (0.75 * xi_wall) / np.sqrt(2),
        (0.75 * xi_wall) / np.sqrt(2),
        (0.5 * xi_wall) / np.sqrt(2),
        (0.5 * xi_wall) / np.sqrt(2),
        shape="full", width=arrow_width, edgecolor="k", facecolor="k")
    ax.arrow(
        -(0.75 * xi_wall) / np.sqrt(2),
        (0.75 * xi_wall) / np.sqrt(2),
        -(0.5 * xi_wall) / np.sqrt(2),
        (0.5 * xi_wall) / np.sqrt(2),
        shape="full", width=arrow_width, edgecolor="k", facecolor="k")
    ax.arrow(
        (0.75 * xi_wall) / np.sqrt(2),
        -(0.75 * xi_wall) / np.sqrt(2),
        (0.5 * xi_wall) / np.sqrt(2),
        -(0.5 * xi_wall) / np.sqrt(2),
        shape="full", width=arrow_width, edgecolor="k", facecolor="k")
    ax.arrow(
        -(0.75 * xi_wall) / np.sqrt(2),
        -(0.75 * xi_wall) / np.sqrt(2),
        -(0.5 * xi_wall) / np.sqrt(2),
        -(0.5 * xi_wall) / np.sqrt(2),
        shape="full", width=arrow_width, edgecolor="k", facecolor="k")

    ax.add_artist(circle)
    ax.axis("off")
    ax.annotate(label, (0.51, -0.1), xycoords="axes fraction", ha="center", va="center", fontsize=30)
    return cs


def main(
        alpha: float = 0.5,
        v_walls: tp.Tuple[float, ...] = (0.44, 0.72, 0.92),
        plot_cbars: tp.Tuple[bool, ...] = (False, False, True),
        n_xi: int = 5000,
        figsize: tp.Tuple[int, int] = (27, 9)) -> plt.Figure:
    setup_matplotlib()

    fig: plt.Figure
    axs: np.ndarray
    fig, axs = plt.subplots(1, len(v_walls), figsize=figsize)

    labels = [
        "subsonic deflagration" + "\n" + r"$v_\mathrm{w} \leq c_s$",
        "supersonic deflagration" + "\n" + r"$c_s<v_\mathrm{w} < c_\mathrm{J}$",
        "detonation" + "\n" + r"$c_s<c_\mathrm{J}\leq v_\mathrm{w}$"
    ]

    for ax, label, v_wall, plot_cbar in zip(axs, labels, v_walls, plot_cbars):
        cs = plot_bubble(ax, label, v_wall, alpha, n_xi)
        if plot_cbar:
            cbar = fig.colorbar(cs, ax=axs)
            cbar.set_label(r"$v/v_\mathrm{peak}$")

    # fig.savefig("plots/all_circle.pdf", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
