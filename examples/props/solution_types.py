r"""
Solution types
==============

Script for generating a figure of the three different types of relativistic combustion.
Original version was developed by Daniel Cutting for the figure 14 of :notes:`\ `.
"""

from __future__ import annotations

import typing as tp

from matplotlib import colormaps, rc_context
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import numpy as np
from scipy.interpolate import interp1d

from examples.utils import save_fig
from pttools import bubble
from pttools.plot.utils import show_fig
import pttools.type_hints as th
from pttools.utils.time import log_duration

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.image import AxesImage

__author__ = "Daniel Cutting"

_VIRIDIS_BIG = colormaps["autumn_r"]
_NEW_COLORS = _VIRIDIS_BIG(np.linspace(0, 1, 256))
_NEW_COLORS[0] = to_rgba("white", alpha=0)
COLORMAP: ListedColormap = ListedColormap(_NEW_COLORS)
#: Matplotlib settings of the figure
RC_PARAMS: dict[str, tp.Any] = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 28,
    "legend.fontsize": 14,
    "lines.linewidth": 1.75,
}


def plot_bubble(ax: Axes, label: str, v_wall: float, alpha: float, n_xi: int) -> AxesImage:
    """Plot the fluid velocity of a bubble as a 2D color map with arrows for the direction of the wall"""
    v_f, _enthalp, xi = bubble.sound_shell_bag(
        v_wall=v_wall, alpha_n=alpha,
        df_dtau_ptr=bubble.DF_DTAU_PTR_BAG,
        ode_method=bubble.DEFAULT_FLUID_INTEGRATE_METHOD, cs2_ptr=bubble.CS2_BAG_SCALAR_PTR, n_xi=n_xi
    )
    n_wall = bubble.find_v_index(xi, v_wall)
    v_fluid = interp1d(xi, v_f, fill_value=0, bounds_error=False)

    xi_wall: float = xi[n_wall]
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
    circle = Circle((0, 0), xi_wall, color="k", linewidth=4, fill=None)
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


@log_duration
def main(
        alpha: float = 0.5,
        v_walls: tuple[float, ...] = (0.44, 0.72, 0.92),
        plot_cbars: tuple[bool, ...] = (False, False, True),
        n_xi: int = 5000,
        figsize: tuple[int, int] = (27, 9),
        path: str | None = None) -> Figure:
    """Plot the three solution types: subsonic deflagration, supersonic deflagration and detonation"""
    with rc_context(RC_PARAMS):
        fig: Figure
        axs: np.ndarray
        fig = Figure(figsize=figsize)
        axs = fig.subplots(1, len(v_walls))

        labels = [
            "subsonic deflagration\n$v_\\mathrm{wall} \\leq c_s$",
            "supersonic deflagration\n$c_s < v_\\mathrm{wall} < v_\\mathrm{CJ}$",
            "detonation\n$c_s < v_\\mathrm{CJ}\\leq v_\\mathrm{wall}$"
        ]

        for ax, label, v_wall, plot_cbar in zip(axs, labels, v_walls, plot_cbars, strict=False):
            cs = plot_bubble(ax, label, v_wall, alpha, n_xi)
            if plot_cbar:
                cbar = fig.colorbar(cs, ax=axs)
                cbar.set_label(r"$v/v_\mathrm{peak}$")

        # tight_box = fig.get_tightbbox(fig.canvas.get_renderer())
        # fig.set(figheight=tight_box.height, figwidth=tight_box.width)

        # This has to be inside the rc_context environment for its settings to work.
        if path is not None:
            save_fig(fig, path, bbox_inches="tight")
        return fig


if __name__ == "__main__":
    _fig = main(path="solution_types")
    # The figure is shown within the same rc_context as it was created and saved in.
    with rc_context(RC_PARAMS):
        show_fig(_fig)
