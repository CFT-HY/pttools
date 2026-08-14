r"""$(\xi, v)$ plane figure for the bag model"""

import matplotlib.pyplot as plt

from examples.utils import save_and_show_fig
from tests.paper.plane import xiv_plane
from tests.paper.plot_plane_paper import plot_plane


def main() -> plt.Figure:
    data = xiv_plane(separate_phases=False)
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    plot_plane(ax=ax, data_s=data, selected_solutions=False)
    return fig


if __name__ == "__main__":
    _fig = main()
    save_and_show_fig(_fig, "xi_v_plane")
