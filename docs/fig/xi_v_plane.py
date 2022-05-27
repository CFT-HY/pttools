r"""$(\xi, v)$ plane figure for the bag model"""

import matplotlib.pyplot as plt

from tests.paper.plane import xiv_plane
from tests.paper.plot_plane import plot_plane


def main():
    data = xiv_plane(separate_phases=False)
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    plot_plane(ax=ax, data_s=data, selected_solutions=False)


if __name__ == "__main__":
    main()
    plt.show()
