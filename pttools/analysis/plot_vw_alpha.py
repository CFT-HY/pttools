import matplotlib.pyplot as plt

from pttools.analysis.bubble_grid import BubbleGridVWAlpha


class VwAlphaPlot:
    def __init__(self, fig: plt.Figure = None, ax: plt.Axes = None):
        if fig is None:
            if ax is not None:
                raise ValueError("Cannot provide ax without fig")
            self.fig = plt.figure()
        else:
            self.fig = fig

        self.ax = fig.add_subplot() if ax is None else ax

        self.ax.grid()
        self.ax.set_xlabel("$v_w$")
        self.ax.set_ylabel(r"$\alpha$")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
