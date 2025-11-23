"""Base class for plotting with Plotly"""

import abc

import plotly.graph_objects as go

from pttools.analysis.utils import ENABLE_DRAWING


class PlotlyPlot(abc.ABC):
    """Base class for plotting with Plotly"""
    def __init__(self):
        self._fig = None

    @abc.abstractmethod
    def create_fig(self):
        """Create the figure"""
        pass

    def fig(self) -> go.Figure:
        """Get the figure"""
        if self._fig is None:
            self._fig = self.create_fig()
        return self._fig

    def save(self, path: str) -> None:
        """Save the figure as a file"""
        fig = self.fig()
        fig.write_html(f"{path}.html")
        fig.write_image(f"{path}.png")

    def show(self) -> None:
        """Show the figure"""
        if ENABLE_DRAWING:
            self.fig().show()
