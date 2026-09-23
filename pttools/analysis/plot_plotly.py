"""Base class for plotting with Plotly."""

import abc
import logging

import plotly.graph_objects as go

from pttools.analysis.plotly import plotly_fix
from pttools.analysis.utils import ENABLE_DRAWING

logger = logging.getLogger(__name__)


class PlotlyPlot(abc.ABC):
    """Base class for plotting with Plotly."""

    def __init__(self):
        self._fig: go.Figure | None = None

    @abc.abstractmethod
    def create_fig(self) -> go.Figure:
        """Create the figure."""

    def fig(self) -> go.Figure:
        """Get the figure."""
        if self._fig is None:
            self._fig = self.create_fig()
        return self._fig

    @plotly_fix
    def save(self, path: str) -> None:
        """Save the figure as a file."""
        fig = self.fig()
        fig.write_html(f"{path}.html")
        fig.write_image(f"{path}.png")

    def show(self) -> None:
        """Show the figure."""
        if ENABLE_DRAWING:
            self.fig().show()
