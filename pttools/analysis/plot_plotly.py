"""Base class for plotting with Plotly"""

import abc
import logging

from kaleido._kaleido_tab import KaleidoError
import plotly.graph_objects as go

from pttools.analysis.utils import ENABLE_DRAWING

logger = logging.getLogger(__name__)


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
        """Save the figure as a file

        Please note that the Kaleido library Plotly uses to create raster graphics such as PNG
        does not work on all headless machines.
        """
        fig = self.fig()
        fig.write_html(f"{path}.html")
        try:
            fig.write_image(f"{path}.png")
        except KaleidoError as err:
            logger.exception(err)

    def show(self) -> None:
        """Show the figure"""
        if ENABLE_DRAWING:
            self.fig().show()
