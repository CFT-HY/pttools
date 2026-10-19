"""Plotting with Plotly, and workarounds for Plotly issues."""

import abc
import logging
import os
import typing as tp

from kaleido._kaleido_tab import KaleidoError
import plotly.graph_objects as go

from pttools.plot.utils import ENABLE_DRAWING

logger: logging.Logger = logging.getLogger(__name__)


# This is defined before the classes, as they use it as a decorator.
def plotly_fix(func: tp.Callable) -> tp.Callable:
    """Suppress Kaleido plotting failures.

    The Kaleido library Plotly uses to create raster graphics such as PNG
    does not work on all headless machines, and its headless browser may time out on slow machines.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KaleidoError, TimeoutError) as err:
            logger.exception(err)
    return wrapper


class PlotlyPlot(abc.ABC):
    """Base class for plotting with Plotly."""

    def __init__(self):
        """Initialize the plot. The figure is created when it's first needed."""
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
    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the figure as a file."""
        fig = self.fig()
        fig.write_html(f"{path}.html")
        fig.write_image(f"{path}.png")

    def show(self) -> None:
        """Show the figure."""
        if ENABLE_DRAWING:
            self.fig().show()
