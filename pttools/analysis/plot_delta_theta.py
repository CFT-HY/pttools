r"""Plot $\Delta \theta ({w}_+, {w}_-)$."""

import typing as tp

import numpy as np
from plotly.basedatatypes import BasePlotlyType
import plotly.graph_objects as go

from pttools.analysis.plot_plotly import PlotlyPlot
from pttools.models.model import Model


class DeltaThetaPlot3D(PlotlyPlot):
    r"""Plot $\Delta \theta ({w}_+, {w}_-)$."""

    def __init__(self):
        """Create an empty plot. Add models to it with :meth:`add`."""
        super().__init__()
        self.plots: list[BasePlotlyType] = []

    def add(self, model: Model) -> None:
        r"""Add the $\Delta \theta ({w}_+, {w}_-)$ surface of a model to the plot.

        The enthalpies are in the range $[0, w_{\text{crit}}]$, and they are plotted normalized by $w_\text{crit}$.

        :param model: equation of state model
        """
        wp = np.linspace(0, model.w_crit)
        wm = wp
        wp_grid, wm_grid = np.meshgrid(wp, wm)
        delta = model.delta_theta(wp_grid, wm_grid, error_on_invalid=False, nan_on_invalid=False, log_invalid=False)
        self.plots.append(go.Surface(
            x=wp/model.w_crit, y=wm / model.w_crit, z=delta, name=model.label_unicode
        ))

    @tp.override
    def create_fig(self) -> go.Figure:
        fig = go.Figure(
            data=[
                *self.plots
            ]
        )
        fig.update_layout({
            "scene": {
                "xaxis_title": "w₊",
                "yaxis_title": "w₋",
                "zaxis_title": "Δθ"
            }
        })
        return fig
