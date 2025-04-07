r"""Plot $\Delta \theta ({w}_+, {w}_-)$"""

import typing as tp

import numpy as np
from plotly.basedatatypes import BasePlotlyType
import plotly.graph_objects as go

from pttools.analysis.plot_plotly import PlotlyPlot
from pttools.models.model import Model


class DeltaThetaPlot3D(PlotlyPlot):
    def __init__(self):
        super().__init__()
        self.plots: tp.List[BasePlotlyType] = []

    def add(self, model: Model):
        wp = np.linspace(0, model.w_crit)
        wm = wp
        wp_grid, wm_grid = np.meshgrid(wp, wm)
        delta = model.delta_theta(wp_grid, wm_grid, error_on_invalid=False, nan_on_invalid=False, log_invalid=False)
        self.plots.append(go.Surface(
            x=wp/model.w_crit, y=wm / model.w_crit, z=delta, name=model.label_unicode
        ))

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
