import typing as tp

import numpy as np
from plotly.basedatatypes import BasePlotlyType
import plotly.graph_objects as go

from pttools.models.model import Model


class DeltaThetaPlot3D:
    def __init__(self):
        self.plots: tp.List[BasePlotlyType] = []

    def add(self, model: Model):
        wp = np.linspace(0, model.wn_max)
        wm = wp
        wp_grid, wm_grid = np.meshgrid(wp, wm)
        delta = model.delta_theta(wp_grid, wm_grid, allow_negative=True)
        self.plots.append(go.Surface(
            x=wp/model.wn_max, y=wm/model.wn_max, z=delta, name=model.label_unicode
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

    def save(self, path: str) -> go.Figure:
        fig = self.create_fig()
        fig.write_html(f"{path}.html")
        fig.write_image(f"{path}.png")
        return fig
