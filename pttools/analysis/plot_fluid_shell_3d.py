import typing as tp

import numpy as np
from plotly.basedatatypes import BasePlotlyType
import plotly.graph_objects as go

from pttools.bubble.bubble import Bubble
from pttools.bubble.boundary import Phase
from pttools.bubble.relativity import lorentz
from pttools.models.model import Model


class BubblePlot3D:
    def __init__(self, model: Model = None):
        self.model = model
        self.bubbles: tp.List[Bubble] = []
        self.plots: tp.List[BasePlotlyType] = []

    def add(self, bubble: Bubble, color: str = None):
        if not bubble.solved:
            bubble.solve()

        self.bubbles.append(bubble)
        kwargs = {}
        if color is not None:
            kwargs["line"] = {
                "color": color
            }
        self.plots.extend([
            go.Scatter3d(
                x=bubble.w, y=bubble.xi, z=bubble.v,
                mode="lines",
                **kwargs
            )
        ])

    def create_fig(self):
        self.mu_surface()
        return go.Figure(
            data=[
                *self.plots
            ],
            layout={
                "scene": {
                    "xaxis_title": "w",
                    "yaxis_title": r"$\xi$",
                    "zaxis_title": "v"
                }
            }
        )

    def mu_surface(self, n_xi: int = 20, n_w: int = 20, w_mult: float = 1.1):
        if self.model is None:
            return
        xi = np.linspace(0, 1, n_xi)
        w = np.linspace(0, w_mult*max(np.max(bubble.w) for bubble in self.bubbles), n_w)
        cs = np.sqrt(self.model.cs2(w, Phase.BROKEN))
        cs_grid, xi_grid = np.meshgrid(cs, xi)
        mu = lorentz(xi_grid, cs_grid)
        mu[mu < 0] = np.nan

        self.plots.append(go.Surface(z=mu, x=w, y=xi, opacity=0.5))

    def save(self, path: str):
        fig = self.create_fig()
        fig.write_html(f"{path}.html")
        fig.write_image(f"{path}.png")

    def shock_surface(self, n_xi: int = 20, n_w: int = 20, w_mult: float = 1.1):
        if self.model is None:
            return
        xi = np.linspace(0, 1, n_xi)
        w = np.linspace(0, w_mult*max(np.max(bubble.w) for bubble in self.bubbles), n_w)

    def show(self):
        self.create_fig().show()
