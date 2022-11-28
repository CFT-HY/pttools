import logging
import typing as tp

import numpy as np
from plotly.basedatatypes import BasePlotlyType
import plotly.graph_objects as go

from pttools.bubble.bubble import Bubble
from pttools.bubble.boundary import Phase
from pttools.bubble.relativity import lorentz
from pttools.bubble.shock import solve_shock
from pttools.models.model import Model

logger = logging.getLogger(__name__)


class BubblePlot3D:
    def __init__(self, model: Model = None, colorscale: str = "YlOrRd"):
        self.model = model
        self.bubbles: tp.List[Bubble] = []
        self.plots: tp.List[BasePlotlyType] = []
        self.colorscale = colorscale

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
                x=bubble.w/bubble.model.wn_max, y=bubble.xi, z=bubble.v,
                mode="lines",
                name=bubble.label,
                **kwargs
            )
        ])

    def create_fig(self) -> go.Figure:
        self.mu_surface()
        self.shock_surfaces()
        fig = go.Figure(
            data=[
                *self.plots
            ]
        )
        fig.update_layout({
            # "margin": {
            #     "l": 0,
            #     "r": 200,
            #     "b": 0,
            #     "t": 0
            # },
            "scene": {
                "xaxis_title": "w/w(Tc)",
                "yaxis_title": "ξ",
                "zaxis_title": "v"
            },
        })
        return fig

    def mu_surface(self, n_xi: int = 20, n_w: int = 20, w_mult: float = 1.1):
        logger.info("Computing mu surface.")
        if self.model is None:
            return
        xi = np.linspace(0, 1, n_xi)
        w = np.linspace(0, w_mult*max(np.max(bubble.w) for bubble in self.bubbles), n_w)
        cs = np.sqrt(self.model.cs2(w, Phase.BROKEN))
        cs_grid, xi_grid = np.meshgrid(cs, xi)
        mu = lorentz(xi_grid, cs_grid)
        mu[mu < 0] = np.nan

        self.plots.append(go.Surface(
            x=w/self.model.wn_max, y=xi, z=mu,
            opacity=0.5, name=r"µ(ξ, cₛ(w))",
            colorbar={
                "lenmode": "fraction",
                "len": 0.5
            },
            colorscale=self.colorscale
        ))
        logger.info("Mu surface ready.")

    def save(self, path: str) -> go.Figure:
        fig = self.create_fig()
        fig.write_html(f"{path}.html")
        fig.write_image(f"{path}.png")
        return fig

    def shock_surfaces(self, n_xi: int = 20, n_w: int = 30, w_mult: float = 1.1):
        if self.model is None:
            return
        logger.info("Computing shock surface.")
        w_max = max(np.max(bubble.w) for bubble in self.bubbles)
        cs2_min, cs2_min_w = self.model.cs2_min(w_max, Phase.SYMMETRIC)
        xi_arr = np.linspace(np.sqrt(cs2_min), 0.99, n_xi)
        wp_arr = np.linspace(0.01, w_mult*w_max, n_w)
        wp_grid, xi_grid = np.meshgrid(wp_arr, xi_arr)
        vm_grid = np.zeros_like(wp_grid)
        wm_grid = np.zeros_like(wp_grid)

        for i_xi, xi in enumerate(xi_arr):
            for i_wp, wp in enumerate(wp_arr):
                vm_tilde, wm = solve_shock(self.model, xi, wp)
                vm_grid[i_xi, i_wp] = lorentz(xi, vm_tilde)
                wm_grid[i_xi, i_wp] = wm

        vm_grid[vm_grid > 1] = np.nan
        wm_grid[wm_grid > w_mult * w_max] = np.nan

        self.plots.append(go.Surface(
            x=wp_arr/self.model.wn_max, y=xi_arr, z=vm_grid,
            opacity=0.5, name="Shock, $w=w₊$",
            colorscale=self.colorscale, showscale=False
        ))
        self.plots.append(go.Surface(
            x=wm_grid/self.model.wn_max, y=xi_grid, z=vm_grid,
            opacity=0.5, name="Shock, $w=w₋$",
            colorscale=self.colorscale, showscale=False
        ))
        logger.info("Shock surface ready.")

    def show(self) -> go.Figure:
        fig = self.create_fig()
        fig.show()
        return fig
