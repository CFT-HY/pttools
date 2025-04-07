r"""Plot $(v,w,\xi)$ for fluid shells"""

import logging
import typing as tp

import numpy as np
from plotly.basedatatypes import BasePlotlyType
import plotly.graph_objects as go

from pttools.analysis.plot_plotly import PlotlyPlot
from pttools.bubble.bubble import Bubble
from pttools.bubble.boundary import Phase
from pttools.bubble.relativity import lorentz
from pttools.bubble.shock import solve_shock
from pttools.models.model import Model

logger = logging.getLogger(__name__)


class BubblePlot3D(PlotlyPlot):
    r"""Create a 3D plot of bubbles in the $(v,w,\xi)$ space"""
    def __init__(self, model: Model = None, colorscale: str = "YlOrRd"):
        super().__init__()
        self.model = model
        self.bubbles: tp.List[Bubble] = []
        self.plots: tp.List[BasePlotlyType] = []
        self.colorscale = colorscale

    def add(self, bubble: Bubble, color: str = None) -> go.Scatter3d:
        """Add a bubble to the plot"""
        if not bubble.solved:
            bubble.solve()

        self.bubbles.append(bubble)
        kwargs = {}
        if color is not None:
            kwargs["line"] = {
                "color": color
            }
        plot = go.Scatter3d(
                x=bubble.w/bubble.model.w_crit, y=bubble.xi, z=bubble.v,
                mode="lines",
                name=bubble.label_unicode,
                **kwargs
            )
        self.plots.extend([plot])
        return plot

    def create_fig(self) -> go.Figure:
        """Create the figure"""
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

    def mu_surface(self, n_xi: int = 20, n_w: int = 20, w_mult: float = 1.5) -> go.Surface:
        r"""Add the $\mu$ surface to the plot"""
        logger.info("Computing mu surface.")
        if self.model is None:
            return
        xi = np.linspace(0, 1, n_xi)
        w = np.linspace(0, w_mult * self.model.w_crit, n_w)
        cs = np.sqrt(self.model.cs2(w, Phase.BROKEN))
        cs_grid, xi_grid = np.meshgrid(cs, xi)
        mu = lorentz(xi_grid, cs_grid)
        mu[mu < 0] = np.nan

        surf = go.Surface(
            x=w/self.model.w_crit, y=xi, z=mu,
            opacity=0.5, name=r"µ(ξ, cₛ(w))",
            colorbar={
                "lenmode": "fraction",
                "len": 0.5
            },
            colorscale=self.colorscale
        )
        self.plots.append(surf)
        logger.info("Mu surface ready.")
        return surf

    def shock_surfaces(self, n_xi: int = 20, n_w: int = 30, w_mult: float = 1.5, wp_surface: bool = False):
        """Add the shock surfaces to the plot"""
        if self.model is None:
            return
        logger.info("Computing shock surface.")
        w_max = w_mult * self.model.w_crit
        cs2_min, cs2_min_w = self.model.cs2_min(w_max, Phase.SYMMETRIC)
        xi_arr = np.linspace(np.sqrt(cs2_min), 0.99, n_xi)
        wp_arr = np.linspace(0.01, w_mult*w_max, n_w)
        wp_grid, xi_grid = np.meshgrid(wp_arr, xi_arr)
        vm_grid = np.zeros_like(wp_grid)
        wm_grid = np.zeros_like(wp_grid)

        for i_xi, xi in enumerate(xi_arr):
            for i_wp, wp in enumerate(wp_arr):
                vm_tilde, wm = solve_shock(self.model, v1_tilde=xi, w1=wp, backwards=True, warn_if_barely_exists=False)
                vm_grid[i_xi, i_wp] = lorentz(xi, vm_tilde)
                wm_grid[i_xi, i_wp] = wm

        vm_grid[vm_grid > 1] = np.nan
        wm_grid[wm_grid > w_mult * w_max] = np.nan

        if wp_surface:
            self.plots.append(go.Surface(
                x=wp_arr/self.model.w_crit, y=xi_arr, z=vm_grid,
                opacity=0.5, name="Shock, $w=w₊$",
                colorscale=self.colorscale, showscale=False
            ))
        self.plots.append(go.Surface(
            x=wm_grid/self.model.w_crit, y=xi_grid, z=vm_grid,
            opacity=0.5, name="Shock, $w=w₋$",
            colorscale=self.colorscale, showscale=False
        ))
        logger.info("Shock surface ready.")
