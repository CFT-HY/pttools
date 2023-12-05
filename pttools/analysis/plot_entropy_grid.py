import logging
import typing as tp

from matplotlib.contour import QuadContourSet
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.cmap import cmap, color_region
from pttools.analysis.bubble_grid import BubbleGridVWAlpha
from pttools.analysis.plot_vw_alpha import VwAlphaPlot
from pttools.bubble.boundary import Phase, SolutionType
from pttools.bubble.bubble import Bubble
from pttools.bubble.chapman_jouguet import v_chapman_jouguet
from pttools.bubble import relativity

if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


class DurationPlot(VwAlphaPlot):
    def __init__(self, grid: BubbleGridVWAlpha, fig: plt.Figure = None, ax: plt.Axes = None):
        super().__init__(fig, ax)
        img = ax.pcolor(grid.v_walls, grid.alpha_ns, np.log10(grid.elapsed()))
        cbar = ax.figure.colorbar(img, ax=ax)

        # cs: QuadContourSet = ax.contourf(grid.v_walls, grid.alpha_ns, grid.elapsed(),
        #                                  locator=ticker.LinearLocator(numticks=20))
        # cbar = self.fig.colorbar(cs)

        cbar.ax.set_ylabel("Time elapsed (log10(s))")
        ax.set_title(f"Time elapsed for {grid.model.label_latex}")


class EntropyPlot(VwAlphaPlot):
    def __init__(
            self,
            grid: BubbleGridVWAlpha,
            entropy: np.ndarray,
            min_level: float,
            max_level: float,
            diff_level: float,
            fig: plt.Figure = None,
            ax: plt.Axes = None):
        super().__init__(fig, ax)
        plot_entropy_data(entropy, grid.v_walls, grid.alpha_ns, min_level, max_level, diff_level, fig=fig, ax=ax)

        color_region(self.ax, grid.v_walls, grid.alpha_ns, grid.numerical_error(), color="red", alpha=0.5)
        color_region(self.ax, grid.v_walls, grid.alpha_ns, grid.unphysical_alpha_plus(), color="green", alpha=0.5)
        color_region(self.ax, grid.v_walls, grid.alpha_ns, grid.solver_failed(), color="black")

        self.ax.plot(v_chapman_jouguet(grid.model, grid.alpha_ns), grid.alpha_ns, 'k--', label=r'$v_{CJ}$')
        self.ax.set_title(rf"$\Delta s / s_n$ for {grid.model.label_latex}")
        self.ax.legend()
        self.ax.set_title("Total net entropy change")


class DeltaEntropyPlot(VwAlphaPlot):
    def __init__(
            self,
            grid: BubbleGridVWAlpha,
            w1: np.ndarray,
            w2: np.ndarray,
            w_ref: np.ndarray,
            title: str,
            fig: plt.Figure = None,
            ax: plt.Axes = None):
        super().__init__(fig, ax)
        rel_change = (w1 - w2) / w_ref
        cs: QuadContourSet = ax.contourf(grid.v_walls, grid.alpha_ns, rel_change, locator=ticker.LinearLocator(numticks=20))
        cbar = self.fig.colorbar(cs)
        cbar.ax.set_ylabel(title)


class EntropyConservationPlot(VwAlphaPlot):
    def __init__(
            self,
            grid: BubbleGridVWAlpha,
            diff: np.ndarray,
            fig: plt.Figure = None,
            ax: plt.Axes = None):
        super().__init__(fig, ax)
        cs: QuadContourSet = ax.contourf(grid.v_walls, grid.alpha_ns, diff, locator=ticker.LinearLocator(numticks=20))
        cbar = self.fig.colorbar(cs)
        cbar.ax.set_ylabel(r"$\tilde{\gamma}_- \tilde{v}_- s_- - \tilde{\gamma}_+ \tilde{v}_+ s_+$")
        self.ax.set_title("Entropy generation at the wall")

        self.ax.plot(v_chapman_jouguet(grid.model, grid.alpha_ns), grid.alpha_ns, 'k--', label=r'$v_{CJ}$')
        color_region(self.ax, grid.v_walls, grid.alpha_ns, grid.solver_failed(), color="black")
        self.ax.legend()


class KappaOmegaSumPlot(VwAlphaPlot):
    def __init__(self, grid: BubbleGridVWAlpha, fig: plt.Figure = None, ax: plt.Axes = None):
        super().__init__(fig, ax)

        kappa_omega_sum = np.abs(grid.kappa() + grid.omega() - 1)
        cs: QuadContourSet = ax.contourf(grid.v_walls, grid.alpha_ns, kappa_omega_sum, locator=ticker.LogLocator())
        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel(r"$|\kappa + \omega - 1|$")
        self.ax.set_title(r"$|\kappa + \omega - 1|$")

        self.ax.plot(v_chapman_jouguet(grid.model, grid.alpha_ns), grid.alpha_ns, 'k--', label=r'$v_{CJ}$')
        self.ax.legend()


COMPUTE_FAIL = (np.nan, ) * 7


def compute(bubble: Bubble):
    try:
        if bubble.no_solution_found or bubble.solver_failed:
            return COMPUTE_FAIL
        return (
            bubble.entropy_density_relative,
            bubble.sp,
            bubble.sm,
            bubble.sm_sh,
            bubble.sn,
            bubble.entropy_flux_diff,
            bubble.entropy_flux_diff_sh,
        )
    except IndexError as e:
        logger.exception(f"Computing entropy quantities failed for {bubble.label_unicode}.", exc_info=e)
        return COMPUTE_FAIL


compute.return_type = (np.float_, np.float_, np.float_, np.float_, np.float_, np.float_, np.float_)


def gen_and_plot_entropy(
        models: tp.List["Model"],
        v_walls: np.ndarray,
        alpha_ns: np.ndarray,
        min_level: float,
        max_level: float,
        diff_level: float,
        use_bag_solver: bool = False,
        path: str = None,
        single_plot: bool = False) -> tp.Tuple[plt.Figure, np.ndarray]:
    figsize = None if single_plot else (16*1.5, 9*1.5)
    fig: plt.Figure = plt.figure(figsize=figsize)
    axs = fig.subplots(nrows=len(models), ncols=1 if single_plot else 4)

    for i_model, model in enumerate(models):
        grid = BubbleGridVWAlpha(model, v_walls, alpha_ns, compute, use_bag_solver=use_bag_solver)
        # These are declared explicitly as above to avoid indexing errors
        s_total_rel = grid.data[0]
        sp = grid.data[1]
        sm = grid.data[2]
        sm_sh = grid.data[3]
        sn = grid.data[4]
        diff = grid.data[5]
        diff_sh = grid.data[6]

        EntropyPlot(grid, s_total_rel, min_level, max_level, diff_level, fig=fig, ax=axs[i_model, 0])
        # DeltaEntropyPlot(
        #     grid, w1=sm, w2=sp, w_ref=sn,
        #     title=r"$\frac{s_- - s_+}{s_n}$", fig=fig, ax=axs[i_model, 1])
        # DeltaEntropyPlot(
        #     grid, w1=sm_sh, w2=sn, w_ref=sn,
        #     title=r"$\frac{s_{sh-} - s_n}{s_n}$", fig=fig, ax=axs[i_model, 2])
        EntropyConservationPlot(grid, diff, fig, axs[i_model, 1])
        # EntropyConservationPlot(grid, diff_sh, fig, axs[i_model, 2])
        KappaOmegaSumPlot(grid, fig, axs[i_model, 2])
        DurationPlot(grid, fig, axs[i_model, 3])

    # fig.tight_layout()

    if path is not None:
        fig.savefig(path)

    return fig, axs


def plot_entropy_data(
        data: np.ndarray,
        v_walls: np.ndarray, alpha_ns: np.ndarray,
        min_level: float, max_level: float, diff_level: float,
        fig: plt.Figure = None,
        ax: plt.Axes = None) -> tp.Tuple[plt.Figure, plt.Axes]:
    if fig is None or ax is None:
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_subplot()

    levels, cols = cmap(min_level, max_level, diff_level)
    cs: QuadContourSet = ax.contourf(v_walls, alpha_ns, data, levels=levels, colors=cols)
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r'$\Delta s / s_n$')

    ax.grid()
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\alpha_n$")
    ax.set_title(rf"$\Delta s / s_n$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.legend()

    return fig, ax
