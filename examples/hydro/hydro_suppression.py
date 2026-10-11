r"""
Suppression factor from simulations
===================================

The suppression factors fitted to the simulations of :correia_2026:`\ `,
and the suppression datasets that combine them with the data of :gowling_2021:`\ `.

The suppression factors are computed by
:py:mod:`pttools.ssm.suppression.suppression_ssm_data.suppression_hydro_calculator`,
and the combined datasets are
:py:data:`pttools.ssm.suppression.suppression.NO_HYBRIDS_EXT_CORREIA_2025` and
:py:data:`pttools.ssm.suppression.suppression.WITH_HYBRIDS_CORREIA_2025`.
This example does not require the simulation data.
"""

from examples.utils import save_and_show_figs
from pttools.analysis.suppression import SuppressionPlot
from pttools.ssm.suppression import CORREIA_2025, NO_HYBRIDS_EXT, NO_HYBRIDS_EXT_CORREIA_2025


def comparison_table() -> str:
    """Format the old and new suppression factors at the parameters of the simulations as a Markdown table."""
    rows = [
        f"| {v_wall} | {alpha_n} | {NO_HYBRIDS_EXT.suppression(float(v_wall), float(alpha_n)):.4g} | {sup:.4g} |"
        for v_wall, alpha_n, sup in zip(CORREIA_2025.v_walls, CORREIA_2025.alpha_ns, CORREIA_2025.suppressions,
                                        strict=True)
    ]
    return "\n".join([
        "| v_wall | alpha_n | No hybrids, extended | Correia et al. 2025 |",
        "|---|---|---|---|",
        *rows
    ])


def main() -> tuple[SuppressionPlot, SuppressionPlot]:
    r"""Plot the suppression factor $\Sigma({v}_\text{wall}, \alpha_n)$ with and without the new data points."""
    print(comparison_table())
    plots = \
        SuppressionPlot(NO_HYBRIDS_EXT, title="No hybrids, extended"), \
        SuppressionPlot(NO_HYBRIDS_EXT_CORREIA_2025, title="With Correia et al. 2025")
    for plot in plots:
        plot.ax.scatter(
            CORREIA_2025.v_walls, CORREIA_2025.alpha_ns, marker="*", s=80,
            facecolors="none", edgecolors="red", label="Correia et al. 2025")
        plot.ax.legend(fontsize=7, loc="upper left")
    return plots


if __name__ == "__main__":
    _plot_old, _plot_new = main()
    save_and_show_figs({
        "hydro_suppression_no_hybrids_ext": _plot_old.fig,
        "hydro_suppression_no_hybrids_ext_correia_2025": _plot_new.fig,
    })
