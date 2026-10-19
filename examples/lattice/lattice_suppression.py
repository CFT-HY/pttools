r"""
Suppression factor from simulations
===================================

The suppression factors fitted to the simulations of :correia_2025:`\ `,
and the suppression datasets that combine them with the data of :gowling_2021:`\ `.

The suppression factors are computed by
:py:mod:`pttools.lattice.suppression`,
and the combined datasets are
:py:data:`pttools.ssm.suppression.suppression.NO_HYBRIDS_EXT_CHRW25` and
:py:data:`pttools.ssm.suppression.suppression.WITH_HYBRIDS_CHRW25`.
This example does not require the simulation data.
"""

import pandas as pd

from examples.utils import save_and_show_figs
from pttools.plot.suppression import SuppressionPlot
from pttools.ssm.suppression import CHRW25, NO_HYBRIDS_EXT, NO_HYBRIDS_EXT_CHRW25


def comparison_table() -> pd.DataFrame:
    """The old and new suppression factors at the parameters of the simulations.

    :return: the table, which can be formatted e.g. as Markdown with :py:meth:`pandas.DataFrame.to_markdown`
    """
    return pd.DataFrame({
        "v_wall": CHRW25.v_walls,
        "alpha_n": CHRW25.alpha_ns,
        "No hybrids, extended": [
            NO_HYBRIDS_EXT.suppression(float(v_wall), float(alpha_n))
            for v_wall, alpha_n in zip(CHRW25.v_walls, CHRW25.alpha_ns, strict=True)
        ],
        "Correia et al. 2025": CHRW25.suppressions,
    })


def main() -> tuple[SuppressionPlot, SuppressionPlot]:
    r"""Plot the suppression factor $\Sigma({v}_\text{wall}, \alpha_n)$ with and without the new data points."""
    print(comparison_table().to_markdown(index=False, floatfmt=".4g"))
    plots = \
        SuppressionPlot(NO_HYBRIDS_EXT, title="No hybrids, extended"), \
        SuppressionPlot(NO_HYBRIDS_EXT_CHRW25, title="With Correia et al. 2025")
    for plot in plots:
        plot.ax.scatter(
            CHRW25.v_walls, CHRW25.alpha_ns, marker="*", s=80,
            facecolors="none", edgecolors="red", label="Correia et al. 2025")
        plot.ax.legend(fontsize=7, loc="upper left")
    return plots


if __name__ == "__main__":
    _plot_old, _plot_new = main()
    save_and_show_figs({
        "lattice_suppression_no_hybrids_ext": _plot_old.fig,
        "lattice_suppression_no_hybrids_ext_chrw25": _plot_new.fig,
    })
