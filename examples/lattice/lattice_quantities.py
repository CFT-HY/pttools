r"""
Simulated and SSM quantities
============================

Compare the quantities that can be computed both from the simulations of :correia_2025:`\ ` and from PTtools:
the RMS fluid velocities, the integral scale, the kinetic energy fraction, the mean adiabatic index
and the growth rate of the GW energy density.

The simulated flows decay, whereas the Sound Shell Model assumes a stationary flow.
Therefore, the SSM values are shown as constants, and the suppression factor that matches the SSM
GW power growth rate with the simulation depends on the time at which it is fitted.
"""

import logging

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd

from examples.lattice.lattice_growth import comparisons
from examples.utils import save_and_show_figs
from pttools.analysis.lattice import LatticeComparison, scalar_comparison
from pttools.analysis.plot_lattice import (
    plot_gw_rate_time_comparison,
    plot_integral_scale_comparison,
    plot_kinetic_energy_comparison,
    plot_ubar_comparison,
)
from pttools.lattice import CHRW25_RUNS

logger: logging.Logger = logging.getLogger(__name__)


def comparison_table(comparison: LatticeComparison) -> pd.DataFrame:
    """The scalar comparison of a run, see :py:func:`pttools.analysis.lattice.scalar_comparison`.

    :param comparison: the comparison
    :return: the table, which can be formatted e.g. as Markdown with :py:meth:`pandas.DataFrame.to_markdown`
    """
    table = pd.DataFrame.from_dict(scalar_comparison(comparison), orient="index", columns=["Simulation", "PTtools"])
    table.index.name = "Quantity"
    table["Ratio"] = table["Simulation"] / table["PTtools"]
    return table


def plot_quantities(comparison: LatticeComparison) -> Figure:
    """Plot the time series of a run and the corresponding SSM values."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    plot_ubar_comparison(comparison, fig=fig, ax=axs[0, 0])
    plot_integral_scale_comparison(comparison, fig=fig, ax=axs[0, 1])
    plot_kinetic_energy_comparison(comparison, fig=fig, ax=axs[1, 0])
    axs[1, 1].axis("off")
    fig.suptitle(comparison.run.label_latex)
    fig.tight_layout()
    return fig


def main() -> dict[str, Figure]:
    """Plot the quantities of the runs of Correia et al. and the corresponding SSM values."""
    figs = {}
    for name, comparison in comparisons().items():
        print(f"{comparison.run.label_unicode}, t = {comparison.spectrum.t:.0f} / T_c")
        # The format is given for each column, including the index.
        print(comparison_table(comparison).to_markdown(floatfmt=("", ".4g", ".4g", ".3g")))
        figs[f"lattice_quantities_{name}"] = plot_quantities(comparison)
        meta = CHRW25_RUNS[name]
        delta_steps = meta.fig_step - meta.fig_step_prev
        figs[f"lattice_gw_rate_time_{name}"] = plot_gw_rate_time_comparison(comparison, delta_steps, y_min=2)[0]
    return figs


if __name__ == "__main__":
    save_and_show_figs(main())
