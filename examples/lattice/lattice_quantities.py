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

from __future__ import annotations

import logging

from examples.lattice.lattice_growth import R_STAR, Y_MIN
from examples.utils import save_and_show_figs, save_table
from pttools.lattice import compare_chrw25_runs
from pttools.plot.lattice import plot_gw_rate_time_comparison, plot_quantities_comparison
import pttools.type_hints as th
from pttools.utils.time import log_duration

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.figure import Figure

    from pttools.lattice import LatticeComparison

logger: logging.Logger = logging.getLogger(__name__)


def quantities(comparison: LatticeComparison, name: str, gw_rate_time_name: str) -> dict[str, Figure]:
    """Save the table of the quantities of a run and the corresponding SSM values, and plot them.

    :param comparison: the comparison of the run with the SSM
    :param name: name of the table and the figure of the quantities
    :param gw_rate_time_name: name of the figure of the GW power growth rate as a function of time
    :return: the figures by their names
    """
    print(f"{comparison.run.label_unicode}, t = {comparison.spectrum.t:.0f} / T_c")
    # The format is given for each column, including the index.
    save_table(comparison.scalars(), name, floatfmt=("", ".4g", ".4g", ".3g"))
    return {
        name: plot_quantities_comparison(comparison),
        gw_rate_time_name: plot_gw_rate_time_comparison(comparison, y_min=Y_MIN)[0],
    }


@log_duration
def main() -> dict[str, Figure]:
    """Plot the quantities of the runs of Correia et al. and the corresponding SSM values."""
    figs = {}
    for name, comparison in compare_chrw25_runs(y_min=Y_MIN, r_star=R_STAR, missing_ok=True).items():
        figs |= quantities(comparison, f"lattice_quantities_{name}", f"lattice_gw_rate_time_{name}")
    return figs


if __name__ == "__main__":
    save_and_show_figs(main())
