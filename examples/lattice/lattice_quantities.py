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

from examples.lattice.lattice_growth import R_STAR, Y_MIN
from examples.utils import save_and_show_figs
from pttools.lattice import compare_chrw25_runs
from pttools.plot.lattice import plot_gw_rate_time_comparison, plot_quantities_comparison

logger: logging.Logger = logging.getLogger(__name__)


def main() -> dict[str, Figure]:
    """Plot the quantities of the runs of Correia et al. and the corresponding SSM values."""
    figs = {}
    for name, comparison in compare_chrw25_runs(y_min=Y_MIN, r_star=R_STAR, missing_ok=True).items():
        print(f"{comparison.run.label_unicode}, t = {comparison.spectrum.t:.0f} / T_c")
        # The format is given for each column, including the index.
        print(comparison.scalars().to_markdown(floatfmt=("", ".4g", ".4g", ".3g")))
        figs[f"lattice_quantities_{name}"] = plot_quantities_comparison(comparison)
        figs[f"lattice_gw_rate_time_{name}"] = plot_gw_rate_time_comparison(comparison, y_min=Y_MIN)[0]
    return figs


if __name__ == "__main__":
    save_and_show_figs(main())
