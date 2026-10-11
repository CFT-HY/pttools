r"""
Simulated and SSM spectra
=========================

Compare the velocity and GW power spectra of the $4096^3$ simulations of :correia_2026:`\ `
with the Sound Shell Model.

The SSM spectra are computed with simultaneous nucleation as in the simulations.
The SSM GW power growth rate is scaled by the suppression factor that is fitted to the simulation,
excluding the lattice artifact at high $k$.
This corresponds to :correia_2026:`\ ` fig. 5 (bottom),
of which the original versions are ``vel-ps-gw_detonation.pdf`` and ``vel-ps-gw_deflagration.pdf``.
"""

import logging

from matplotlib.figure import Figure

from examples.utils import save_and_show_figs
from pttools.analysis.hydro import HydroComparison, compare_hydro_run
from pttools.analysis.plot_hydro import plot_spectra_comparison
from pttools.imports.hydro import CORREIA_2025_RUNS, data_dir, load_correia_2025_run

logger: logging.Logger = logging.getLogger(__name__)


def comparisons() -> dict[str, HydroComparison]:
    """Compare the runs of Correia et al. with the SSM, or return an empty dict if the data is not available."""
    if data_dir() is None:
        logger.warning("The data of Correia et al. was not found. Skipping the example.")
        return {}
    return {
        name: compare_hydro_run(load_correia_2025_run(name), step=meta.step, step_prev=meta.step_prev, y_min=2)
        for name, meta in CORREIA_2025_RUNS.items()
    }


def main() -> dict[str, Figure]:
    """Plot the simulated and SSM spectra of the runs of Correia et al."""
    figs = {}
    for name, comparison in comparisons().items():
        logger.info("%s: suppression factor %s", name, comparison.fit)
        figs[f"hydro_spectra_{name}"] = plot_spectra_comparison(comparison)
    return figs


if __name__ == "__main__":
    save_and_show_figs(main())
