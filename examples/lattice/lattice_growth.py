r"""
Simulated and SSM growth rates
==============================

Compare the velocity spectra and the GW power growth rates of the $4096^3$ simulations of :correia_2025:`\ `
with the Sound Shell Model.

The SSM spectra are computed with simultaneous nucleation as in the simulations.
Both spectra are given the Hubble-scaled mean bubble spacing $r_* = 0.1$.
The velocity spectra and the GW power growth rates do not depend on it,
but the GW power spectra and their conversion to $\Omega_{\text{gw},0}$ do.
The SSM GW power growth rate is scaled by the suppression factor that is fitted to the simulation,
excluding the lattice artifact at high $k$.
This corresponds to :correia_2025:`\ ` fig. 5 (bottom),
of which the original versions are ``vel-ps-gw_detonation.pdf`` and ``vel-ps-gw_deflagration.pdf``.
"""

import logging

from matplotlib.figure import Figure

from examples.utils import save_and_show_figs
from pttools.analysis.lattice import LatticeComparison, compare_lattice_run
from pttools.analysis.plot_lattice import plot_spectra_comparison
from pttools.lattice import CORREIA_2025_RUNS, data_dir, load_correia_2025_run

logger: logging.Logger = logging.getLogger(__name__)

#: $r_*$, Hubble-scaled mean bubble spacing of the spectra
R_STAR: float = 0.1


def comparisons() -> dict[str, LatticeComparison]:
    """Compare the runs of Correia et al. with the SSM, or return an empty dict if the data is not available."""
    if data_dir() is None:
        logger.warning("The data of Correia et al. was not found. Skipping the example.")
        return {}
    return {
        name: compare_lattice_run(
            load_correia_2025_run(name), step=meta.fig_step, step_prev=meta.fig_step_prev, y_min=2, r_star=R_STAR)
        for name, meta in CORREIA_2025_RUNS.items()
    }


def main() -> dict[str, Figure]:
    """Plot the simulated and SSM velocity spectra and GW power growth rates of the runs of Correia et al."""
    figs = {}
    for name, comparison in comparisons().items():
        logger.info("%s: suppression factor %s", name, comparison.fit)
        figs[f"lattice_growth_{name}"] = plot_spectra_comparison(comparison)
    return figs


if __name__ == "__main__":
    save_and_show_figs(main())
