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
from pttools.lattice import compare_chrw25_runs
from pttools.plot.lattice import plot_spectra_comparison

logger: logging.Logger = logging.getLogger(__name__)

#: $r_*$, Hubble-scaled mean bubble spacing of the spectra
R_STAR: float = 0.1
#: Smallest $z = kR_*$ included in the fit of the suppression factor.
#: This excludes the lowest bin, which contains only one mode.
Y_MIN: float = 2.


def main() -> dict[str, Figure]:
    """Plot the simulated and SSM velocity spectra and GW power growth rates of the runs of Correia et al."""
    figs = {}
    for name, comparison in compare_chrw25_runs(y_min=Y_MIN, r_star=R_STAR, missing_ok=True).items():
        logger.info("%s: suppression factor %s", name, comparison.fit)
        figs[f"lattice_growth_{name}"] = plot_spectra_comparison(comparison)
    return figs


if __name__ == "__main__":
    save_and_show_figs(main())
