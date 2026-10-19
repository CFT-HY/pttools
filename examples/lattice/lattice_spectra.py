r"""
Simulated and SSM spectra
=========================

Compare the GW power spectra of the $4096^3$ simulations of :correia_2025:`\ `
scaled to their late-time asymptotes with the Sound Shell Model.

The GW power of the simulations is still growing at their end,
and therefore their GW power spectra are scaled to the asymptote with
:py:meth:`pttools.lattice.spectrum.LatticeSpectrum.pow_gw_scaled_to_asymptote`,
i.e. by the ratio of the extrapolated asymptotic total GW power $\Omega_\text{gw}^\infty$
and the total GW power $\Omega_\text{gw}(t)$ at the time of the spectrum.
This assumes that the shape of the spectrum no longer changes.
For comparison, the asymptotes $\mathcal{P}_\text{gw}^\infty(k)$ of the fits of each wavenumber
of :py:mod:`examples.lattice.lattice_gw_asymptote` are also shown.

Both spectra are given the Hubble-scaled mean bubble spacing $r_* = 0.1$.
The SSM spectra are computed with simultaneous nucleation as in the simulations,
and are shown both without suppression
and with the suppression factor fitted to the GW power growth rate of the simulation,
see :py:mod:`examples.lattice.lattice_growth`.
Note that the simulations are in flat space and their flows decay,
whereas the SSM spectrum is that of a stationary source that lasts for $N_\text{sh} = 1$ shock formation time
in an expanding universe.
"""

import logging

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from examples.lattice.lattice_growth import R_STAR, comparisons
from examples.lattice.lattice_gw_asymptote import FIT_SETTINGS
from examples.utils import save_and_show_figs
from pttools.analysis.lattice import LatticeComparison
from pttools.analysis.plot_lattice import plot_pow_gw_asymptote_comparison

logger: logging.Logger = logging.getLogger(__name__)


def omega_table(lattice_comparisons: dict[str, LatticeComparison]) -> pd.DataFrame:
    r"""The total GW power of the simulations and of the SSM.

    The values are integrated over the $z = kR_*$ points of each spectrum.

    :param lattice_comparisons: the comparisons of the runs
    :return: the table
    """
    rows = {}
    for name, comparison in lattice_comparisons.items():
        spectrum = comparison.spectrum
        ssm = comparison.ssm
        step_end = int(FIT_SETTINGS[name]["step_end"])
        ln_y = np.log(spectrum.y)
        rows[comparison.run.label_unicode] = {
            "Omega_gw(t)": np.trapezoid(spectrum.pow_gw, ln_y),
            "Omega_gw_inf": np.trapezoid(spectrum.pow_gw_scaled_to_asymptote(step_end=step_end), ln_y),
            "Omega_gw SSM, suppressed": comparison.fit.factor * np.trapezoid(ssm.pow_gw, np.log(ssm.y)),
            "Omega_gw SSM, unsuppressed": np.trapezoid(ssm.pow_gw, np.log(ssm.y)),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> dict[str, Figure]:
    """Plot the asymptotically scaled simulated GW power spectra and the SSM spectra of the runs of Correia et al."""
    lattice_comparisons = comparisons()
    if not lattice_comparisons:
        return {}
    print(f"r_* = {R_STAR}")
    print(omega_table(lattice_comparisons).to_markdown(floatfmt=".3e"))
    fig, axs = plt.subplots(1, len(lattice_comparisons), figsize=(5 * len(lattice_comparisons), 4), squeeze=False)
    for ax, (name, comparison) in zip(axs[0], lattice_comparisons.items(), strict=True):
        settings = FIT_SETTINGS[name]
        step_end = int(settings["step_end"])
        pow_fit = comparison.run.pow_gw_asymptote(start_fraction=float(settings["start_fraction"]), step_end=step_end)
        plot_pow_gw_asymptote_comparison(comparison, pow_fit=pow_fit, step_end=step_end, fig=fig, ax=ax)
    fig.tight_layout()
    return {"lattice_spectra": fig}


if __name__ == "__main__":
    save_and_show_figs(main())
