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
import pandas as pd

from examples.lattice.lattice_growth import R_STAR, Y_MIN
from examples.lattice.lattice_gw_asymptote import FIT_SETTINGS
from examples.utils import save_and_show_figs
from pttools.lattice import compare_chrw25_runs
from pttools.plot.lattice import plot_pow_gw_asymptote_comparison

logger: logging.Logger = logging.getLogger(__name__)


def main() -> dict[str, Figure]:
    """Plot the asymptotically scaled simulated GW power spectra and the SSM spectra of the runs of Correia et al."""
    lattice_comparisons = compare_chrw25_runs(y_min=Y_MIN, r_star=R_STAR, missing_ok=True)
    if not lattice_comparisons:
        return {}
    print(f"r_* = {R_STAR}")

    omega = pd.DataFrame.from_dict({
        comparison.run.label_unicode: comparison.omega(step_end=int(FIT_SETTINGS[name]["step_end"]))
        for name, comparison in lattice_comparisons.items()
    }, orient="index")
    print(omega.to_markdown(floatfmt=".3e"))

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
