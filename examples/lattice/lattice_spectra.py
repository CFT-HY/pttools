r"""
Simulated and SSM spectra
=========================

Compare the GW power spectra of the $4096^3$ simulations of :correia_2025:`\ `
scaled to their late-time asymptotes with the Sound Shell Model.

The GW power of the simulations is still growing at their end,
and therefore their GW power spectra are scaled to the asymptote with
:py:meth:`pttools.lattice.spectrum.LatticeSpectrum.pow_gw_asymptote`,
i.e. by the ratio of the extrapolated asymptotic total GW power $\Omega_\text{gw}^\infty$
and the total GW power $\Omega_\text{gw}(t)$ at the time of the spectrum.
This assumes that the shape of the spectrum no longer changes.
For comparison, the asymptotes $\mathcal{P}_\text{gw}^\infty(k)$ of the fits of each wavenumber
of :py:mod:`examples.lattice.lattice_gw_asymptote` are also shown,
here over the ranges of :py:data:`examples.lattice.lattice_spectra.Y_MIN_FIT`
and :py:data:`examples.lattice.lattice_spectra.Y_MAX_FIT`.
Below $kR_* = 4$, the fit of the detonation does not converge,
whereas those of the deflagration start from the smallest wavenumber of the data.
The fits use all the time steps of the data with the global quantities,
see :py:meth:`pttools.lattice.run.LatticeRun.last_step`.
At higher wavenumbers, the GW power is still growing at an accelerating rate at the end of the simulations,
and therefore its approach to an asymptote is not yet determined by them.
For the detonation, this is the case above $kR_* \approx 10^3$,
which includes the bump of the lattice artifacts.
For the deflagration, the fits fail to converge without bounds above $kR_* \approx 380$,
and their uncertainties exceed 50% already above $kR_* \approx 360$.
The fits of the deflagration are also noisy above $kR_* \approx 30$.

Both spectra are given the Hubble-scaled mean bubble spacing $r_* = 0.1$.
The SSM spectra are computed with simultaneous nucleation as in the simulations,
and are shown both without suppression
and with the suppression factor fitted to the GW power growth rate of the simulation,
see :py:mod:`examples.lattice.lattice_growth`.
Note that the simulations are in flat space and their flows decay,
whereas the SSM spectrum is that of a stationary source that lasts for $N_\text{sh} = 1$ shock formation time
in an expanding universe.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

from examples.lattice.lattice_growth import R_STAR, Y_MIN
from examples.utils import save_and_show_figs
from pttools.lattice import CHRW25_RUNS, compare_chrw25_runs
from pttools.plot.lattice import plot_pow_gw_asymptote_comparison
import pttools.type_hints as th
from pttools.utils.time import log_duration

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.figure import Figure

logger: logging.Logger = logging.getLogger(__name__)

#: Smallest $z = kR_*$ of the fits of each wavenumber of each run
Y_MIN_FIT: dict[str, float] = {"detonation": 4., "deflagration": 0.}
#: Largest $z = kR_*$ of the fits of each wavenumber of each run,
#: above which the GW power is not yet approaching an asymptote at the end of the simulation
Y_MAX_FIT: dict[str, float] = {"detonation": 1e3, "deflagration": 360.}


@log_duration
def main() -> dict[str, Figure]:
    """Plot the asymptotically scaled simulated GW power spectra and the SSM spectra of the runs of Correia et al."""
    lattice_comparisons = compare_chrw25_runs(y_min=Y_MIN, r_star=R_STAR, missing_ok=True)
    if not lattice_comparisons:
        return {}
    print(f"r_* = {R_STAR}")

    step_end = {name: comparison.run.last_step() for name, comparison in lattice_comparisons.items()}
    omega = pd.DataFrame.from_dict({
        comparison.run.label_unicode: comparison.omega(step_end=step_end[name])
        for name, comparison in lattice_comparisons.items()
    }, orient="index")
    print(omega.to_markdown(floatfmt=".3e"))

    fig, axs = plt.subplots(1, len(lattice_comparisons), figsize=(5 * len(lattice_comparisons), 4), squeeze=False)
    for ax, (name, comparison) in zip(axs[0], lattice_comparisons.items(), strict=True):
        pow_fit = comparison.run.pow_gw_asymptote(
            y_min=Y_MIN_FIT[name], y_max=Y_MAX_FIT[name],
            start_fraction=CHRW25_RUNS[name].fit_start_fraction, step_end=step_end[name])
        plot_pow_gw_asymptote_comparison(comparison, pow_fit=pow_fit, step_end=step_end[name], fig=fig, ax=ax)
    fig.tight_layout()
    return {"lattice_spectra": fig}


if __name__ == "__main__":
    save_and_show_figs(main())
