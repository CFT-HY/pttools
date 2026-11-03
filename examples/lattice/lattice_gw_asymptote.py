r"""
GW power at late times
======================

Extrapolate the GW power of the $4096^3$ simulations of :correia_2025:`\ ` to late times.
This corresponds to :correia_2025:`\ ` fig. 6,
of which the original versions are ``fits_rho_gw_*.pdf``, ``index_*.pdf``, ``psGW_*.pdf`` and ``total_rho_gw_*.pdf``.

The rows of the figure are, from top to bottom:

- $\mathcal{P}_\text{gw}(k, t)$ of selected wavenumbers $kR_* \in [8.6, 80.9]$ (solid)
  and their fits to $\mathcal{P}_\text{gw}^\infty(k) [1 - (t / {t}_\text{decay}(k))^{-d(k)}]$ (dashed),
- the fitted indices $d(k)$ of each wavenumber
  and their prediction $2\zeta + 2\lambda - 1$
  from the decay of the kinetic energy and the growth of the integral scale,
- the extrapolated $\mathcal{P}_\text{gw}^\infty(k)$ and the GW power spectrum at the end of the simulation,
- the total and partial GW power $\Omega_\text{gw}(t)$ (solid), their fits (dotted)
  and the extrapolated total power $\Omega_\text{gw}^\infty$ (dashed).

The fits use the same settings as the scripts of the article, which produce the values of its tables IV and V,
see :py:meth:`pttools.lattice.run.LatticeRun.omega_gw_asymptote`.
However, the scripts of the article end the fits at the time steps 24800 and 39600,
see :py:data:`pttools.lattice.chrw25.CHRW25_RUNS`, whereas the data with the global quantities
extends to the time steps 24800 and 40000, see :py:meth:`pttools.lattice.run_spectra.LatticeRunSpectra.last_step`.
The fits are therefore done both up to the last time steps of the article and up to those of the data.
As they are the same for the detonation, the figure has three columns:
the detonation, and the deflagration with the last time steps of the article and of the data.
The fitted values depend on the last time step considerably more than their uncertainties suggest,
e.g. ${d}_\text{tot}$ of the deflagration is 0.79 with the time steps of the article and 0.91 with all of them,
which indicates that the approach to the asymptote is not yet well determined by the simulations.
The values of the GW power differ from those of the axes of the figure of the article,
as its scripts have issues with the scale, see :py:attr:`pttools.lattice.run.LatticeRun.gw_asymptote_scale`.
"""

from __future__ import annotations

import logging

from matplotlib.figure import Figure

from examples.utils import save_and_show_figs, save_table
from pttools.lattice import CHRW25_RUNS, fit_cases, fit_table, load_chrw25_runs, step_ends
from pttools.plot.lattice import plot_gw_asymptote
from pttools.utils.time import log_duration

logger: logging.Logger = logging.getLogger(__name__)


@log_duration
def main() -> dict[str, Figure]:
    """Plot the extrapolation of the GW power of the runs of Correia et al."""
    lattice_runs = load_chrw25_runs(missing_ok=True)
    if not lattice_runs:
        return {}
    save_table(fit_table(lattice_runs, CHRW25_RUNS, step_ends).reset_index(), "lattice_gw_asymptote", index=False)
    cases = fit_cases(lattice_runs, step_ends)
    fig = Figure(figsize=(5 * len(cases), 14))
    axs = fig.subplots(4, len(cases), squeeze=False)
    for i, (name, run, step_end, description) in enumerate(cases):
        plot_gw_asymptote(
            run, axs[:, i], start_fraction=CHRW25_RUNS[name].fit_start_fraction, step_end=step_end,
            title=f"{run.label_latex}\nlast time step {step_end} ({description})")
    fig.tight_layout()
    return {"lattice_gw_asymptote": fig}


if __name__ == "__main__":
    save_and_show_figs(main())
