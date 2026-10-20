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
However, the scripts of the article end the fits at the time steps 24800 and 39600, see :py:data:`FIT_SETTINGS`,
whereas the data with the global quantities extends to the time steps 24800 and 40000, see :py:func:`step_ends`.
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

import matplotlib.pyplot as plt
import pandas as pd

from examples.utils import save_and_show_figs
from pttools.lattice import CHRW25_RUNS, LatticeRun, load_chrw25_runs
from pttools.plot.lattice import plot_gw_asymptote
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.figure import Figure

logger: logging.Logger = logging.getLogger(__name__)

#: Settings of the fits of each run as in the scripts of the article:
#: the fraction of the final value at which the fits of each wavenumber start,
#: and the last time step of the fits.
#: The last time steps are not a physical choice: the script ``plot_U.py`` of the article selects the time steps
#: with ``range(0, 25000, 200)`` and ``range(0, 40000, 400)``, which exclude their end points.
#: For the detonation, this agrees with the end of the data, as its log of the global quantities ends at the time step
#: 24950, although its power spectra extend to 25400. For the deflagration, the data extends to 40000.
FIT_SETTINGS: dict[str, dict[str, float | int]] = {
    "detonation": {"start_fraction": 0.8, "step_end": 24800},
    "deflagration": {"start_fraction": 0.6, "step_end": 39600},
}


def fit_cases(lattice_runs: dict[str, LatticeRun]) -> list[tuple[str, LatticeRun, int, str]]:
    """The fits to show: each run with each of its distinct last time steps, see :py:func:`step_ends`.

    :param lattice_runs: the runs
    :return: the name of the run, the run, the last time step, and the description of the last time step
    """
    cases = []
    for name, run in lattice_runs.items():
        descriptions: dict[int, list[str]] = {}
        for description, step_end in step_ends(name, run).items():
            descriptions.setdefault(step_end, []).append(description)
        cases += [(name, run, step_end, ", ".join(desc)) for step_end, desc in descriptions.items()]
    return cases


def fit_table(lattice_runs: dict[str, LatticeRun]) -> pd.DataFrame:
    r"""The fitted indices and the asymptotic GW power of the runs, cf. :correia_2025:`\ ` tables IV and V.

    The values are given both for the last time steps of the article and for the last time steps of the data,
    see :py:func:`fit_cases`.
    The ratio $\Omega_\text{gw}^\infty / \Omega_\text{gw}(t)$ is evaluated at the time step of the comparison
    with the Sound Shell Model, see :py:data:`pttools.lattice.chrw25.CHRW25_RUNS`.

    :param lattice_runs: the runs
    :return: the table
    """
    rows = {}
    for name, run, step_end, description in fit_cases(lattice_runs):
        summary = run.gw_asymptote_summary(CHRW25_RUNS[name].fig_step, step_end=step_end)
        rows[(run.label_unicode, f"{step_end} ({description})")] = {
            key: f"{value:.2f} ± {summary[f'{key}_err']:.2f}" if f"{key}_err" in summary else f"{value:.4g}"
            for key, value in summary.items() if not key.endswith("_err")
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["Run", "Last time step"]
    return table


def step_ends(name: str, run: LatticeRun) -> dict[str, int]:
    """The last time steps of the fits: those of the scripts of the article and those of the data.

    The last time step of the data is the last time step of the power spectra
    that is within the log of the global quantities, which gives the times and the enthalpies of the time steps.

    :param name: name of the run, a key of :py:data:`FIT_SETTINGS`
    :param run: the run
    :return: the last time steps by their description
    """
    steps = run.steps()
    return {
        "article": int(FIT_SETTINGS[name]["step_end"]),
        "data": int(steps[steps <= run.global_steps[-1]][-1]),
    }


def main() -> dict[str, Figure]:
    """Plot the extrapolation of the GW power of the runs of Correia et al."""
    lattice_runs = load_chrw25_runs(missing_ok=True)
    if not lattice_runs:
        return {}
    print(fit_table(lattice_runs).reset_index().to_markdown(index=False))
    cases = fit_cases(lattice_runs)
    fig, axs = plt.subplots(4, len(cases), figsize=(5 * len(cases), 14), squeeze=False)
    for i, (name, run, step_end, description) in enumerate(cases):
        plot_gw_asymptote(
            run, axs[:, i], start_fraction=float(FIT_SETTINGS[name]["start_fraction"]), step_end=step_end,
            title=f"{run.label_latex}\nlast time step {step_end} ({description})")
    fig.tight_layout()
    return {"lattice_gw_asymptote": fig}


if __name__ == "__main__":
    save_and_show_figs(main())
