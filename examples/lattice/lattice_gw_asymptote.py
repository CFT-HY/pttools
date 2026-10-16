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
The values of the GW power differ from those of the axes of the figure of the article,
as its scripts have issues with the scale, see :py:func:`pttools.analysis.lattice.gw_asymptote_scale`.
"""

import logging

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd

from examples.utils import save_and_show_figs
from pttools.analysis.lattice import gw_asymptote_scale
from pttools.analysis.plot_lattice import plot_gw_asymptote
from pttools.lattice import CORREIA_2025_RUNS, LatticeRun, data_dir, load_correia_2025_run

logger: logging.Logger = logging.getLogger(__name__)

#: Settings of the fits of each run as in the scripts of the article:
#: the fraction of the final value at which the fits of each wavenumber start,
#: and the last time step of the fits
FIT_SETTINGS: dict[str, dict[str, float | int]] = {
    "detonation": {"start_fraction": 0.8, "step_end": 24800},
    "deflagration": {"start_fraction": 0.6, "step_end": 39600},
}


def runs() -> dict[str, LatticeRun]:
    """Load the runs of Correia et al., or return an empty dict if the data is not available."""
    if data_dir() is None:
        logger.warning("The data of Correia et al. was not found. Skipping the example.")
        return {}
    return {name: load_correia_2025_run(name) for name in CORREIA_2025_RUNS}


def fit_table(lattice_runs: dict[str, LatticeRun]) -> pd.DataFrame:
    r"""The fitted indices and the asymptotic GW power of the runs, cf. :correia_2025:`\ ` tables IV and V.

    The ratio $\Omega_\text{gw}^\infty / \Omega_\text{gw}(t)$ is evaluated at the time step of the comparison
    with the Sound Shell Model, see :py:data:`pttools.lattice.correia_2025.CORREIA_2025_RUNS`.

    :param lattice_runs: the runs
    :return: the table
    """
    rows = {}
    for name, run in lattice_runs.items():
        step_end = int(FIT_SETTINGS[name]["step_end"])
        zeta = run.kinetic_energy_decay_index(step_end)
        lambda_ = run.integral_scale_growth_index(step_end)
        prediction, prediction_err = run.gw_growth_index_prediction(step_end)
        omega_fit = run.omega_gw_asymptote(step_end=step_end)
        meta = CORREIA_2025_RUNS[name]
        spectrum = run.spectrum(meta.step)
        rows[run.label_unicode] = {
            "zeta": f"{zeta.index:.2f} ± {zeta.index_err:.2f}",
            "lambda": f"{lambda_.index:.2f} ± {lambda_.index_err:.2f}",
            "2 zeta + 2 lambda - 1": f"{prediction:.2f} ± {prediction_err:.2f}",
            "d_tot": f"{omega_fit.index:.2f} ± {omega_fit.index_err:.2f}",
            "Omega_inf (scaled)": f"{omega_fit.value / gw_asymptote_scale(run):.4f}",
            "t [1/T_c]": f"{spectrum.t:.0f}",
            "Omega_inf / Omega(t)": f"{spectrum.omega_gw_asymptote_ratio(step_end=step_end):.3f}",
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> dict[str, Figure]:
    """Plot the extrapolation of the GW power of the runs of Correia et al."""
    lattice_runs = runs()
    if not lattice_runs:
        return {}
    print(fit_table(lattice_runs).to_markdown())
    fig, axs = plt.subplots(4, len(lattice_runs), figsize=(5 * len(lattice_runs), 14), squeeze=False)
    for i, (name, run) in enumerate(lattice_runs.items()):
        settings = FIT_SETTINGS[name]
        plot_gw_asymptote(
            run, axs[:, i], start_fraction=float(settings["start_fraction"]), step_end=int(settings["step_end"]))
    fig.tight_layout()
    return {"lattice_gw_asymptote": fig}


if __name__ == "__main__":
    save_and_show_figs(main())
