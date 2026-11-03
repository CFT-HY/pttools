r"""
Simulations of Cutting et al.
=============================

Compare the $960^3$ simulations of :cutting_2019:`\ ` with the Sound Shell Model,
and extrapolate their GW power to late times,
as :py:mod:`examples.lattice.lattice_quantities`, :py:mod:`examples.lattice.lattice_spectra`
and :py:mod:`examples.lattice.lattice_gw_asymptote` do for the simulations of :correia_2025:`\ `.
These require the simulation data, see :py:mod:`pttools.lattice.chw19`.
Without the data the example is skipped.

By default, all the runs of :py:func:`pttools.lattice.chw19.chw19_runs` are processed.
Only some of them can be processed by giving the names of their directories, or their full paths,
as command-line arguments, e.g.
``python -m examples.lattice.chw19 alph-0.05-vw-0.24-Nb-8-L-960-dx-1.0``.
The figures and the tables of each run are saved in the subdirectory ``chw19`` of the directory of each format,
e.g. ``fig/png/chw19/alph-0.05-vw-0.24-Nb-8-L-960-dx-1.0_asymptote.png``, with the suffixes:

- ``_quantities`` and ``_gw_rate_time``: the quantities of the simulation and of the SSM,
  see :py:mod:`examples.lattice.lattice_quantities`
- ``_spectra`` and ``_spectra_omega``: the GW power spectra scaled to their asymptotes and the SSM spectra,
  see :py:mod:`examples.lattice.lattice_spectra`
- ``_asymptote``: the extrapolation of the GW power to late times,
  see :py:mod:`examples.lattice.lattice_gw_asymptote`

The table ``chw19/table_i`` compares the runs with :cutting_2019:`\ ` table I,
see :py:func:`pttools.lattice.chw19.table_i_values`.

The spectra are compared at the time steps of the GW power growth rates of :cutting_2019:`\ ` table I,
see :py:func:`pttools.lattice.chw19.run_params`.
Most of the runs do not have the power spectra of $X^i$, and therefore
$\bar{U}_\parallel$ is approximated by the RMS compressional fluid 3-velocity $\bar{v}_\parallel$,
see :py:meth:`pttools.lattice.run_spectra.LatticeRunSpectra.u_quantity`.
The integrals of the power spectra of these runs are 10-30 % larger than the global quantities,
see :py:mod:`pttools.lattice.chw19`.
The runs are shorter than those of :correia_2025:`\ ` in units of the shock formation time,
especially for the slow walls,
and therefore the extrapolation of the GW power to late times is not well determined,
and its fits may fail. The failures are logged, and the remaining figures and tables of the run are still created.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import typing as tp

import pandas as pd

from examples.lattice.lattice_growth import R_STAR, Y_MIN
from examples.lattice.lattice_gw_asymptote import gw_asymptote
from examples.lattice.lattice_quantities import quantities
from examples.lattice.lattice_spectra import spectra
from examples.utils import save_figs, save_table
from pttools.lattice import LatticeComparison, chw19
from pttools.plot import close_figs
import pttools.type_hints as th
from pttools.utils.time import log_duration

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.figure import Figure

    from pttools.lattice import LatticeRun, LatticeRunParams

logger: logging.Logger = logging.getLogger(__name__)

#: Subdirectory of the figures and the tables within the directory of each format
SUBDIR: str = "chw19"
#: Smallest $z = kR_*$ of the fits of each wavenumber of the GW power spectra
Y_MIN_FIT: float = 0.
#: Largest $z = kR_*$ of the fits of each wavenumber of the GW power spectra
Y_MAX_FIT: float = 300.


def run_dirs(names: tp.Iterable[str] | None = None) -> list[Path]:
    """The directories of the runs to process.

    :param names: names of the run directories or their full paths, defaults to all the runs
    :return: the directories of the runs that were found. A warning is logged for those that were not.
    """
    if names is None:
        root = chw19.data_dir()
        return [] if root is None else [root / name for name in chw19.chw19_runs(root, missing_ok=True)]
    root = chw19.data_dir()
    dirs = []
    for name in names:
        path = Path(name)
        if not path.is_dir() and len(path.parts) == 1 and root is not None:
            path = root / name
        if path.is_dir():
            dirs.append(path)
        else:
            logger.warning("The run directory of Cutting et al. was not found: %s", name)
    return dirs


def run_figures(run: LatticeRun, params: LatticeRunParams) -> tuple[dict[str, Figure], dict[str, float | str]]:
    """Create the figures and the tables of a run.

    The failures of the individual figures are logged, and the other figures are still created.

    :param run: the run
    :param params: the parameters of the run
    :return: the figures by their names, and the row of the run in the table ``table_i``
    """
    name = run.name
    prefix = f"{SUBDIR}/{name}"
    figs: dict[str, Figure] = {}
    row: dict[str, float | str] = {"v_wall": run.v_wall, "alpha_n": run.alpha_n, "dx": run.dx}
    try:
        row |= chw19.table_i_values(run)
    except (KeyError, ValueError) as e:
        logger.error("Could not compute the values of table I for the run %s: %s", name, e)
    try:
        step, ubar_max = run.ubar_par_max
        row["Ubar_par,max (spectra)"] = ubar_max
        row["Ubar_par from"] = "X" if run.has_x else "v"
        row["t_max / t_fin"] = float(run.time(step)) / chw19.T_FIN.get(run.v_wall, float("nan"))
        comparison = LatticeComparison(run.spectrum(*params.fig_steps, r_star=R_STAR), y_min=Y_MIN)
        row["Suppression"] = comparison.suppression
    except (RuntimeError, ValueError) as e:
        logger.error("Could not compare the run %s with the Sound Shell Model: %s", name, e)
        comparison = None
    if comparison is not None:
        for description, func in (
                ("quantities", lambda: quantities(comparison, f"{prefix}_quantities", f"{prefix}_gw_rate_time")),
                ("spectra", lambda: {f"{prefix}_spectra": spectra(
                    {name: comparison}, {name: params}, {name: Y_MIN_FIT}, {name: Y_MAX_FIT},
                    f"{prefix}_spectra_omega")})):
            try:
                figs |= func()
            except (RuntimeError, ValueError) as e:
                logger.error("Could not create the %s of the run %s: %s", description, name, e)
    try:
        figs[f"{prefix}_asymptote"] = gw_asymptote({name: run}, {name: params}, step_ends, f"{prefix}_asymptote")
    except (RuntimeError, ValueError) as e:
        logger.error("Could not extrapolate the GW power of the run %s: %s", name, e)
    return figs, row


def step_ends(name: str, run: LatticeRun) -> dict[str, int]:
    """The last time steps of the fits of the extrapolation of the GW power: all the time steps of the data.

    :param name: name of the run
    :param run: the run
    :return: the last time step by its description, see :py:func:`pttools.lattice.asymptote.fit_cases`
    """
    return {"data": run.last_step()}


@log_duration
def main(names: tp.Iterable[str] | None = None, save: bool = False) -> dict[str, Figure]:
    """Plot and tabulate the runs of Cutting et al.

    :param names: names of the run directories or their full paths, defaults to all the runs
    :param save: whether to save the figures of each run and close them before processing the next run,
        to limit the memory usage. The tables are always saved.
    :return: the figures by their names, or an empty dict if ``save`` is True
    """
    figs: dict[str, Figure] = {}
    rows = {}
    for path in run_dirs(names):
        params = chw19.run_params(path)
        logger.info("Processing the run %s", params.name)
        try:
            run = chw19.load_chw19_run(str(path), params=params)
        except (FileNotFoundError, ValueError) as e:
            logger.error("Could not load the run %s: %s", params.name, e)
            continue
        run_figs, rows[params.name] = run_figures(run, params)
        if save:
            # The names of the runs contain dots, which would otherwise be interpreted as file extensions.
            save_figs(run_figs, force_formats=True)
            close_figs(*run_figs.values())
        else:
            figs |= run_figs
    if rows:
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "Run"
        save_table(table, f"{SUBDIR}/table_i", floatfmt=".3g")
    return figs


if __name__ == "__main__":
    main(sys.argv[1:] or None, save=True)
