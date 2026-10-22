r"""The $4096^3$ simulations of :correia_2025:`\ `.

The data is available in the Zenodo release of the article (https://doi.org/10.5281/zenodo.16760912)
and in the source repository of the article (``UETC_paper_Hydro``).
The Zenodo release does not contain the logs of the global quantities (``*.out``),
which are needed for the normalisation of the spectra,
so the run directories of the source repository should be used when available.

The location of the data is given by the environment variable ``PTTOOLS_LATTICE_DATA_DIR``,
which should point to the root of the ``UETC_paper_Hydro`` repository.
If it's not set, ``UETC_paper_Hydro`` in the same directory as the project repository is used,
see :py:data:`DEFAULT_DATA_DIR`.
"""

import logging
import os
from pathlib import Path
import typing as tp

from pttools.lattice.comparison import LatticeComparison
from pttools.lattice.params import LatticeRunParams
from pttools.lattice.readers import INPUT_FILE_NAME
from pttools.lattice.run import LatticeRun
from pttools.ssm.nucleation import NucType
from pttools.utils.system import PROJECT_DIR

logger: logging.Logger = logging.getLogger(__name__)

#: The runs of :correia_2025:`\ `.
#: The time steps are those of ``plot_ps.py`` of the Zenodo scripts, which produced fig. 5.
#: The deflagration was run twice to different final times, and the longer run is used here.
#: The settings of the fits of the extrapolation of the GW power are those of ``plot_U.py`` of the Zenodo scripts,
#: which produced fig. 6 and the indices of tables IV and V.
#: Its last time steps are not a physical choice: it selects the time steps
#: with ``range(0, 25000, 200)`` and ``range(0, 40000, 400)``, which exclude their end points.
#: For the detonation, this agrees with the end of the data,
#: as its log of the global quantities ends at the time step 24950, although its power spectra extend to 25400.
#: For the deflagration, the data extends to the time step 40000.
#: See :py:meth:`pttools.lattice.run.LatticeRun.last_step` for the last time steps of the data.
CHRW25_RUNS: dict[str, LatticeRunParams] = {
    "detonation": LatticeRunParams(
        name="detonation", subdir="detonation_alpha067_vw092",
        v_wall=0.92, alpha_n=0.67, nuc_type=NucType.SIMULTANEOUS, fig_step=18000, fig_step_prev=12000,
        fit_start_fraction=0.8, fit_step_end=24800),
    "deflagration": LatticeRunParams(
        name="deflagration", subdir="deflagration_alpha050_vw044/ps-all-tfinal40k",
        v_wall=0.44, alpha_n=0.5, nuc_type=NucType.SIMULTANEOUS, fig_step=22000, fig_step_prev=18000,
        fit_start_fraction=0.6, fit_step_end=39600),
}
#: Environment variable for the location of the data
DATA_DIR_ENV: str = "PTTOOLS_LATTICE_DATA_DIR"
#: Default location of the data, ``UETC_paper_Hydro`` in the same directory as the project repository.
#: If PTtools is installed as a package (see :py:data:`pttools.utils.system.IS_PACKAGE`),
#: the project repository is the one at whose root the virtual environment of PTtools is.
#: Otherwise, it's the PTtools repository. See :py:data:`pttools.utils.system.PROJECT_DIR`.
DEFAULT_DATA_DIR: Path | None = None if PROJECT_DIR is None else PROJECT_DIR.parent / "UETC_paper_Hydro"


def compare_chrw25_runs(
        y_min: float | None = None,
        r_star: float | None = None,
        path: str | os.PathLike[str] | None = None,
        missing_ok: bool = False,
        **kwargs: tp.Any) -> dict[str, LatticeComparison]:
    r"""Compare the simulation runs of :correia_2025:`\ ` with the Sound Shell Model.

    The runs are compared at the time steps of :correia_2025:`\ ` fig. 5,
    see :py:attr:`pttools.lattice.params.LatticeRunParams.fig_step`.

    :param y_min: smallest $z = kR_*$ to include in the fit of the suppression factor
    :param r_star: $r_*$, Hubble-scaled mean bubble spacing of the spectra,
        see :py:class:`pttools.lattice.comparison.LatticeComparison`
    :param path: root directory of the data, defaults to :py:func:`data_dir`
    :param missing_ok: whether to return an empty dict with a warning instead of raising an error
        if the data is not found, see :py:func:`load_chrw25_runs`
    :param kwargs: additional arguments for :py:class:`pttools.lattice.comparison.LatticeComparison`
    :return: the comparisons with the keys of :py:data:`CHRW25_RUNS`
    :raises FileNotFoundError: if the data is not found and ``missing_ok`` is False
    """
    return {
        name: LatticeComparison(
            run.spectrum(CHRW25_RUNS[name].fig_step, CHRW25_RUNS[name].fig_step_prev, r_star=r_star),
            y_min=y_min, **kwargs)
        for name, run in load_chrw25_runs(path, missing_ok=missing_ok).items()
    }


def data_dir() -> Path | None:
    r"""Find the directory of the data of :correia_2025:`\ `.

    :return: the root directory of the ``UETC_paper_Hydro`` repository, or None if it's not found
    """
    path = os.environ.get(DATA_DIR_ENV)
    if path is not None:
        if Path(path).is_dir():
            return Path(path)
        logger.warning("The directory given by %s does not exist: %s", DATA_DIR_ENV, path)
        return None
    return DEFAULT_DATA_DIR if DEFAULT_DATA_DIR is not None and DEFAULT_DATA_DIR.is_dir() else None


def load_chrw25_run(name: str, path: str | os.PathLike[str] | None = None) -> LatticeRun:
    r"""Load a simulation run of :correia_2025:`\ `.

    The input file of the deflagration is only in the directory of the longer run,
    and therefore it's searched for in the sibling directories as well.

    :param name: name of the run, a key of :py:data:`CHRW25_RUNS`
    :param path: root directory of the data, defaults to :py:func:`data_dir`
    :return: the run
    :raises FileNotFoundError: if the data is not found
    """
    root = data_dir() if path is None else Path(path)
    if root is None:
        raise FileNotFoundError(
            f"The data of Correia et al. was not found. Please set the environment variable {DATA_DIR_ENV}.")
    meta = CHRW25_RUNS[name]
    run_dir = root / meta.subdir
    input_path = run_dir / INPUT_FILE_NAME
    if not input_path.is_file():
        candidates = sorted(run_dir.parent.glob(f"*/{INPUT_FILE_NAME}"))
        if candidates:
            input_path = candidates[0]
    return LatticeRun(
        run_dir,
        v_wall=meta.v_wall,
        alpha_n=meta.alpha_n,
        nuc_type=meta.nuc_type,
        name=meta.name,
        label_latex=meta.label_latex,
        label_unicode=meta.label_unicode,
        input_path=input_path
    )


def load_chrw25_runs(path: str | os.PathLike[str] | None = None, missing_ok: bool = False) -> dict[str, LatticeRun]:
    r"""Load all the simulation runs of :correia_2025:`\ `.

    :param path: root directory of the data, defaults to :py:func:`data_dir`
    :param missing_ok: whether to return an empty dict with a warning instead of raising an error
        if the data is not found, e.g. for skipping examples
    :return: the runs with the keys of :py:data:`CHRW25_RUNS`
    :raises FileNotFoundError: if the data is not found and ``missing_ok`` is False
    """
    root = data_dir() if path is None else Path(path)
    if missing_ok and (root is None or not root.is_dir()):
        logger.warning(
            "The data of Correia et al. was not found. Please set the environment variable %s.", DATA_DIR_ENV)
        return {}
    return {name: load_chrw25_run(name, root) for name in CHRW25_RUNS}


def step_ends(name: str, run: LatticeRun) -> dict[str, int]:
    """The last time steps of the fits: those of the scripts of the article and those of the data.

    See :py:attr:`pttools.lattice.params.LatticeRunParams.fit_step_end`
    and :py:meth:`pttools.lattice.run.LatticeRun.last_step`.

    :param name: name of the run, a key of :py:data:`CHRW25_RUNS`
    :param run: the run
    :return: the last time steps by their description
    """
    step_end = CHRW25_RUNS[name].fit_step_end
    return {"article": run.last_step() if step_end is None else step_end, "data": run.last_step()}
