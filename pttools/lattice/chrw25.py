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

from pttools.lattice.params import LatticeRunParams
from pttools.lattice.run import LatticeRun
from pttools.utils.system import PROJECT_DIR

logger: logging.Logger = logging.getLogger(__name__)

#: Environment variable for the location of the data
DATA_DIR_ENV: str = "PTTOOLS_LATTICE_DATA_DIR"
#: Default location of the data, ``UETC_paper_Hydro`` in the same directory as the project repository.
#: If PTtools is installed as a package (see :py:data:`pttools.utils.system.IS_PACKAGE`),
#: the project repository is the one at whose root the virtual environment of PTtools is.
#: Otherwise, it's the PTtools repository. See :py:data:`pttools.utils.system.PROJECT_DIR`.
DEFAULT_DATA_DIR: Path | None = None if PROJECT_DIR is None else PROJECT_DIR.parent / "UETC_paper_Hydro"


#: The runs of :correia_2025:`\ `.
#: The time steps are those of ``plot_ps.py`` of the Zenodo scripts, which produced fig. 5.
#: The deflagration was run twice to different final times, and the longer run is used here.
CHRW25_RUNS: dict[str, LatticeRunParams] = {
    "detonation": LatticeRunParams(
        name="detonation", subdir="detonation_alpha067_vw092",
        v_wall=0.92, alpha_n=0.67, fig_step=18000, fig_step_prev=12000),
    "deflagration": LatticeRunParams(
        name="deflagration", subdir="deflagration_alpha050_vw044/ps-all-tfinal40k",
        v_wall=0.44, alpha_n=0.5, fig_step=22000, fig_step_prev=18000),
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
    input_path = run_dir / "input-test-largerun"
    if not input_path.is_file():
        candidates = sorted(run_dir.parent.glob("*/input-test-largerun"))
        if candidates:
            input_path = candidates[0]
    return LatticeRun(
        run_dir,
        v_wall=meta.v_wall,
        alpha_n=meta.alpha_n,
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
