r"""The $4096^3$ simulations of :correia_2026:`\ `.

The data is available in the Zenodo release of the article (https://doi.org/10.5281/zenodo.16760912)
and in the source repository of the article (``UETC_paper_Hydro``).
The Zenodo release does not contain the logs of the global quantities (``*.out``),
which are needed for the normalisation of the spectra,
so the run directories of the source repository should be used when available.

The location of the data is given by the environment variable ``PTTOOLS_HYDRO_DATA_DIR``,
which should point to the root of the ``UETC_paper_Hydro`` repository.
If it's not set, ``UETC_paper_Hydro`` in the same directory as the PTtools repository is used.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path

from pttools.imports.hydro.run import HydroRun

logger: logging.Logger = logging.getLogger(__name__)

#: Environment variable for the location of the data
DATA_DIR_ENV: str = "PTTOOLS_HYDRO_DATA_DIR"
#: Default location of the data, next to the PTtools repository
DEFAULT_DATA_DIR: Path = Path(__file__).resolve().parents[3].parent / "UETC_paper_Hydro"


@dataclass(frozen=True)
class CorreiaRun:
    r"""Metadata of a simulation run of :correia_2026:`\ `."""

    #: Name of the run
    name: str
    #: Directory of the run relative to the data directory
    subdir: str
    #: ${v}_\text{wall}$, nominal wall speed
    v_wall: float
    #: $\alpha_n$, transition strength
    alpha_n: float
    #: The time step of the GW power growth rate of :correia_2026:`\ ` fig. 5
    step: int
    #: The earlier time step of the GW power growth rate of :correia_2026:`\ ` fig. 5
    step_prev: int

    @property
    def label_latex(self) -> str:
        """Label of the run in LaTeX format."""
        return rf"{self.name.capitalize()}, $v_w={self.v_wall}, \alpha_n={self.alpha_n}$"

    @property
    def label_unicode(self) -> str:
        """Label of the run in Unicode format."""
        return f"{self.name.capitalize()}, vw={self.v_wall}, αₙ={self.alpha_n}"


#: The runs of :correia_2026:`\ `.
#: The time steps are those of ``plot_ps.py`` of the Zenodo scripts, which produced fig. 5.
#: The deflagration was run twice to different final times, and the longer run is used here.
CORREIA_2025_RUNS: dict[str, CorreiaRun] = {
    "detonation": CorreiaRun(
        name="detonation", subdir="detonation_alpha067_vw092",
        v_wall=0.92, alpha_n=0.67, step=18000, step_prev=12000),
    "deflagration": CorreiaRun(
        name="deflagration", subdir="deflagration_alpha050_vw044/ps-all-tfinal40k",
        v_wall=0.44, alpha_n=0.5, step=22000, step_prev=18000),
}


def data_dir() -> Path | None:
    r"""Find the directory of the data of :correia_2026:`\ `.

    :return: the root directory of the ``UETC_paper_Hydro`` repository, or None if it's not found
    """
    path = os.environ.get(DATA_DIR_ENV)
    if path is not None:
        if Path(path).is_dir():
            return Path(path)
        logger.warning("The directory given by %s does not exist: %s", DATA_DIR_ENV, path)
        return None
    return DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.is_dir() else None


def load_correia_2025_run(name: str, path: str | os.PathLike[str] | None = None) -> HydroRun:
    r"""Load a simulation run of :correia_2026:`\ `.

    The input file of the deflagration is only in the directory of the longer run,
    and therefore it's searched for in the sibling directories as well.

    :param name: name of the run, a key of :py:data:`CORREIA_2025_RUNS`
    :param path: root directory of the data, defaults to :py:func:`data_dir`
    :return: the run
    :raises FileNotFoundError: if the data is not found
    """
    root = data_dir() if path is None else Path(path)
    if root is None:
        raise FileNotFoundError(
            f"The data of Correia et al. was not found. Please set the environment variable {DATA_DIR_ENV}.")
    meta = CORREIA_2025_RUNS[name]
    run_dir = root / meta.subdir
    input_path = run_dir / "input-test-largerun"
    if not input_path.is_file():
        candidates = sorted(run_dir.parent.glob("*/input-test-largerun"))
        if candidates:
            input_path = candidates[0]
    return HydroRun(
        run_dir,
        v_wall=meta.v_wall,
        alpha_n=meta.alpha_n,
        name=meta.name,
        label_latex=meta.label_latex,
        label_unicode=meta.label_unicode,
        input_path=input_path
    )
