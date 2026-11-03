r"""The $960^3$ simulations of :cutting_2019:`\ `.

The location of the data is given by the environment variable ``PTTOOLS_CHW19_DATA_DIR``,
which should point to the directory that contains the run directories.
If it's not set, ``CHW19_simulation_data`` in the same directory as the project repository is used,
see :py:data:`DEFAULT_DATA_DIR`.

The name of each run directory gives its parameters, e.g. ``alph-0.05-vw-0.24-Nb-8-L-960-dx-1.0``,
see :py:data:`RUN_NAME_PATTERN`, optionally followed by a variant.
The variants differ from the corresponding run without a variant as follows, according to their input files:

- ``seed-2``: the random seed of the bubble locations
- ``dt-0.1`` and ``dt-0.4``: the time step $\delta t$
- ``dabya-0.05`` and ``dabya-0.5``: $-(a_b - a_s) / a_s$ = 0.05 and 0.5 instead of $5.9 \times 10^{-3}$,
  with the potential and $T_n$ adjusted to give the same $\alpha_n$
- ``globals``: run with the later version of the simulation code, which writes more global quantities
  and the power spectra of $X^i$ and $J^i$
- ``movie``: run for the visualisation slices of the article, with few or no power spectra
- ``Tvort``: no differences in the input file

The runs with the lattice spacings $\delta x \neq 1$ have the same physical volume
and are the resolution tests of :cutting_2019:`\ ` figs. 5-7.

These runs were made with an older version of the simulation code than those of :correia_2025:`\ `
(:py:mod:`pttools.lattice.chrw25`), and therefore:

- The global quantities are in ``globals.txt`` without a header,
  see :py:data:`pttools.lattice.readers.GLOBALS_TXT_COLUMNS`.
- The power spectrum files have other names, see :py:data:`pttools.lattice.convert.LEGACY_SPECTRUM_FILES`.
- Most runs have only the power spectra of the fluid 3-velocity $v^i$ and not those of $X^i$,
  in which case $\bar{U}_\parallel$ is approximated by $\bar{v}_\parallel$,
  see :py:meth:`pttools.lattice.run_spectra.LatticeRunSpectra.u_quantity`.
- The integrals of the power spectra are 10-30 % larger than the corresponding quantities
  of the global quantities, depending on the time, e.g. $\int \frac{dk}{k} \mathcal{P}_\text{gw}$
  compared with ``GW Energy``, and $\bar{w} \int \frac{dk}{k} \mathcal{P}_{U}$
  compared with ``kin fluid`` divided by the volume in the runs that have the power spectra of $X^i$.
  Therefore, e.g. $\Omega_\text{gw}$ and $\bar{U}_\parallel$ of the spectra are larger than those of the article.
  For the runs of :correia_2025:`\ `, these agree within 0.3 %.
  The difference is similar for $\delta x = 1$ and 1.5, and is therefore not a lattice artifact,
  but probably due to the normalisation of the bins of the spectra in this version of the code.
- Some runs have the bubble location file ``bubblelocs.txt``, see
  :py:attr:`pttools.lattice.run_base.LatticeRunBase.bubble_locs`.

The article normalises $\Omega_\text{gw} = \rho_\text{gw} / \rho_c$ and $H_n$ with the radiation energy density
$\epsilon_r = 3 a_0 T_n^4$ (:py:func:`epsilon_r`), which excludes the vacuum energy,
whereas :py:attr:`pttools.lattice.run_base.LatticeRunBase.epsilon_0` and :correia_2025:`\ ` include it.
As $\bar{\epsilon}_0 = (1 + \alpha_n) \epsilon_r$ in the bag model,
the GW power growth rates of PTtools are smaller than those of the article by $(1 + \alpha_n)^2$.
With this normalisation and the GW energy of the global quantities,
the GW power growth rates of :cutting_2019:`\ ` table I are reproduced within 4 %, see :py:func:`table_i_values`.
"""

import logging
import os
from pathlib import Path
import re
import typing as tp

import numpy as np

from pttools.lattice.convert import find_spectrum_files
from pttools.lattice.params import LatticeRunParams
from pttools.lattice.quantities import Quantity
from pttools.lattice.readers import find_input_file, parse_input_file
from pttools.lattice.run import LatticeRun
from pttools.ssm.nucleation import NucType
from pttools.utils.system import PROJECT_DIR

__all__ = ["chw19_runs", "load_chw19_run", "load_chw19_runs"]

logger: logging.Logger = logging.getLogger(__name__)

#: Environment variable for the location of the data
DATA_DIR_ENV: str = "PTTOOLS_CHW19_DATA_DIR"
#: Default location of the data, ``CHW19_simulation_data`` in the same directory as the project repository,
#: see :py:data:`pttools.lattice.chrw25.DEFAULT_DATA_DIR`.
DEFAULT_DATA_DIR: Path | None = None if PROJECT_DIR is None else PROJECT_DIR.parent / "CHW19_simulation_data"
#: Pattern of the names of the run directories, e.g. ``alph-0.05-vw-0.24-Nb-8-L-960-dx-1.0-seed-2``
RUN_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"alph-(?P<alpha_n>\d+(?:\.\d+)?)-vw-(?P<v_wall>\d+(?:\.\d+)?)-Nb-(?P<n_bubbles>\d+)"
    r"-L-(?P<L>\d+)-dx-(?P<dx>\d+(?:\.\d+)?)(?:-(?P<variant>.+))?")
#: ${t}_\text{fin}$, the end time of the runs of each wall speed in :cutting_2019:`\ ` table I.
#: The GW power growth rates of the table are averaged over the final $\Delta t = 2 R_*$.
#: Some runs of the data continue beyond these.
T_FIN: dict[float, float] = {0.24: 4800., 0.44: 2400., 0.56: 2800., 0.82: 2800., 0.92: 2400.}


def chw19_runs(path: str | os.PathLike[str] | None = None, missing_ok: bool = False) -> dict[str, LatticeRunParams]:
    r"""The parameters of the runs of :cutting_2019:`\ `.

    The runs are the subdirectories of the data directory whose names match :py:data:`RUN_NAME_PATTERN`,
    see :py:func:`run_params`.

    :param path: directory of the data, defaults to :py:func:`data_dir`
    :param missing_ok: whether to return an empty dict with a warning instead of raising an error
        if the data is not found, e.g. for skipping examples
    :return: the parameters of the runs by the names of their directories, in alphabetical order
    :raises FileNotFoundError: if the data is not found and ``missing_ok`` is False
    """
    root = _root(path, missing_ok)
    if root is None:
        return {}
    return {
        run_dir.name: run_params(run_dir)
        for run_dir in sorted(root.iterdir())
        if run_dir.is_dir() and RUN_NAME_PATTERN.fullmatch(run_dir.name)
    }


def data_dir() -> Path | None:
    r"""Find the directory of the data of :cutting_2019:`\ `.

    :return: the directory, or None if it's not found
    """
    path = os.environ.get(DATA_DIR_ENV)
    if path is not None:
        if Path(path).is_dir():
            return Path(path)
        logger.warning("The directory given by %s does not exist: %s", DATA_DIR_ENV, path)
        return None
    return DEFAULT_DATA_DIR if DEFAULT_DATA_DIR is not None and DEFAULT_DATA_DIR.is_dir() else None


def epsilon_r(run: LatticeRun) -> float:
    r"""$\epsilon_r = 3 a_0 T_n^4$, radiation energy density of the symmetric phase at the nucleation temperature.

    $a_0 = \frac{\pi^2}{90} g_*$, :cutting_2019:`\ ` eq. 5,
    where $g_*$ is that of the input file of the simulation.
    This is the normalisation of $\alpha$ (:cutting_2019:`\ ` eq. 8),
    and also that of $\Omega_\text{gw}$ and $H_n$ in the article, see :py:func:`table_i_values`.
    In the bag model, $\bar{\epsilon}_0 = (1 + \alpha_n) \epsilon_r$.

    :param run: the run
    :return: $\epsilon_r$
    """
    a_0 = np.pi**2 / 90 * float(run.params["gstar"])
    return float(3 * a_0 * run.T_n**4)


def load_chw19_run(
        name: str,
        path: str | os.PathLike[str] | None = None,
        params: LatticeRunParams | None = None,
        **kwargs: tp.Any) -> LatticeRun:
    r"""Load a simulation run of :cutting_2019:`\ `.

    :param name: name of the directory of the run, or its full path
    :param path: directory of the data, defaults to :py:func:`data_dir`. Ignored if ``name`` is a full path.
    :param params: the parameters of the run, defaults to :py:func:`run_params`
    :param kwargs: additional arguments for :py:class:`pttools.lattice.run.LatticeRun`
    :return: the run
    :raises FileNotFoundError: if the data is not found
    """
    run_dir = Path(name)
    if not run_dir.is_absolute() and len(run_dir.parts) == 1:
        root = data_dir() if path is None else Path(path)
        if root is None:
            raise FileNotFoundError(
                "The data of Cutting et al. was not found. "
                f"Please set the environment variable {DATA_DIR_ENV}.")
        run_dir = root / name
    meta = run_params(run_dir) if params is None else params
    return LatticeRun(
        run_dir,
        v_wall=meta.v_wall,
        alpha_n=meta.alpha_n,
        nuc_type=meta.nuc_type,
        name=meta.name,
        label_latex=meta.label_latex,
        label_unicode=meta.label_unicode,
        n_bubbles=meta.n_bubbles,
        L=meta.L,
        dx=meta.dx,
        **kwargs
    )


def load_chw19_runs(
        names: tp.Iterable[str] | None = None,
        path: str | os.PathLike[str] | None = None,
        missing_ok: bool = False) -> dict[str, LatticeRun]:
    r"""Load the simulation runs of :cutting_2019:`\ `.

    :param names: names of the directories of the runs, or their full paths, defaults to all the runs
        of :py:func:`chw19_runs`
    :param path: directory of the data, defaults to :py:func:`data_dir`
    :param missing_ok: whether to return an empty dict with a warning instead of raising an error
        if the data is not found, see :py:func:`chw19_runs`
    :return: the runs by the names of their directories
    :raises FileNotFoundError: if the data is not found and ``missing_ok`` is False
    """
    if names is None:
        return {
            name: load_chw19_run(name, path, params=params)
            for name, params in chw19_runs(path, missing_ok=missing_ok).items()
        }
    runs = {}
    for name in names:
        run = load_chw19_run(name, path)
        runs[run.name] = run
    return runs


def _root(path: str | os.PathLike[str] | None, missing_ok: bool) -> Path | None:
    """The directory of the data, or None if it's not found and ``missing_ok`` is True."""
    root = data_dir() if path is None else Path(path)
    if root is None or not root.is_dir():
        if missing_ok:
            logger.warning(
                "The data of Cutting et al. was not found. Please set the environment variable %s.", DATA_DIR_ENV)
            return None
        raise FileNotFoundError(
            f"The data of Cutting et al. was not found. Please set the environment variable {DATA_DIR_ENV}.")
    return root


def run_params(path: str | os.PathLike[str]) -> LatticeRunParams:
    r"""The parameters of a run of :cutting_2019:`\ ` from the name of its directory.

    $\alpha_n$, ${v}_\text{wall}$, $N_b$, $L$ and $\delta x$ are taken from the name of the directory,
    see :py:data:`RUN_NAME_PATTERN`.
    The time steps of the figures are those of the GW power growth rates of :cutting_2019:`\ ` table I,
    which are averaged over the final $\Delta t = 2 R_*$ of the runs, i.e. from ${t}_\text{fin} - 2 R_*$
    to ${t}_\text{fin}$ of :py:data:`T_FIN`, where $R_* = L \delta x / N_b^{1/3}$.
    They are the last time step of the power spectra at or before ${t}_\text{fin}$
    and the time step that is nearest to $2 R_*$ before it,
    using $t = n \delta t$ with $\delta t$ of the input file.
    They are None if the run does not have the power spectra or the input file,
    or if its wall speed is not in :py:data:`T_FIN`.
    The fits of the extrapolation of the GW power use all the time steps,
    as the article does not extrapolate the GW power.

    :param path: directory of the run
    :return: the parameters
    :raises ValueError: if the name of the directory does not match :py:data:`RUN_NAME_PATTERN`
    """
    path = Path(path)
    match = RUN_NAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"The name of the run directory does not match the pattern of the runs: {path.name}")
    v_wall = float(match["v_wall"])
    n_bubbles = int(match["n_bubbles"])
    L = int(match["L"])
    dx = float(match["dx"])
    variant = match["variant"]
    fig_step, fig_step_prev = _fig_steps(path, v_wall, R_star=L * dx / n_bubbles**(1 / 3))
    return LatticeRunParams(
        name=path.name,
        subdir=path.name,
        v_wall=v_wall,
        alpha_n=float(match["alpha_n"]),
        nuc_type=NucType.SIMULTANEOUS,
        fig_step=fig_step,
        fig_step_prev=fig_step_prev,
        n_bubbles=n_bubbles,
        L=L,
        dx=dx,
        label="CHW19" if variant is None else f"CHW19 {variant}",
    )


def _fig_steps(path: Path, v_wall: float, R_star: float) -> tuple[int | None, int | None]:
    """The time steps of the figures, see :py:func:`run_params`."""
    t_fin = T_FIN.get(v_wall)
    if t_fin is None or not path.is_dir():
        return None, None
    try:
        input_path = find_input_file(path)
        files = find_spectrum_files(path)
    except (FileNotFoundError, ValueError) as e:
        logger.warning("Could not determine the time steps of the figures of the run %s: %s", path.name, e)
        return None, None
    if input_path is None:
        return None, None
    dt = float(parse_input_file(input_path)["dt"])
    velocity = Quantity.X_DIV if Quantity.X_DIV in files else Quantity.VEL_DIV
    steps = np.array(sorted(set(files.get(Quantity.GW_TT, {})) & set(files.get(velocity, {}))), dtype=np.int_)
    steps = steps[steps * dt <= t_fin + dt / 2]
    if not steps.size:
        return None, None
    step = int(steps[-1])
    step_prev = int(steps[np.argmin(np.abs(steps * dt - (step * dt - 2 * R_star)))])
    if step_prev == step:
        return None, None
    return step, step_prev


def table_i_values(run: LatticeRun, t_fin: float | None = None) -> dict[str, float]:
    r"""The quantities of :cutting_2019:`\ ` table I that are computed from the global quantities.

    - $\bar{U}_{f,\text{max}}$, the maximum of
      :py:attr:`pttools.lattice.run_base.LatticeRunBase.ubarf_global` until ${t}_\text{fin}$
    - The GW power growth rate
      $$\overline{\left( \frac{\Omega_\text{gw}}{H_n t} \right)} \frac{1}{H_n R_*},$$
      where the mean is over the rows of the global quantities in the final $\Delta t = 2 R_*$ until
      ${t}_\text{fin}$, $\Omega_\text{gw} = \rho_\text{gw} / \epsilon_r$
      with $\rho_\text{gw}$ = ``GW Energy`` of the global quantities,
      and $H_n^2 = \frac{8 \pi G}{3} \epsilon_r$, see :py:func:`epsilon_r`.

    These reproduce the values of the table within 4 % for the runs of the table.

    :param run: the run
    :param t_fin: ${t}_\text{fin}$, defaults to that of :py:data:`T_FIN` for the wall speed of the run,
        or the end of the global quantities if the wall speed is not in it
    :return: the values with the keys ``Ubar_f,max`` and ``Omega_gw/(H_n t)/(H_n R_*)``
    """
    if t_fin is None:
        t_fin = T_FIN.get(run.v_wall, float(run.global_times[-1]))
    t = run.global_times
    e_r = epsilon_r(run)
    H_n = np.sqrt(8 * np.pi / 3 * e_r)
    # The small tolerance accounts for the times not being exact multiples of the time step.
    tol = 1e-3 * t_fin
    final = (t >= t_fin - 2 * run.R_star - tol) & (t <= t_fin + tol)
    omega_gw = run.globals["GW Energy"][final] / e_r
    before = t <= t_fin + tol
    return {
        "Ubar_f,max": float(np.max(run.ubarf_global[before][1:])),
        "Omega_gw/(H_n t)/(H_n R_*)": float(np.mean(omega_gw / (H_n * t[final])) / (H_n * run.R_star)),
    }
