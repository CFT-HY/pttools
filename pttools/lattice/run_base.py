r"""The parameters and global quantities of a run of a hydrodynamic lattice simulation.

The run directory contains the power spectrum files,
and optionally the input file of the simulation and the log of the global quantities (``*.out``),
see :py:mod:`pttools.lattice.readers`.
When the run is first loaded, they are converted to an HDF5 file in the run directory,
which is read afterwards, see :py:mod:`pttools.lattice.convert`.
The quantities are in lattice units, where $T_c = 1$ and $G = 1$, see :correia_2025:`\ ` table I.
The simulations are in flat space, and the Hubble rate $H_n$ only enters through the Friedmann equation
$$H_n^2 = \frac{8 \pi G}{3} \bar{\epsilon}_0,$$
where $\bar{\epsilon}_0$ is the initial mean energy density, :correia_2025:`\ ` eq. 8.
As the expansion of the universe during the phase transition is neglected,
the Hubble rate at the time of GW production is $H_* = H_n$.

The run is split into classes, each of which extends the previous one:

- :py:class:`LatticeRunBase`: the parameters of the run and the global quantities of the ``.out`` file
- :py:class:`pttools.lattice.run_spectra.LatticeRunSpectra`: the power spectra and their time series
- :py:class:`pttools.lattice.run.LatticeRun`: the spectra at a given time step, the extrapolation to late times
  and the comparison with the Sound Shell Model
"""

import functools
import logging
import os
from pathlib import Path

import numpy as np

from pttools.lattice.convert import RunData, load_run
from pttools.omgw0.factors import H
from pttools.ssm.nucleation import NucType
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)

#: Time step or an array of time steps
type Steps = float | th.FloatArr1D | th.IntArr1D
#: Gravitational constant in the lattice units of the simulations
G_LATTICE: float = 1.


class LatticeRunBase:
    r"""The parameters and global quantities of a run of a hydrodynamic lattice simulation.

    This is the base class of :py:class:`pttools.lattice.run.LatticeRun`, which should be used instead of this.

    The parameters of the run are read from the input file if it's available.
    Otherwise, they have to be given as arguments.
    The wall speed and the transition strength are not in the input file, so they have to be given as arguments.

    The quantities that depend on the energy density, such as the normalisation of the GW power spectrum,
    require either the ``.out`` file or the argument ``epsilon_0``.
    The quantities that depend on the mean enthalpy, such as $\bar{U}$, require the ``.out`` file.

    The data of the run is read from the HDF5 file in the run directory,
    which is created from the original files when the run is first loaded, see :py:mod:`pttools.lattice.convert`.
    """

    def __init__(
            self,
            path: str | os.PathLike[str],
            v_wall: float,
            alpha_n: float,
            nuc_type: NucType,
            name: str | None = None,
            label_latex: str | None = None,
            label_unicode: str | None = None,
            input_path: str | os.PathLike[str] | None = None,
            out_path: str | os.PathLike[str] | None = None,
            n_bubbles: int | None = None,
            L: float | None = None,
            dx: float | None = None,
            dt: float | None = None,
            epsilon_0: float | None = None,
            reconvert: bool = False):
        r"""Load the metadata of a simulation run.

        :param path: directory of the run
        :param v_wall: ${v}_\text{wall}$, nominal wall speed
        :param alpha_n: $\alpha_n$, transition strength at the nucleation temperature
        :param nuc_type: nucleation type of the bubbles of the simulation,
            used for the Sound Shell Model spectrum, see :py:meth:`pttools.lattice.run.LatticeRun.ssm_spectrum`
        :param name: name of the run, defaults to the name of the directory
        :param label_latex: label of the run in LaTeX format
        :param label_unicode: label of the run in Unicode format
        :param input_path: path of the input file for the conversion to HDF5,
            defaults to :py:data:`pttools.lattice.readers.INPUT_FILE_NAME` in the run directory
        :param out_path: path of the log of the global quantities for the conversion to HDF5,
            defaults to the last ``*.out`` file in the run directory
        :param n_bubbles: $N_b$, number of bubbles, overrides the input file
        :param L: $L$, side length of the lattice in lattice spacings, overrides the input file
        :param dx: $\delta x$, lattice spacing, overrides the input file
        :param dt: $\delta t$, time step, overrides the input file
        :param epsilon_0: $\bar{\epsilon}_0$, initial mean energy density, overrides the ``.out`` file
        :param reconvert: whether to convert the original files to HDF5 again, e.g. after they have changed,
            see :py:func:`pttools.lattice.convert.load_run`
        :raises FileNotFoundError: if the run directory does not exist,
            or if it has neither the HDF5 file nor the power spectrum files
        :raises ValueError: if a required parameter is neither given nor in the input file
        """
        self.path: Path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"The run directory does not exist: {self.path}")
        self.v_wall: float = v_wall
        self.alpha_n: float = alpha_n
        #: Nucleation type of the bubbles of the simulation
        self.nuc_type: NucType = nuc_type
        self.name: str = self.path.name if name is None else name
        self.label_latex: str = rf"{self.name}, $v_w={v_wall}, \alpha_n={alpha_n}$" \
            if label_latex is None else label_latex
        self.label_unicode: str = f"{self.name}, vw={v_wall}, αₙ={alpha_n}" \
            if label_unicode is None else label_unicode

        #: The data of the run as read from its HDF5 file
        self.data: RunData = load_run(self.path, input_path=input_path, out_path=out_path, reconvert=reconvert)
        #: Parameters of the input file of the simulation, or an empty dict if it's not available
        self.params: dict[str, str] = self.data.params
        if not self.params:
            logger.debug("No input file found for the run %s.", self.name)
        # The bins that contain no modes are removed, and so is the first bin, see power_spectrum().
        self._valid_bins: th.BoolArr1D = self.data.counts > 0
        self._valid_bins[0] = False

        #: $L$, side length of the lattice in lattice spacings
        self.L: float = self._param(L, "Lx", float)
        #: $\delta x$, lattice spacing
        self.dx: float = self._param(dx, "dx", float)
        #: $\delta t$, time step
        self.dt: float = self._param(dt, "dt", float)
        #: $N_b$, number of bubbles
        self.n_bubbles: int = self._param(n_bubbles, "bubbles", int)
        self._epsilon_0: float | None = epsilon_0

    def _param[T: (int, float)](self, value: T | None, key: str, type_: type[T]) -> T:
        if value is not None:
            return value
        if key not in self.params:
            raise ValueError(
                f"The parameter \"{key}\" was not given, and it's not in the input file of the run {self.name}.")
        return type_(self.params[key])

    def __repr__(self) -> str:
        """Representation of the run with its name, parameters and path."""
        return (
            f"{type(self).__name__}(name={self.name}, v_wall={self.v_wall}, alpha_n={self.alpha_n}, path={self.path})")

    # -----
    # Lattice parameters
    # -----

    @property
    def R_star(self) -> float:
        r"""$R_* = (V / N_b)^{1/3}$, mean bubble spacing."""
        return (self.volume / self.n_bubbles)**(1/3)

    @property
    def T_n(self) -> float:
        r"""$T_n$, nucleation temperature, i.e. the initial temperature of the simulation."""
        return float(self.params["Tconst"])

    @property
    def volume(self) -> float:
        r"""$V = (L \delta x)^3$, volume of the lattice."""
        return (self.L * self.dx)**3

    # -----
    # Global quantities
    # -----

    @property
    def broken_fraction(self) -> th.FloatArr1D:
        r"""$f_b(t)$, fraction of the volume in the broken phase at the rows of the ``.out`` file."""
        return self.globals["N broken"] * self.dx**3 / self.volume

    def enthalpy(self, step: Steps) -> th.FloatOrArr:
        r"""$\bar{w}$, mean enthalpy density at the given time steps."""
        return self._interp_global(step, self.enthalpy_global)

    @property
    def enthalpy_global(self) -> th.FloatArr1D:
        r"""$\bar{w}(t)$, mean enthalpy density at the rows of the ``.out`` file."""
        return (self._global_sum("rest E") + self._global_sum("pressure")) / self.volume

    @functools.cached_property
    def epsilon_0(self) -> float:
        r"""$\bar{\epsilon}_0$, initial mean energy density.

        This is the energy density of the symmetric phase at $T_n$, including the vacuum energy.
        It's the normalisation of $\Omega_\text{gw}$ and $\mathcal{P}_\text{gw}$, see :correia_2025:`\ ` eq. 25.
        """
        if self._epsilon_0 is not None:
            return self._epsilon_0
        return float(self._global_sum("rest E")[0] / self.volume)

    @property
    def global_steps(self) -> th.FloatArr1D:
        """The time steps of the rows of the ``.out`` file."""
        return self.globals["step"]

    def _global_sum(self, name: str) -> th.FloatArr1D:
        """Sum of the symmetric and broken phase contributions of a global quantity."""
        return self.globals[f"{name} sym"] + self.globals[f"{name} broken"]

    @property
    def global_times(self) -> th.FloatArr1D:
        """$t$, the times of the rows of the ``.out`` file."""
        return self.globals["time"]

    @functools.cached_property
    def globals(self) -> dict[str, th.FloatArr1D]:
        """The global quantities of the ``.out`` file.

        :raises FileNotFoundError: if the file is not available
        """
        if not self.has_globals:
            raise FileNotFoundError(
                f"The log of the global quantities (*.out) is not available for the run {self.name}.")
        return self.data.globals

    @property
    def H_star(self) -> float:
        r"""$H_* = H_n = \sqrt{\frac{8\pi G}{3} \bar{\epsilon}_0}$, Hubble rate, :correia_2025:`\ ` eq. 8.

        As the expansion of the universe during the phase transition is neglected,
        the Hubble rate at the time of GW production equals that at nucleation $H_n$.
        This is computed with :py:func:`pttools.omgw0.factors.H` with $G$ = :py:data:`G_LATTICE`.
        """
        return float(H(self.epsilon_0, G=G_LATTICE))

    @property
    def has_globals(self) -> bool:
        """Whether the log of the global quantities is available."""
        return bool(self.data.globals)

    def _interp_global(self, step: Steps, values: th.FloatArr1D) -> th.FloatOrArr:
        steps = self.global_steps
        step_arr = np.atleast_1d(step)
        outside = step_arr[(step_arr > steps[-1]) | (step_arr < steps[0])]
        if outside.size:
            logger.debug(
                "The time steps %s are outside the range of the .out file of the run %s. Using the nearest values.",
                outside.tolist(), self.name)
        return np.interp(step, steps, values)

    @property
    def kinetic_energy_fraction_global(self) -> th.FloatArr1D:
        r"""$K(t) = \frac{\langle w \gamma^2 v^2 \rangle}{\bar{\epsilon}_0}$, kinetic energy fraction of the fluid."""
        return self._global_sum("kin fluid") / self.volume / self.epsilon_0

    def mean_adiabatic_index(self, step: Steps) -> th.FloatOrArr:
        r"""$\Gamma = \bar{w} / \bar{\epsilon}_0$, mean adiabatic index, :correia_2025:`\ ` eq. 41.

        This is not exactly the adiabatic index, as $\bar{\epsilon}_0$ includes the vacuum energy.
        """
        return self.enthalpy(step) / self.epsilon_0

    @property
    def r_star(self) -> float:
        r"""$r_* = H_* R_*$, Hubble-scaled mean bubble spacing."""
        return self.H_star * self.R_star

    def time(self, step: Steps) -> th.FloatOrArr:
        r"""$t$, time at the given time steps.

        The times are read from the ``.out`` file if it's available, as the time steps are not exactly $\delta t$.
        """
        if self.has_globals:
            steps = self.global_steps
            step_arr = np.asarray(step, dtype=np.float64)
            # Extrapolate linearly beyond the last row of the .out file.
            slope = (self.global_times[-1] - self.global_times[0]) / (steps[-1] - steps[0])
            ret = np.where(
                step_arr > steps[-1],
                self.global_times[-1] + slope * (step_arr - steps[-1]),
                np.interp(step_arr, steps, self.global_times)
            )
            return float(ret) if ret.ndim == 0 else ret
        step_arr = np.asarray(step, dtype=np.float64)
        return float(self.dt * step_arr) if step_arr.ndim == 0 else self.dt * step_arr

    @property
    def ubarf_global(self) -> th.FloatArr1D:
        r"""$\bar{U}_f(t)$, RMS enthalpy-weighted fluid velocity from the global quantities.

        $$\bar{U}_f^2 = \frac{\langle w \gamma^2 v^2 \rangle}{\bar{w}}$$
        """
        return np.sqrt(self._global_sum("kin fluid") / (self._global_sum("rest E") + self._global_sum("pressure")))

    @property
    def wall_speed_global(self) -> th.FloatArr1D:
        r"""$v_w(t)$, mean wall speed estimated from the growth of the broken phase volume.

        $$v_w = \frac{3}{2} \frac{dV_b / dt}{A},$$
        where $V_b$ is the volume of the broken phase and $A$ is the number of the lattice links
        that cross the phase boundary, times $\delta x^2$.
        This is the estimate of ``plot_e.py`` of the Zenodo scripts of :correia_2025:`\ `.
        The endpoints are computed with one-sided differences.
        """
        volume = self.globals["N broken"] * self.dx**3
        area = self.globals["N links"] * self.dx**2
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.5 * np.gradient(volume, self.global_times) / area
