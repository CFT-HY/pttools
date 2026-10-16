r"""A single run of a hydrodynamic lattice simulation.

The run directory contains the power spectrum files described in :py:mod:`pttools.lattice.quantities`,
and optionally the input file of the simulation (``input-test-largerun``)
and the log of the global quantities (``*.out``, a CSV file with one row per output step).
The quantities are in lattice units, where $T_c = 1$ and $G = 1$, see :correia_2025:`\ ` table I.
The simulations are in flat space, and the Hubble rate $H_n$ only enters through the Friedmann equation
$$H_n^2 = \frac{8 \pi G}{3} \bar{\epsilon}_0,$$
where $\bar{\epsilon}_0$ is the initial mean energy density, :correia_2025:`\ ` eq. 8.
As the expansion of the universe during the phase transition is neglected,
the Hubble rate at the time of GW production is $H_* = H_n$.
"""

import functools
import logging
import os
from pathlib import Path
import typing as tp

import numpy as np
import pandas as pd

from pttools.lattice.quantities import Quantity, integral_scale, mean_square, read_power_spectrum
from pttools.omgw0.factors import H
import pttools.type_hints as th

if tp.TYPE_CHECKING:
    from pttools.lattice.spectrum import LatticeSpectrum

logger: logging.Logger = logging.getLogger(__name__)

#: Time step or an array of time steps
type Steps = float | th.FloatArr1D | th.IntArr1D

#: Default name of the input file of the simulation
INPUT_FILE_NAME: str = "input-test-largerun"
#: Gravitational constant in the lattice units of the simulations
G_LATTICE: float = 1.


def parse_input_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse the input file of the simulation.

    The file has one ``key value`` pair per line, and comments start with ``#``.

    :param path: path of the input file
    :return: the parameters as strings
    """
    params = {}
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            content = line.split("#", 1)[0].strip()
            if not content:
                continue
            parts = content.split(maxsplit=1)
            params[parts[0]] = parts[1] if len(parts) > 1 else ""
    return params


def read_globals(path: str | os.PathLike[str]) -> dict[str, th.FloatArr1D]:
    """Read the CSV log of the global quantities.

    The file is read with the C engine of :py:func:`pandas.read_csv`.
    If the run was interrupted, the last row may be incomplete, in which case it's ignored with a warning.

    :param path: path of the ``.out`` file
    :return: the columns of the file as arrays, with the column names as keys
    """
    data = pd.read_csv(path, engine="c", dtype=np.float64, skipinitialspace=True)
    data.columns = [str(name).strip() for name in data.columns]
    # The missing values of an incomplete row are read as nan.
    if len(data) > 0 and data.iloc[-1].isna().any():
        logger.warning("Ignoring the incomplete last row of %s, as the run was probably interrupted.", path)
        data = data.iloc[:-1]
    return {name: data[name].to_numpy(dtype=np.float64) for name in data.columns}


class LatticeRun:
    r"""A run of a hydrodynamic lattice simulation.

    The parameters of the run are read from the input file if it's available.
    Otherwise, they have to be given as arguments.
    The wall speed and the transition strength are not in the input file, so they have to be given as arguments.

    The quantities that depend on the energy density, such as the normalisation of the GW power spectrum,
    require either the ``.out`` file or the argument ``epsilon_0``.
    The quantities that depend on the mean enthalpy, such as $\bar{U}$, require the ``.out`` file.
    """

    def __init__(
            self,
            path: str | os.PathLike[str],
            v_wall: float,
            alpha_n: float,
            name: str | None = None,
            label_latex: str | None = None,
            label_unicode: str | None = None,
            input_path: str | os.PathLike[str] | None = None,
            out_path: str | os.PathLike[str] | None = None,
            n_bubbles: int | None = None,
            L: float | None = None,
            dx: float | None = None,
            dt: float | None = None,
            epsilon_0: float | None = None):
        r"""Load the metadata of a simulation run.

        :param path: directory of the run
        :param v_wall: ${v}_\text{wall}$, nominal wall speed
        :param alpha_n: $\alpha_n$, transition strength at the nucleation temperature
        :param name: name of the run, defaults to the name of the directory
        :param label_latex: label of the run in LaTeX format
        :param label_unicode: label of the run in Unicode format
        :param input_path: path of the input file, defaults to ``input-test-largerun`` in the run directory
        :param out_path: path of the log of the global quantities,
            defaults to the last ``*.out`` file in the run directory
        :param n_bubbles: $N_b$, number of bubbles, overrides the input file
        :param L: $L$, side length of the lattice in lattice spacings, overrides the input file
        :param dx: $\delta x$, lattice spacing, overrides the input file
        :param dt: $\delta t$, time step, overrides the input file
        :param epsilon_0: $\bar{\epsilon}_0$, initial mean energy density, overrides the ``.out`` file
        :raises FileNotFoundError: if the run directory does not exist
        :raises ValueError: if a required parameter is neither given nor in the input file
        """
        self.path: Path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"The run directory does not exist: {self.path}")
        self.v_wall: float = v_wall
        self.alpha_n: float = alpha_n
        self.name: str = self.path.name if name is None else name
        self.label_latex: str = rf"{self.name}, $v_w={v_wall}, \alpha_n={alpha_n}$" \
            if label_latex is None else label_latex
        self.label_unicode: str = f"{self.name}, vw={v_wall}, αₙ={alpha_n}" \
            if label_unicode is None else label_unicode

        input_path = self.path / INPUT_FILE_NAME if input_path is None else Path(input_path)
        #: Parameters of the input file of the simulation, or an empty dict if it's not available
        self.params: dict[str, str] = parse_input_file(input_path) if input_path.is_file() else {}
        if not self.params:
            logger.debug("No input file found for the run %s at %s.", self.name, input_path)

        if out_path is None:
            out_files = sorted(self.path.glob("*.out"))
            #: Path of the log of the global quantities, or None if it's not available
            self.out_path: Path | None = out_files[-1] if out_files else None
        else:
            self.out_path = Path(out_path)

        #: $L$, side length of the lattice in lattice spacings
        self.L: float = self._param(L, "Lx", float)
        #: $\delta x$, lattice spacing
        self.dx: float = self._param(dx, "dx", float)
        #: $\delta t$, time step
        self.dt: float = self._param(dt, "dt", float)
        #: $N_b$, number of bubbles
        self.n_bubbles: int = self._param(n_bubbles, "bubbles", int)
        self._epsilon_0: float | None = epsilon_0
        self._steps_cache: dict[Quantity, th.IntArr1D] = {}
        self._spectrum_cache: dict[tuple[Quantity, int], tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]] = {}

    def _param[T: (int, float)](self, value: T | None, key: str, type_: type[T]) -> T:
        if value is not None:
            return value
        if key not in self.params:
            raise ValueError(
                f"The parameter \"{key}\" was not given, and it's not in the input file of the run {self.name}.")
        return type_(self.params[key])

    def __repr__(self) -> str:
        """Representation of the run with its name, parameters and path."""
        return f"LatticeRun(name={self.name}, v_wall={self.v_wall}, alpha_n={self.alpha_n}, path={self.path})"

    # -----
    # Lattice parameters
    # -----

    @property
    def volume(self) -> float:
        r"""$V = (L \delta x)^3$, volume of the lattice."""
        return (self.L * self.dx)**3

    @property
    def R_star(self) -> float:
        r"""$R_* = (V / N_b)^{1/3}$, mean bubble spacing."""
        return (self.volume / self.n_bubbles)**(1/3)

    @property
    def T_n(self) -> float:
        r"""$T_n$, nucleation temperature, i.e. the initial temperature of the simulation."""
        return float(self.params["Tconst"])

    # -----
    # Global quantities
    # -----

    @property
    def has_globals(self) -> bool:
        """Whether the log of the global quantities is available."""
        return self.out_path is not None and self.out_path.is_file()

    @functools.cached_property
    def globals(self) -> dict[str, th.FloatArr1D]:
        """The global quantities of the ``.out`` file.

        :raises FileNotFoundError: if the file is not available
        """
        if not self.has_globals:
            raise FileNotFoundError(
                f"The log of the global quantities (*.out) is not available for the run {self.name}.")
        return read_globals(self.out_path)  # pyrefly: ignore[bad-argument-type]

    def _global_sum(self, name: str) -> th.FloatArr1D:
        """Sum of the symmetric and broken phase contributions of a global quantity."""
        return self.globals[f"{name} sym"] + self.globals[f"{name} broken"]

    @property
    def global_steps(self) -> th.FloatArr1D:
        """The time steps of the rows of the ``.out`` file."""
        return self.globals["step"]

    @property
    def global_times(self) -> th.FloatArr1D:
        """$t$, the times of the rows of the ``.out`` file."""
        return self.globals["time"]

    def _interp_global(self, step: Steps, values: th.FloatArr1D) -> th.FloatOrArr:
        steps = self.global_steps
        if np.any(np.asarray(step) > steps[-1]) or np.any(np.asarray(step) < steps[0]):
            logger.debug(
                "Step %s is outside the range of the .out file of the run %s. Using the nearest value.",
                step, self.name)
        return np.interp(step, steps, values)

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
    def H_star(self) -> float:
        r"""$H_* = H_n = \sqrt{\frac{8\pi G}{3} \bar{\epsilon}_0}$, Hubble rate, :correia_2025:`\ ` eq. 8.

        As the expansion of the universe during the phase transition is neglected,
        the Hubble rate at the time of GW production equals that at nucleation $H_n$.
        This is computed with :py:func:`pttools.omgw0.factors.H` with $G$ = :py:data:`G_LATTICE`.
        """
        return float(H(self.epsilon_0, G=G_LATTICE))

    @property
    def r_star(self) -> float:
        r"""$r_* = H_* R_*$, Hubble-scaled mean bubble spacing."""
        return self.H_star * self.R_star

    @property
    def enthalpy_global(self) -> th.FloatArr1D:
        r"""$\bar{w}(t)$, mean enthalpy density at the rows of the ``.out`` file."""
        return (self._global_sum("rest E") + self._global_sum("pressure")) / self.volume

    def enthalpy(self, step: Steps) -> th.FloatOrArr:
        r"""$\bar{w}$, mean enthalpy density at the given time steps."""
        return self._interp_global(step, self.enthalpy_global)

    def mean_adiabatic_index(self, step: Steps) -> th.FloatOrArr:
        r"""$\Gamma = \bar{w} / \bar{\epsilon}_0$, mean adiabatic index, :correia_2025:`\ ` eq. 41.

        This is not exactly the adiabatic index, as $\bar{\epsilon}_0$ includes the vacuum energy.
        """
        return self.enthalpy(step) / self.epsilon_0

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
    def broken_fraction(self) -> th.FloatArr1D:
        r"""$f_b(t)$, fraction of the volume in the broken phase at the rows of the ``.out`` file."""
        return self.globals["N broken"] * self.dx**3 / self.volume

    @property
    def ubarf_global(self) -> th.FloatArr1D:
        r"""$\bar{U}_f(t)$, RMS enthalpy-weighted fluid velocity from the global quantities.

        $$\bar{U}_f^2 = \frac{\langle w \gamma^2 v^2 \rangle}{\bar{w}}$$
        """
        return np.sqrt(self._global_sum("kin fluid") / (self._global_sum("rest E") + self._global_sum("pressure")))

    @property
    def kinetic_energy_fraction_global(self) -> th.FloatArr1D:
        r"""$K(t) = \frac{\langle w \gamma^2 v^2 \rangle}{\bar{\epsilon}_0}$, kinetic energy fraction of the fluid."""
        return self._global_sum("kin fluid") / self.volume / self.epsilon_0

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

    # -----
    # Power spectra
    # -----

    def steps(self, quantity: Quantity = Quantity.GW_TT) -> th.IntArr1D:
        """The time steps at which the power spectrum of the given quantity is available.

        :param quantity: the quantity
        :return: the time steps in increasing order
        """
        quantity = Quantity(quantity)
        if quantity not in self._steps_cache:
            prefix = quantity.file_prefix
            steps = sorted(int(path.stem.removeprefix(prefix)) for path in self.path.glob(f"{prefix}*.txt"))
            self._steps_cache[quantity] = np.array(steps, dtype=np.int_)
        return self._steps_cache[quantity]

    def spectrum_path(self, quantity: Quantity, step: int) -> Path:
        """Path of the power spectrum file of the given quantity and time step."""
        return self.path / f"{Quantity(quantity).file_prefix}{step}.txt"

    def power_spectrum(
            self,
            quantity: Quantity,
            step: int) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]:
        r"""Read the raw power spectrum of the given quantity at the given time step.

        The bins that contain no modes are removed.

        :param quantity: the quantity
        :param step: the time step
        :return: $k$, $\mathcal{P}(k)$ in lattice units, and the number of modes in each bin
        """
        key = (Quantity(quantity), int(step))
        if key not in self._spectrum_cache:
            k, p, counts = read_power_spectrum(self.spectrum_path(*key))
            valid = counts > 0
            self._spectrum_cache[key] = k[valid], p[valid], counts[valid]
        return self._spectrum_cache[key]

    def pow_normalised(self, quantity: Quantity, step: int) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        r"""Power spectrum normalised to a dimensionless quantity, where applicable.

        - $X$ is divided by $\bar{w}$ to give the spectrum of $U^i$ of :correia_2025:`\ ` eq. 20.
        - $\mathcal{P}_\text{gw}$ is divided by $\bar{\epsilon}_0$, :correia_2025:`\ ` eq. 25.
        - The other quantities are returned as they are.

        :param quantity: the quantity
        :param step: the time step
        :return: $k$ and the normalised power spectrum
        """
        quantity = Quantity(quantity)
        k, p, _ = self.power_spectrum(quantity, step)
        if quantity in (Quantity.X_DIV, Quantity.X_ROT, Quantity.X_TOT):
            return k, p / self.enthalpy(step)
        if quantity == Quantity.GW_TT:
            return k, p / self.epsilon_0
        return k, p

    def rms(self, quantity: Quantity, step: int) -> float:
        r"""RMS value of the normalised field, e.g. $\bar{U}_\parallel$ for :py:attr:`Quantity.X_DIV`.

        $$\bar{Y} = \sqrt{\int \frac{dk}{k} \mathcal{P}_Y(k)}$$
        """
        return np.sqrt(mean_square(*self.pow_normalised(quantity, step)))

    def integral_scale(self, quantity: Quantity, step: int) -> float:
        r"""$\xi$, integral scale of the given quantity, :correia_2025:`\ ` eq. 39."""
        k, p, _ = self.power_spectrum(quantity, step)
        return integral_scale(k, p)

    def omega_gw(self, step: int) -> float:
        r"""$\Omega_\text{gw} = \int \frac{dk}{k} \mathcal{P}_\text{gw}(k)$, GW energy density fraction."""
        return mean_square(*self.pow_normalised(Quantity.GW_TT, step))

    def pow_gw_rate(self, step: int, step_prev: int) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        r"""Growth rate of the GW power spectrum from the difference of two time steps.

        $$\frac{1}{H_n R_*} \frac{\Delta \mathcal{P}_\text{gw}}{H_n \Delta t}$$
        :correia_2025:`\ ` fig. 5.

        :param step: the later time step
        :param step_prev: the earlier time step
        :return: $k$ and the growth rate
        """
        k, p = self.pow_normalised(Quantity.GW_TT, step)
        k_prev, p_prev = self.pow_normalised(Quantity.GW_TT, step_prev)
        if not np.array_equal(k, k_prev):
            raise ValueError("The wavenumbers of the two time steps differ.")
        delta_t = self.time(step) - self.time(step_prev)
        return k, (p - p_prev) / (self.H_star * delta_t) / self.r_star

    def spectrum(self, step: int, step_prev: int | None = None, **kwargs: tp.Any) -> "LatticeSpectrum":
        """Create a spectrum object for the given time step.

        See :py:class:`pttools.lattice.spectrum.LatticeSpectrum`.

        :param step: the time step
        :param step_prev: the earlier time step for computing the GW power growth rate
        :param kwargs: additional arguments for :py:class:`pttools.lattice.spectrum.LatticeSpectrum`,
            e.g. ``r_star``, ``T_star``, ``g_star`` and ``gs_star``
        :return: the spectrum
        """
        from pttools.lattice.spectrum import LatticeSpectrum  # noqa: PLC0415
        return LatticeSpectrum(self, step=step, step_prev=step_prev, **kwargs)

    # -----
    # Time series
    # -----

    def _series_steps(self, quantity: Quantity, steps: tp.Iterable[int] | None) -> th.IntArr1D:
        return self.steps(quantity) if steps is None else np.asarray(steps, dtype=np.int_)

    def rms_series(
            self,
            quantity: Quantity = Quantity.X_DIV,
            steps: tp.Iterable[int] | None = None) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        """RMS value of the given quantity as a function of time, see :py:meth:`rms`.

        :param quantity: the quantity
        :param steps: the time steps, defaults to all the available steps
        :return: the times and the RMS values
        """
        steps_arr = self._series_steps(quantity, steps)
        return np.asarray(self.time(steps_arr)), np.array([self.rms(quantity, int(step)) for step in steps_arr])

    def integral_scale_series(
            self,
            quantity: Quantity = Quantity.X_DIV,
            steps: tp.Iterable[int] | None = None) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        """Integral scale of the given quantity as a function of time, see :py:meth:`integral_scale`.

        :param quantity: the quantity
        :param steps: the time steps, defaults to all the available steps
        :return: the times and the integral scales
        """
        steps_arr = self._series_steps(quantity, steps)
        return np.asarray(self.time(steps_arr)), \
            np.array([self.integral_scale(quantity, int(step)) for step in steps_arr])

    def omega_gw_series(self, steps: tp.Iterable[int] | None = None) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        r"""$\Omega_\text{gw}(t)$, GW energy density fraction as a function of time.

        :param steps: the time steps, defaults to all the available steps
        :return: the times and $\Omega_\text{gw}$
        """
        steps_arr = self._series_steps(Quantity.GW_TT, steps)
        return np.asarray(self.time(steps_arr)), np.array([self.omega_gw(int(step)) for step in steps_arr])

    @functools.cached_property
    def ubar_par_max(self) -> tuple[int, float]:
        r"""The time step and the value of the maximum of $\bar{U}_\parallel$, :correia_2025:`\ ` table IV."""
        steps = self.steps(Quantity.X_DIV)
        _, ubar = self.rms_series(Quantity.X_DIV, steps)
        i_max = int(np.argmax(ubar))
        return int(steps[i_max]), float(ubar[i_max])

    @property
    def t_sh(self) -> float:
        r"""$t_\text{sh} = R_* / \bar{U}_{\parallel,*}$, shock formation time, :correia_2025:`\ ` eq. 28."""
        return self.R_star / self.ubar_par_max[1]

    def kinetic_energy_fraction_par(self, step: int) -> float:
        r"""$K_\parallel = \Gamma \bar{U}_\parallel^2$, compressional kinetic energy fraction.

        :correia_2025:`\ ` eq. 41.
        """
        return float(self.mean_adiabatic_index(step) * self.rms(Quantity.X_DIV, step)**2)
