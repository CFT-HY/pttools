r"""A single run of a hydrodynamic lattice simulation.

The run directory contains the power spectrum files,
and optionally the input file of the simulation and the log of the global quantities (``*.out``),
see :py:mod:`pttools.lattice.readers`.
When the run is first loaded, they are converted to an HDF5 file in the run directory,
which is read afterwards, see :py:mod:`pttools.lattice.convert`.
The power spectra of each quantity are stored as a 2D array with the shape (time steps, bins),
and therefore the methods that take time steps accept both a single time step and an array of them.
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
import numpy.typing as npt

from pttools.bubble import Bubble
from pttools.lattice.asymptote import (
    DEFAULT_RUNNING_MEAN_WINDOW,
    PowGwAsymptoteFit,
    fit_pow_gw_modes,
    gw_growth_index_prediction,
)
from pttools.lattice.convert import RunData, load_run
from pttools.lattice.quantities import Quantity, integral_scale, mean_square
from pttools.lattice.spectrum import LatticeSpectrum
from pttools.math.asymptote import AsymptoteFit, fit_asymptote
from pttools.math.numerics import running_mean
from pttools.math.power_law import PowerLawFit, fit_power_law
from pttools.models import BagModel, Model
from pttools.omgw0.factors import H
from pttools.ssm.nucleation import NucType
from pttools.ssm.spectrum import SSMSpectrum
from pttools.ssm.suppression import SuppressionMethod
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)

#: Time step or an array of time steps
type Steps = float | th.FloatArr1D | th.IntArr1D

#: Default $z = kR_*$ points of the Sound Shell Model spectra for comparisons with simulations.
#: The range is wide enough to cover the simulation data and to converge the integrals over $z$.
DEFAULT_Y_LATTICE: th.FloatArr1D = np.logspace(-1, 4, 1000)
#: Gravitational constant in the lattice units of the simulations
G_LATTICE: float = 1.


class LatticeRun:
    r"""A run of a hydrodynamic lattice simulation.

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
        self._power_cache: dict[Quantity, th.FloatArr2D] = {}
        self._fit_cache: dict[tuple[tp.Any, ...], tp.Any] = {}

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
        return bool(self.data.globals)

    @functools.cached_property
    def globals(self) -> dict[str, th.FloatArr1D]:
        """The global quantities of the ``.out`` file.

        :raises FileNotFoundError: if the file is not available
        """
        if not self.has_globals:
            raise FileNotFoundError(
                f"The log of the global quantities (*.out) is not available for the run {self.name}.")
        return self.data.globals

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

    @property
    def counts(self) -> th.FloatArr1D:
        """The numbers of the lattice modes in the bins of the power spectra, see :py:meth:`power_spectrum`."""
        return self.data.counts[self._valid_bins]

    @property
    def k(self) -> th.FloatArr1D:
        """$k$, wavenumbers of the bins of the power spectra in lattice units, see :py:meth:`power_spectrum`."""
        return self.data.k[self._valid_bins]

    def steps(self, quantity: Quantity = Quantity.GW_TT) -> th.IntArr1D:
        """The time steps at which the power spectrum of the given quantity is available.

        :param quantity: the quantity
        :return: the time steps in increasing order, or an empty array if the quantity is not available
        """
        return self.data.steps.get(Quantity(quantity), np.array([], dtype=np.int_))

    def _steps_or_all(self, quantity: Quantity, steps: npt.ArrayLike | None) -> th.IntArr1D:
        """The given time steps as an array, or all the time steps of the quantity if not given."""
        return self.steps(quantity) if steps is None else np.asarray(steps, dtype=np.int_)

    def _step_indices(self, quantity: Quantity, steps: npt.ArrayLike) -> int | th.IntArr1D:
        """Indices of the given time steps in the power spectra of the quantity.

        :raises ValueError: if the power spectrum is not available at some of the time steps
        """
        available = self.steps(quantity)
        steps_arr = np.asarray(steps, dtype=np.int_)
        indices = np.searchsorted(available, steps_arr)
        found = (indices < available.size) & (available[np.minimum(indices, available.size - 1)] == steps_arr) \
            if available.size else np.zeros(steps_arr.shape, dtype=np.bool_)
        if not np.all(found):
            raise ValueError(
                f"The power spectrum of {Quantity(quantity).value} of the run {self.name} is not available "
                f"at the time steps {np.atleast_1d(steps_arr)[~np.atleast_1d(found)].tolist()}.")
        return int(indices) if indices.ndim == 0 else indices

    def _power(self, quantity: Quantity) -> th.FloatArr2D:
        """The power spectra of the valid bins of the quantity at all its time steps, read once and cached."""
        quantity = Quantity(quantity)
        if quantity not in self._power_cache:
            self._power_cache[quantity] = self.data.power(quantity)[:, self._valid_bins]
        return self._power_cache[quantity]

    @tp.overload
    def power_spectrum(
            self, quantity: Quantity, steps: int) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]: ...

    @tp.overload
    def power_spectrum(
            self,
            quantity: Quantity,
            steps: npt.ArrayLike | None = None) -> tuple[th.FloatArr1D, th.FloatArr2D, th.FloatArr1D]: ...

    def power_spectrum(
            self,
            quantity: Quantity,
            steps: npt.ArrayLike | None = None) -> tuple[th.FloatArr1D, th.FloatArr, th.FloatArr1D]:
        r"""The raw power spectra of the given quantity at the given time steps.

        The bins that contain no modes are removed, and so is the first bin.
        The first bin $k \in [0, 2\pi / L)$ contains only the zero mode $k = 0$,
        but the files give it at the centre of the bin.
        The zero mode is not a wave, and e.g. for the energy density it's the mean,
        so including it would distort the integrals over $k$, such as $\Omega_\text{gw}$ and $\xi$.
        The scripts of :correia_2025:`\ ` also exclude it.

        :param quantity: the quantity
        :param steps: a time step, an array of time steps, or None for all the time steps of the quantity
        :return: $k$, $\mathcal{P}(k)$ in lattice units, and the number of modes in each bin.
            For an array of time steps, $\mathcal{P}(k)$ has the shape (time steps, bins).
        :raises ValueError: if the power spectrum is not available at some of the time steps
        """
        power = self._power(quantity)
        if steps is not None:
            power = power[self._step_indices(quantity, steps)]
        return self.k, power, self.counts

    @tp.overload
    def pow_normalised(self, quantity: Quantity, steps: int) -> tuple[th.FloatArr1D, th.FloatArr1D]: ...

    @tp.overload
    def pow_normalised(
            self, quantity: Quantity, steps: npt.ArrayLike | None = None) -> tuple[th.FloatArr1D, th.FloatArr2D]: ...

    def pow_normalised(
            self, quantity: Quantity, steps: npt.ArrayLike | None = None) -> tuple[th.FloatArr1D, th.FloatArr]:
        r"""Power spectra normalised to a dimensionless quantity, where applicable.

        - $X$ is divided by $\bar{w}$ to give the spectrum of $U^i$ of :correia_2025:`\ ` eq. 20.
        - $\mathcal{P}_\text{gw}$ is divided by $\bar{\epsilon}_0$, :correia_2025:`\ ` eq. 25.
        - The other quantities are returned as they are.

        :param quantity: the quantity
        :param steps: a time step, an array of time steps, or None for all the time steps of the quantity
        :return: $k$ and the normalised power spectra, see :py:meth:`power_spectrum`
        """
        quantity = Quantity(quantity)
        k, p, _ = self.power_spectrum(quantity, steps)
        if quantity in (Quantity.X_DIV, Quantity.X_ROT, Quantity.X_TOT):
            enthalpy = np.asarray(self.enthalpy(self._steps_or_all(quantity, steps)))
            return k, p / enthalpy[..., np.newaxis]
        if quantity == Quantity.GW_TT:
            return k, p / self.epsilon_0
        return k, p

    @tp.overload
    def rms(self, quantity: Quantity, steps: int) -> float: ...

    @tp.overload
    def rms(self, quantity: Quantity, steps: npt.ArrayLike | None = None) -> th.FloatArr1D: ...

    def rms(self, quantity: Quantity, steps: npt.ArrayLike | None = None) -> float | th.FloatArr1D:
        r"""RMS value of the normalised field, e.g. $\bar{U}_\parallel$ for :py:attr:`Quantity.X_DIV`.

        $$\bar{Y} = \sqrt{\int \frac{dk}{k} \mathcal{P}_Y(k)}$$

        :param quantity: the quantity
        :param steps: a time step, an array of time steps, or None for all the time steps of the quantity
        :return: the RMS value at each time step
        """
        return np.sqrt(mean_square(*self.pow_normalised(quantity, steps)))

    @tp.overload
    def integral_scale(self, quantity: Quantity, steps: int) -> float: ...

    @tp.overload
    def integral_scale(self, quantity: Quantity, steps: npt.ArrayLike | None = None) -> th.FloatArr1D: ...

    def integral_scale(self, quantity: Quantity, steps: npt.ArrayLike | None = None) -> float | th.FloatArr1D:
        r"""$\xi$, integral scale of the given quantity, :correia_2025:`\ ` eq. 39.

        :param quantity: the quantity
        :param steps: a time step, an array of time steps, or None for all the time steps of the quantity
        :return: $\xi$ at each time step, or nan where the power spectrum is zero
        """
        k, p, _ = self.power_spectrum(quantity, steps)
        return integral_scale(k, p)

    @tp.overload
    def omega_gw(self, steps: int) -> float: ...

    @tp.overload
    def omega_gw(self, steps: npt.ArrayLike | None = None) -> th.FloatArr1D: ...

    def omega_gw(self, steps: npt.ArrayLike | None = None) -> float | th.FloatArr1D:
        r"""$\Omega_\text{gw} = \int \frac{dk}{k} \mathcal{P}_\text{gw}(k)$, GW energy density fraction.

        :param steps: a time step, an array of time steps, or None for all the time steps of the GW power spectrum
        :return: $\Omega_\text{gw}$ at each time step
        """
        return mean_square(*self.pow_normalised(Quantity.GW_TT, steps))

    def pow_gw_rate(self, step: int, step_prev: int) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        r"""Growth rate of the GW power spectrum from the difference of two time steps.

        $$\frac{1}{H_n R_*} \frac{\Delta \mathcal{P}_\text{gw}}{H_n \Delta t}$$
        :correia_2025:`\ ` fig. 5.

        :param step: the later time step
        :param step_prev: the earlier time step
        :return: $k$ and the growth rate
        """
        k, p = self.pow_normalised(Quantity.GW_TT, [step_prev, step])
        delta_t = self.time(step) - self.time(step_prev)
        return k, (p[1] - p[0]) / (self.H_star * delta_t) / self.r_star

    def spectrum(self, step: int, step_prev: int | None = None, **kwargs: tp.Any) -> LatticeSpectrum:
        """Create a spectrum object for the given time step.

        See :py:class:`pttools.lattice.spectrum.LatticeSpectrum`.

        :param step: the time step
        :param step_prev: the earlier time step for computing the GW power growth rate
        :param kwargs: additional arguments for :py:class:`pttools.lattice.spectrum.LatticeSpectrum`,
            e.g. ``r_star``, ``T_star``, ``g_star`` and ``gs_star``
        :return: the spectrum
        """
        return LatticeSpectrum(self, step=step, step_prev=step_prev, **kwargs)

    # -----
    # Time series
    # -----

    def rms_series(
            self,
            quantity: Quantity = Quantity.X_DIV,
            steps: npt.ArrayLike | None = None) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        """RMS value of the given quantity as a function of time, see :py:meth:`rms`.

        :param quantity: the quantity
        :param steps: the time steps, defaults to all the available steps
        :return: the times and the RMS values
        """
        steps_arr = self._steps_or_all(quantity, steps)
        return np.asarray(self.time(steps_arr)), np.asarray(self.rms(quantity, steps_arr))

    def integral_scale_series(
            self,
            quantity: Quantity = Quantity.X_DIV,
            steps: npt.ArrayLike | None = None) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        """Integral scale of the given quantity as a function of time, see :py:meth:`integral_scale`.

        :param quantity: the quantity
        :param steps: the time steps, defaults to all the available steps
        :return: the times and the integral scales
        """
        steps_arr = self._steps_or_all(quantity, steps)
        return np.asarray(self.time(steps_arr)), np.asarray(self.integral_scale(quantity, steps_arr))

    def omega_gw_series(self, steps: npt.ArrayLike | None = None) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        r"""$\Omega_\text{gw}(t)$, GW energy density fraction as a function of time.

        :param steps: the time steps, defaults to all the available steps
        :return: the times and $\Omega_\text{gw}$
        """
        steps_arr = self._steps_or_all(Quantity.GW_TT, steps)
        return np.asarray(self.time(steps_arr)), np.asarray(self.omega_gw(steps_arr))

    def omega_gw_rate_series(
            self,
            delta_steps: int,
            steps: npt.ArrayLike | None = None) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        r"""$\frac{1}{H_n R_*} \frac{\Delta \Omega_\text{gw}}{H_n \Delta t}$, GW energy density growth rate.

        The growth rate of the GW energy density fraction is computed as a finite difference as a function of time.

        :param delta_steps: the difference of the time steps of the finite difference
        :param steps: the later time steps, defaults to all the available steps for which
            the earlier step is also available
        :return: the midpoint times of the time step pairs and the growth rates
        """
        if steps is None:
            available = self.steps(Quantity.GW_TT)
            steps_arr = available[np.isin(available - delta_steps, available)]
        else:
            steps_arr = np.asarray(steps, dtype=np.int_)
        t1 = np.asarray(self.time(steps_arr))
        t0 = np.asarray(self.time(steps_arr - delta_steps))
        omega1 = np.asarray(self.omega_gw(steps_arr))
        omega0 = np.asarray(self.omega_gw(steps_arr - delta_steps))
        return 0.5 * (t0 + t1), (omega1 - omega0) / (self.H_star * (t1 - t0)) / self.r_star

    @functools.cached_property
    def ubar_par_max(self) -> tuple[int, float]:
        r"""The time step and the value of the maximum of $\bar{U}_\parallel$, :correia_2025:`\ ` table IV."""
        steps = self.steps(Quantity.X_DIV)
        ubar = np.asarray(self.rms(Quantity.X_DIV))
        i_max = int(np.argmax(ubar))
        return int(steps[i_max]), float(ubar[i_max])

    @property
    def t_sh(self) -> float:
        r"""${t}_\text{sh} = R_* / \bar{U}_{\parallel,*}$, shock formation time, :correia_2025:`\ ` eq. 28."""
        return self.R_star / self.ubar_par_max[1]

    def kinetic_energy_fraction_par(self, step: int) -> float:
        r"""$K_\parallel = \Gamma \bar{U}_\parallel^2$, compressional kinetic energy fraction.

        :correia_2025:`\ ` eq. 41.
        """
        return float(self.mean_adiabatic_index(step) * self.rms(Quantity.X_DIV, step)**2)

    # -----
    # Extrapolation to late times
    # -----

    @property
    def t_max(self) -> float:
        r"""${t}_\text{max}$, time of the maximum of $\bar{U}_\parallel$, see :py:attr:`ubar_par_max`."""
        return float(self.time(self.ubar_par_max[0]))

    @property
    def gw_asymptote_scale(self) -> float:
        r"""Scale of the GW power in the plots of its extrapolation, :correia_2025:`\ ` fig. 6.

        $$(\Gamma \bar{U}_\parallel^2)_*^2 (H_n R_*) (H_n {t}_\text{sh}),$$
        where $(\Gamma \bar{U}_\parallel^2)_*$ is :py:meth:`kinetic_energy_fraction_par`
        at the maximum of $\bar{U}_\parallel$.
        This is the scale of the axes of the figure and of its scripts.
        The text of the article gives the scale as $(\Gamma \bar{U}_\parallel^2)_*^2 (H_n R_*)^2$.

        The values of the axes of the figure differ from those of this scale due to two issues of the scripts:
        $\Gamma$ is taken from a row of the log of the global quantities whose index is that of the maximum
        in the list of the time steps, i.e. at $t T_c = 200$ and 180 instead of ${t}_\text{max} T_c = 800$ and 1440,
        which gives $\Gamma = 0.79$ and 0.89 instead of 0.99 and 1.03, and therefore larger values of
        $\Omega_\text{gw}$ by factors of 1.58 and 1.36.
        Furthermore, $\mathcal{P}_\text{gw}(k)$ is not divided by $\bar{\epsilon}_0$ (0.167 and 0.204),
        unlike $\Omega_\text{gw}$.
        These do not affect the fitted indices, the shapes of the curves or the ratios of the values.
        """
        return self.kinetic_energy_fraction_par(self.ubar_par_max[0])**2 * self.r_star * self.H_star * self.t_sh

    def gw_asymptote_summary(self, step: int, step_end: int | None = None) -> dict[str, float]:
        r"""The fitted indices and the asymptotic GW power, cf. :correia_2025:`\ ` tables IV and V.

        The values are those of :py:meth:`kinetic_energy_decay_index` ($\zeta$),
        :py:meth:`integral_scale_growth_index` ($\lambda$), :py:meth:`gw_growth_index_prediction`
        and :py:meth:`omega_gw_asymptote` (${d}_\text{tot}$ and $\Omega_\text{gw}^\infty$),
        where $\Omega_\text{gw}^\infty$ is scaled by :py:attr:`gw_asymptote_scale`.
        The ratio $\Omega_\text{gw}^\infty / \Omega_\text{gw}(t)$ is evaluated at the given time step,
        see :py:meth:`pttools.lattice.spectrum.LatticeSpectrum.omega_gw_asymptote_ratio`.

        :param step: the time step of the ratio $\Omega_\text{gw}^\infty / \Omega_\text{gw}(t)$
        :param step_end: last time step of the fits, defaults to the last available step
        :return: the values, and the uncertainties of the fitted values with the suffix ``_err``
        """
        zeta = self.kinetic_energy_decay_index(step_end)
        lambda_ = self.integral_scale_growth_index(step_end)
        prediction, prediction_err = self.gw_growth_index_prediction(step_end)
        omega_fit = self.omega_gw_asymptote(step_end=step_end)
        spectrum = self.spectrum(step)
        return {
            "zeta": zeta.index,
            "zeta_err": zeta.index_err,
            "lambda": lambda_.index,
            "lambda_err": lambda_.index_err,
            "2 zeta + 2 lambda - 1": prediction,
            "2 zeta + 2 lambda - 1_err": prediction_err,
            "d_tot": omega_fit.index,
            "d_tot_err": omega_fit.index_err,
            "Omega_inf (scaled)": omega_fit.value / self.gw_asymptote_scale,
            "t [1/T_c]": spectrum.t,
            "Omega_inf / Omega(t)": spectrum.omega_gw_asymptote_ratio(step_end=step_end),
        }

    def _fit_steps(self, quantity: Quantity, step_end: int | None, after_max: bool = False) -> th.IntArr1D:
        r"""The time steps of the fits, optionally only those after the maximum of $\bar{U}_\parallel$."""
        steps = self.steps(quantity)
        if step_end is not None:
            steps = steps[steps <= step_end]
        if after_max:
            steps = steps[steps >= self.ubar_par_max[0]]
        return steps

    def kinetic_energy_decay_index(self, step_end: int | None = None) -> PowerLawFit:
        r"""$\zeta$, decay index of the kinetic energy, where $K^2 \propto t^{-2\zeta}$ at late times.

        Fitted to $\bar{U}_\parallel(t)$ after its maximum, see :py:mod:`pttools.lattice.asymptote`.

        :param step_end: last time step of the fit, defaults to the last available step
        :return: the fit
        """
        key = ("zeta", step_end)
        if key not in self._fit_cache:
            t, ubar = self.rms_series(Quantity.X_DIV, self._fit_steps(Quantity.X_DIV, step_end, after_max=True))
            self._fit_cache[key] = fit_power_law(t, ubar, factor=-0.5)
        return self._fit_cache[key]

    def integral_scale_growth_index(self, step_end: int | None = None) -> PowerLawFit:
        r"""$\lambda$, growth index of the integral scale, where $\xi \propto t^\lambda$ at late times.

        Fitted to $\xi({U}_\parallel)$ after the maximum of $\bar{U}_\parallel$,
        see :py:mod:`pttools.lattice.asymptote`.

        :param step_end: last time step of the fit, defaults to the last available step
        :return: the fit
        """
        key = ("lambda", step_end)
        if key not in self._fit_cache:
            t, xi = self.integral_scale_series(
                Quantity.X_DIV, self._fit_steps(Quantity.X_DIV, step_end, after_max=True))
            self._fit_cache[key] = fit_power_law(t, xi, factor=1.)
        return self._fit_cache[key]

    def gw_growth_index_prediction(self, step_end: int | None = None) -> tuple[float, float]:
        r"""$2\zeta + 2\lambda - 1$, predicted index of the approach of the GW power to its asymptote.

        See :py:func:`pttools.lattice.asymptote.gw_growth_index_prediction`.

        :param step_end: last time step of the fits, defaults to the last available step
        :return: the prediction and its uncertainty
        """
        return gw_growth_index_prediction(
            self.kinetic_energy_decay_index(step_end), self.integral_scale_growth_index(step_end))

    def omega_gw_asymptote(
            self,
            t_start: float | None = None,
            step_end: int | None = None,
            window: int = DEFAULT_RUNNING_MEAN_WINDOW) -> AsymptoteFit:
        r"""$\Omega_\text{gw}^\infty$, asymptotic GW energy density fraction.

        $\Omega_\text{gw}(t)$ is extrapolated to late times with a fit to
        $$\Omega_\text{gw}(t) = \Omega_\text{gw}^\infty
        \left[ 1 - \left( \frac{t}{{t}_\text{decay}} \right)^{-{d}_\text{tot}} \right],$$
        :correia_2025:`\ ` eq. 47.
        As in the article, the fit is done in the range $t \in [3 {t}_\text{max}, {t}_\text{end}]$
        to the running mean of $\Omega_\text{gw}(t)$.
        The asymptotic value is :py:attr:`pttools.math.asymptote.AsymptoteFit.value` of the returned fit.

        :param t_start: start of the fit, defaults to $3 {t}_\text{max}$.
            The fit starts at the time of the output that is nearest to it.
        :param step_end: last time step of the fit, defaults to the last available step.
            The fit is sensitive to the last points, as the running mean is mirrored at the end of the time series.
        :param window: number of points in the window of the running mean
        :return: the fit
        """
        key = ("omega_gw", t_start, step_end, window)
        if key not in self._fit_cache:
            t, omega = self.omega_gw_series(self._fit_steps(Quantity.GW_TT, step_end))
            omega = running_mean(omega, window)
            # The fit starts at the time nearest to t_start, as the times of the simulation output
            # are not exact multiples of the time step, e.g. 4319.5 instead of 3 * 1439.9.
            start = int(np.argmin(np.abs(t - (3 * self.t_max if t_start is None else t_start))))
            self._fit_cache[key] = fit_asymptote(t[start:], omega[start:], t_decay_guess=self.t_max)
        return self._fit_cache[key]

    def pow_gw_asymptote(
            self,
            y_min: float = 8.,
            y_max: float = 81.,
            start_fraction: float = 0.8,
            step_end: int | None = None) -> PowGwAsymptoteFit:
        r"""$\mathcal{P}_\text{gw}^\infty(k)$, asymptotic GW power spectrum.

        $\mathcal{P}_\text{gw}(k, t)$ of each wavenumber in the given range is extrapolated to late times
        with a fit to
        $$\mathcal{P}_\text{gw}(k, t) = \mathcal{P}_\text{gw}^\infty(k)
        \left[ 1 - \left( \frac{t}{{t}_\text{decay}(k)} \right)^{-d(k)} \right],$$
        :correia_2025:`\ ` eq. 46, see :py:func:`pttools.lattice.asymptote.fit_pow_gw_modes`.

        :param y_min: smallest $z = kR_*$ to fit
        :param y_max: largest $z = kR_*$ to fit
        :param start_fraction: fraction of the final value of each wavenumber at which its fit starts.
            The article states 0.85, whereas its scripts use 0.8 for the detonation and 0.6 for the deflagration.
        :param step_end: last time step of the fits, defaults to the last available step
        :return: the fits
        """
        key = ("pow_gw", y_min, y_max, start_fraction, step_end)
        if key not in self._fit_cache:
            steps = self._fit_steps(Quantity.GW_TT, step_end)
            k, pow_gw = self.pow_normalised(Quantity.GW_TT, steps)
            y = k * self.R_star
            modes = (y >= y_min) & (y <= y_max)
            self._fit_cache[key] = fit_pow_gw_modes(
                t=np.asarray(self.time(steps)), k=k[modes], y=y[modes], pow_gw=pow_gw[:, modes],
                start_fraction=start_fraction, t_decay_guess=self.t_max)
        return self._fit_cache[key]

    # -----
    # Comparison with the Sound Shell Model
    # -----

    def ssm_spectrum(
            self,
            model: Model | None = None,
            nuc_type: NucType = NucType.SIMULTANEOUS,
            y: th.FloatArr1D = DEFAULT_Y_LATTICE,
            **kwargs: tp.Any) -> SSMSpectrum:
        r"""Compute the Sound Shell Model spectrum for the parameters of the run, without suppression.

        :param model: the equation of state, defaults to the bag model as in the simulations
        :param nuc_type: nucleation type. The simulations of :correia_2025:`\ ` nucleate all bubbles simultaneously.
        :param y: $z = kR_*$ points of the spectrum
        :param kwargs: additional arguments for :py:class:`pttools.ssm.spectrum.SSMSpectrum`
        :return: the spectrum
        """
        if model is None:
            model = BagModel(alpha_n_min=min(0.001, self.alpha_n / 2))
        bubble = Bubble(model, v_wall=self.v_wall, alpha_n=self.alpha_n)
        return SSMSpectrum(
            bubble, y=y, nuc_type=nuc_type, suppression_method=SuppressionMethod.NONE,
            label_latex=f"SSM, {nuc_type}", label_unicode=f"SSM, {nuc_type}",
            **kwargs
        )
