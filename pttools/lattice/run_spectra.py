r"""Power spectra and time series of a run of a hydrodynamic lattice simulation.

See :py:mod:`pttools.lattice.run_base` for the parameters and global quantities of the run,
and :py:mod:`pttools.lattice.run` for the extrapolation to late times and the comparison with the Sound Shell Model.
"""

import functools
import logging
import typing as tp

import numpy as np
import numpy.typing as npt

from pttools.lattice.quantities import Quantity, integral_scale, mean_square
from pttools.lattice.run_base import LatticeRunBase
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)


class LatticeRunSpectra(LatticeRunBase):
    r"""A run of a hydrodynamic lattice simulation with its power spectra and their time series.

    The power spectra of each quantity are stored as a 2D array with the shape (time steps, bins),
    and therefore the methods that take time steps accept both a single time step and an array of them.
    See :py:class:`pttools.lattice.run_base.LatticeRunBase` for the parameters.
    """

    @functools.cached_property
    def _power_cache(self) -> dict[Quantity, th.FloatArr2D]:
        """The power spectra of each quantity, read from the data of the run when first needed."""
        return {}

    # -----
    # Power spectra
    # -----

    @property
    def counts(self) -> th.FloatArr1D:
        """The numbers of the lattice modes in the bins of the power spectra, see :py:meth:`power_spectrum`."""
        return self.data.counts[self._valid_bins]

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

    @property
    def k(self) -> th.FloatArr1D:
        """$k$, wavenumbers of the bins of the power spectra in lattice units, see :py:meth:`power_spectrum`."""
        return self.data.k[self._valid_bins]

    def last_step(self, quantity: Quantity = Quantity.GW_TT) -> int:
        """The last time step of the power spectra of the quantity that is within the log of the global quantities.

        The times and the mean enthalpies of the later time steps would have to be extrapolated,
        see :py:meth:`time` and :py:meth:`enthalpy`.

        :param quantity: the quantity
        :return: the last such time step, or the last time step of the quantity if the log is not available
        """
        steps = self.steps(quantity)
        if self.has_globals:
            steps = steps[steps <= self.global_steps[-1]]
        return int(steps[-1])

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

    def steps(self, quantity: Quantity = Quantity.GW_TT) -> th.IntArr1D:
        """The time steps at which the power spectrum of the given quantity is available.

        :param quantity: the quantity
        :return: the time steps in increasing order, or an empty array if the quantity is not available
        """
        return self.data.steps.get(Quantity(quantity), np.array([], dtype=np.int_))

    def _steps_or_all(self, quantity: Quantity, steps: npt.ArrayLike | None) -> th.IntArr1D:
        """The given time steps as an array, or all the time steps of the quantity if not given."""
        return self.steps(quantity) if steps is None else np.asarray(steps, dtype=np.int_)

    # -----
    # Time series
    # -----

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

    def kinetic_energy_fraction_par(self, step: int) -> float:
        r"""$K_\parallel = \Gamma \bar{U}_\parallel^2$, compressional kinetic energy fraction.

        :correia_2025:`\ ` eq. 41.
        """
        return float(self.mean_adiabatic_index(step) * self.rms(Quantity.X_DIV, step)**2)

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

    def omega_gw_series(self, steps: npt.ArrayLike | None = None) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        r"""$\Omega_\text{gw}(t)$, GW energy density fraction as a function of time.

        :param steps: the time steps, defaults to all the available steps
        :return: the times and $\Omega_\text{gw}$
        """
        steps_arr = self._steps_or_all(Quantity.GW_TT, steps)
        return np.asarray(self.time(steps_arr)), np.asarray(self.omega_gw(steps_arr))

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

    @property
    def t_sh(self) -> float:
        r"""${t}_\text{sh} = R_* / \bar{U}_{\parallel,*}$, shock formation time, :correia_2025:`\ ` eq. 28."""
        return self.R_star / self.ubar_par_max[1]

    @functools.cached_property
    def ubar_par_max(self) -> tuple[int, float]:
        r"""The time step and the value of the maximum of $\bar{U}_\parallel$, :correia_2025:`\ ` table IV."""
        steps = self.steps(Quantity.X_DIV)
        ubar = np.asarray(self.rms(Quantity.X_DIV))
        i_max = int(np.argmax(ubar))
        return int(steps[i_max]), float(ubar[i_max])
