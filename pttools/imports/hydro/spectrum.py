r"""Power spectra of a hydrodynamic lattice simulation at a given time."""

import functools

import numpy as np

from pttools.imports.hydro.quantities import Quantity
from pttools.imports.hydro.run import HydroRun
from pttools.ssm.spectrum_base import SpectrumBase
import pttools.type_hints as th


class HydroSpectrum(SpectrumBase):
    r"""Power spectra of a hydrodynamic lattice simulation at a given time step.

    The spectra are functions of $z = kR_*$ and normalised as in :py:class:`pttools.ssm.spectrum_base.SpectrumBase`,
    so that they can be compared directly with :py:class:`pttools.ssm.spectrum.SSMSpectrum`.
    The GW power growth rate is computed from the difference of the GW power spectra at
    ``step`` and ``step_prev``, as in :correia_2025:`\ ` fig. 5.
    """

    def __init__(self, run: HydroRun, step: int, step_prev: int | None = None):
        """Create the spectrum of a run at a given time step.

        :param run: the simulation run
        :param step: the time step
        :param step_prev: the earlier time step for computing the GW power growth rate
        """
        self.run: HydroRun = run
        self.step: int = step
        self.step_prev: int | None = step_prev
        k, _, counts = run.power_spectrum(Quantity.X_DIV, step)
        #: $k$, wavenumbers in lattice units
        self.k: th.FloatArr1D = k
        #: Number of lattice modes in each bin
        self.counts: th.FloatArr1D = counts
        self.y: th.FloatArr1D = k * run.R_star
        t_sh = f"{self.t / run.t_sh:.2f}" if run.has_globals else "?"
        self.label_latex: str = rf"{run.name}, $t={t_sh} \, t_\text{{sh}}$"
        self.label_unicode: str = f"{run.name}, t={t_sh} t_sh"

    def _pow(self, quantity: Quantity) -> th.FloatArr1D:
        k, p = self.run.pow_normalised(quantity, self.step)
        if not np.array_equal(k, self.k):
            raise ValueError(f"The wavenumbers of {quantity} differ from those of {Quantity.X_DIV}.")
        return p

    @property
    def t(self) -> float:
        """$t$, time of the spectrum."""
        return float(self.run.time(self.step))

    @property
    def t_prev(self) -> float:
        """$t$, time of the earlier time step."""
        if self.step_prev is None:
            raise ValueError("step_prev has not been given.")
        return float(self.run.time(self.step_prev))

    @functools.cached_property
    def pow_U_par(self) -> th.FloatArr1D:
        r"""$\mathcal{P}_{U_{\parallel}}(z)$, power spectrum of the compressional component of $U^i$."""
        return self._pow(Quantity.X_DIV)

    @functools.cached_property
    def pow_U_perp(self) -> th.FloatArr1D:
        r"""$\mathcal{P}_{U_{\perp}}(z)$, power spectrum of the vortical component of $U^i$.

        This is the sum of both transverse polarisations.
        """
        return self._pow(Quantity.X_ROT)

    @functools.cached_property
    def pow_v_par(self) -> th.FloatArr1D:
        r"""$\mathcal{P}_{v_{\parallel}}(z)$, power spectrum of the compressional component of the fluid 3-velocity."""
        return self._pow(Quantity.VEL_DIV)

    @functools.cached_property
    def pow_v_perp(self) -> th.FloatArr1D:
        r"""$\mathcal{P}_{v_{\perp}}(z)$, power spectrum of the vortical component of the fluid 3-velocity.

        This is the sum of both transverse polarisations.
        """
        return self._pow(Quantity.VEL_ROT)

    @functools.cached_property
    def pow_gw(self) -> th.FloatArr1D:
        r"""$\mathcal{P}_\text{gw}(z)$, GW power spectrum as a fraction of $\bar{\epsilon}_0$.

        :correia_2025:`\ ` eq. 25.
        """
        return self._pow(Quantity.GW_TT)

    @functools.cached_property
    def pow_gw_rate(self) -> th.FloatArr1D:
        r"""$\frac{1}{H_n R_*} \frac{\Delta \mathcal{P}_\text{gw}}{H_n \Delta t}$, growth rate of the GW power spectrum.

        :raises ValueError: if ``step_prev`` has not been given
        """
        if self.step_prev is None:
            raise ValueError("step_prev is required for computing the growth rate of the GW power spectrum.")
        k, rate = self.run.pow_gw_rate(self.step, self.step_prev)
        if not np.array_equal(k, self.k):
            raise ValueError("The wavenumbers of the GW power spectrum differ from those of the velocity.")
        return rate

    @property
    def mean_adiabatic_index(self) -> float:
        r"""$\Gamma = \bar{w} / \bar{\epsilon}_0$, mean adiabatic index.

        See :py:meth:`pttools.imports.hydro.run.HydroRun.mean_adiabatic_index`.
        """
        return float(self.run.mean_adiabatic_index(self.step))
