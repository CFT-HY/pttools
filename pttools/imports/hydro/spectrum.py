r"""Power spectra of a hydrodynamic lattice simulation at a given time."""

import functools
import typing as tp

import numpy as np

from pttools.imports.hydro.quantities import Quantity
from pttools.imports.hydro.run import HydroRun
from pttools.omgw0 import const
from pttools.omgw0.base import Omgw0Base
import pttools.type_hints as th
from pttools.utils.decorators import cached_property


class HydroSpectrum(Omgw0Base):
    r"""Power spectra of a hydrodynamic lattice simulation at a given time step.

    The spectra are functions of $z = kR_*$ and normalised as in :py:class:`pttools.ssm.spectrum_base.SpectrumBase`,
    so that they can be compared directly with :py:class:`pttools.ssm.spectrum.SSMSpectrum`.
    The GW power growth rate is computed from the difference of the GW power spectra at
    ``step`` and ``step_prev``, as in :correia_2025:`\ ` fig. 5.
    The expansion of the universe is neglected in the simulations, and therefore $H_* = H_n$.

    The simulation is in flat space, and its Hubble-scaled mean bubble spacing ${r}_{*,\text{lat}}$
    (:py:attr:`HydroRun.r_star`) follows from the energy density and the gravitational constant in lattice units.
    For comparisons with the Sound Shell Model, e.g. with :py:class:`pttools.omgw0.spectrum.Spectrum`,
    the spectrum can be scaled to another Hubble-scaled mean bubble spacing ${r}_*$.
    The GW power spectrum is proportional to $(H_* R_*)^2$, as the source lasts for a time proportional to $R_*$,
    and therefore :correia_2025:`\ ` give it as $\mathcal{P}_\text{gw} / (H_n R_*)^2$.
    Accordingly, the GW power spectrum is scaled as
    $$\mathcal{P}_\text{gw} = \left( \frac{{r}_*}{{r}_{*,\text{lat}}} \right)^2 \mathcal{P}_{\text{gw},\text{lat}}.$$
    The mean bubble spacing $R_*$ is kept in lattice units, and the Hubble rate is therefore $H_* = {r}_* / R_*$.
    The other spectra and the GW power growth rate, which is normalised by $H_n R_*$ and $H_n t$,
    do not depend on ${r}_*$.
    """

    def __init__(
            self,
            run: HydroRun,
            step: int,
            step_prev: int | None = None,
            r_star: float | None = None,
            T_star: float = const.DEFAULT_T_STAR,
            g_star: float = const.DEFAULT_G_STAR,
            gs_star: float | None = None):
        r"""Create the spectrum of a run at a given time step.

        :param run: the simulation run
        :param step: the time step
        :param step_prev: the earlier time step for computing the GW power growth rate
        :param r_star: $r_*$, Hubble-scaled mean bubble spacing to which the spectrum is scaled.
            Defaults to that of the simulation (:py:attr:`HydroRun.r_star`),
            or nan if it's not available, as it requires the initial energy density.
        :param T_star: $T_*$, temperature at the time of GW production in GeV,
            for the conversion to the GW power spectrum today
        :param g_star: $g_*$, degrees of freedom for pressure at the time of GW production
        :param gs_star: $g_{s,*}$, degrees of freedom for entropy at the time of GW production.
            If not given, $g_{s,*} = g_*$ is assumed.
        :raises ValueError: if $r_*$ is given and is not positive
        """
        if r_star is not None and not r_star > 0:
            raise ValueError(f"r_star must be positive. Got r_star={r_star}.")
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
        if r_star is None:
            try:
                r_star = run.r_star
            except FileNotFoundError:
                r_star = np.nan
        self.r_star: float = r_star
        self.T_star: float = T_star
        self.g_star: float = g_star
        self.gs_star: float = g_star if gs_star is None else gs_star

    def _pow(self, quantity: Quantity) -> th.FloatArr1D:
        k, p = self.run.pow_normalised(quantity, self.step)
        if not np.array_equal(k, self.k):
            raise ValueError(f"The wavenumbers of {quantity} differ from those of {Quantity.X_DIV}.")
        return p

    @property
    @tp.override
    def H_star(self) -> float:
        return self.r_star / self.R_star

    @property
    @tp.override
    def R_star(self) -> float:
        return self.run.R_star

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

    @cached_property
    @tp.override
    def pow_gw(self) -> th.FloatArr1D:
        return (self.r_star / self.run.r_star)**2 * self._pow(Quantity.GW_TT)

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
