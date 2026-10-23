r"""Power spectra of a hydrodynamic lattice simulation at a given time."""

from __future__ import annotations

import functools
import typing as tp

import numpy as np

from pttools.bubble.const import DEFAULT_NU_GDH2024
from pttools.lattice.quantities import Quantity
from pttools.omgw0 import const
from pttools.omgw0.base import Omgw0Base
from pttools.ssm.barotropic import expansion_factor
from pttools.ssm.const import DEFAULT_A_STAR_A_R_RATIO
from pttools.ssm.suppression.fit import lattice_artifact_start
import pttools.type_hints as th
from pttools.utils.decorators import cached_property

if tp.TYPE_CHECKING:
    from pttools.lattice.run import LatticeRun


class LatticeSpectrum(Omgw0Base):
    r"""Power spectra of a hydrodynamic lattice simulation at a given time step.

    The spectra are functions of $z = kR_*$ and normalised as in :py:class:`pttools.ssm.spectrum_base.SpectrumBase`,
    so that they can be compared directly with :py:class:`pttools.ssm.spectrum.SSMSpectrum`.
    The GW power growth rate is computed from the difference of the GW power spectra at
    ``step`` and ``step_prev``, as in :correia_2025:`\ ` fig. 5.
    The expansion of the universe is neglected in the simulations, and therefore $H_* = H_n$.

    The simulation is in flat space, and its Hubble-scaled mean bubble spacing ${r}_{\ast,\text{lat}}$
    (:py:attr:`pttools.lattice.run_base.LatticeRunBase.r_star`) follows from the energy density
    and the gravitational constant in lattice units.
    For comparisons with the Sound Shell Model, e.g. with :py:class:`pttools.omgw0.spectrum.Spectrum`,
    the spectrum can be scaled to another Hubble-scaled mean bubble spacing ${r}_*$.
    The mean bubble spacing $R_*$ is kept in lattice units, and the Hubble rate is therefore $H_* = {r}_* / R_*$.
    The spectra other than $\mathcal{P}_\text{gw}$ and the GW power growth rate,
    which is normalised by $H_n R_*$ and $H_n t$, do not depend on ${r}_*$.

    The GW power spectrum is scaled in the same way as that of the Sound Shell Model,
    :py:attr:`pttools.ssm.spectrum.SSMSpectrum.pow_gw_ssm`, which is related to its flat-space growth rate by
    $$\mathcal{P}_\text{gw} = r_* \mathcal{H}_* \eta_\text{v} \left( \frac{a_*}{a_r} \right)^\frac{2\nu}{1+\nu}
    \frac{1}{H_n R_*} \frac{d\mathcal{P}_\text{gw}}{d(H_n t)},$$
    see :py:attr:`H_star_eta_v` and :py:attr:`dilution_of_e`.
    The simulation is considered as a source that has lasted from the start of the simulation, where the bubbles
    are nucleated, until the time $t$ of the spectrum, i.e. for $\Delta \tau_\text{v} = t / R_*$
    (:py:attr:`delta_tau_v`), with the mean growth rate
    $$\frac{1}{H_n R_*} \frac{d\mathcal{P}_\text{gw}}{d(H_n t)}
    \approx \frac{\mathcal{P}_{\text{gw},\text{lat}}}{{r}_{\ast,\text{lat}}^2 \Delta \tau_\text{v}},$$
    where $\mathcal{P}_{\text{gw},\text{lat}}$ is the GW power spectrum of the simulation in flat space.
    As $\mathcal{H}_* \eta_\text{v} = r_* \Delta \tau_\text{v} \, \eta_\text{v} / \Delta \eta_\text{v}$,
    this gives
    $$\mathcal{P}_\text{gw} = \left( \frac{{r}_*}{{r}_{\ast,\text{lat}}} \right)^2
    \frac{\eta_\text{v}}{\Delta \eta_\text{v}} \left( \frac{a_*}{a_r} \right)^\frac{2\nu}{1+\nu}
    \mathcal{P}_{\text{gw},\text{lat}}.$$
    The first factor follows from the GW power spectrum being proportional to $(H_* R_*)^2$
    in flat space, as the source lasts for a time proportional to $R_*$,
    and therefore :correia_2025:`\ ` give it as $\mathcal{P}_\text{gw} / (H_n R_*)^2$,
    see :py:attr:`pow_gw_r_star_scale` and :py:attr:`pow_gw_flat`.
    The second factor $\eta_\text{v} / \Delta \eta_\text{v}$ (:py:attr:`expansion_factor`)
    is the reduction of the GW power by the expansion of the universe during the source,
    which approaches 1 for $r_* \Delta \tau_\text{v} \ll 1$.
    It assumes that the growth rate is constant, which is the assumption of the Sound Shell Model as well.
    The last factor is the dilution of the energy density before the radiation era, which is 1 for $\nu = 0$.
    As $\mathcal{P}_\text{gw}$ then depends on ${r}_*$ also through the expansion factor,
    the GW power spectrum of an unscaled spectrum, i.e. with ${r}_* = {r}_{\ast,\text{lat}}$,
    is not that of the simulation, but :py:attr:`pow_gw_flat` is.
    """

    def __init__(
            self,
            run: LatticeRun,
            step: int,
            step_prev: int | None = None,
            r_star: float | None = None,
            T_star: float = const.DEFAULT_T_STAR,
            g_star: float = const.DEFAULT_G_STAR,
            gs_star: float | None = None,
            a_star_a_r_ratio: float = DEFAULT_A_STAR_A_R_RATIO,
            nu_gdh2024: float = DEFAULT_NU_GDH2024):
        r"""Create the spectrum of a run at a given time step.

        :param run: the simulation run
        :param step: the time step
        :param step_prev: the earlier time step for computing the GW power growth rate
        :param r_star: $r_*$, Hubble-scaled mean bubble spacing to which the spectrum is scaled.
            Defaults to that of the simulation (:py:attr:`pttools.lattice.run_base.LatticeRunBase.r_star`),
            or nan if it's not available, as it requires the initial energy density.
        :param T_star: $T_*$, temperature at the time of GW production in GeV,
            for the conversion to the GW power spectrum today
        :param g_star: $g_*$, degrees of freedom for pressure at the time of GW production
        :param gs_star: $g_{s,*}$, degrees of freedom for entropy at the time of GW production.
            If not given, $g_{s,*} = g_*$ is assumed.
        :param a_star_a_r_ratio: $\frac{a_*}{a_r}$, ratio of the scale factors at the time of GW production
            and at the start of the radiation era, see :py:attr:`dilution_of_e`
        :param nu_gdh2024: $\nu_\text{gdh2024}$, equation of state parameter of the expansion of the universe.
            The default of 0 corresponds to radiation domination, as for the bag model of the simulations.
        :raises ValueError: if $r_*$ is given and is not positive
        """
        if r_star is not None and not r_star > 0:
            raise ValueError(f"r_star must be positive. Got r_star={r_star}.")
        self.run: LatticeRun = run
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
        self.a_star_a_r_ratio: float = a_star_a_r_ratio
        self._nu_gdh2024: float = nu_gdh2024

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
    @tp.override
    def delta_tau_v(self) -> float:
        r"""$\Delta \tau_\text{v} = t / R_*$, source duration in units of the mean bubble spacing.

        This is the time from the start of the simulation, when the bubbles are nucleated, to the time step.
        """
        return self.t / self.R_star

    @property
    @tp.override
    def nu_gdh2024(self) -> float:
        return self._nu_gdh2024

    @property
    def delta_steps(self) -> int:
        """The difference of the time steps of the GW power growth rate.

        :raises ValueError: if ``step_prev`` has not been given
        """
        if self.step_prev is None:
            raise ValueError("step_prev has not been given.")
        return self.step - self.step_prev

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
        return self.pow_gw_scale * self._pow(Quantity.GW_TT)

    @functools.cached_property
    def pow_gw_flat(self) -> th.FloatArr1D:
        r"""$\mathcal{P}_\text{gw}(z)$, GW power spectrum in flat space, scaled to $r_*$.

        $$\mathcal{P}_\text{gw,flat} = \left( \frac{{r}_*}{{r}_{\ast,\text{lat}}} \right)^2
        \mathcal{P}_{\text{gw},\text{lat}},$$
        see :py:attr:`pow_gw_r_star_scale`.
        This is :py:attr:`pow_gw` without the effects of the expansion of the universe.
        For ${r}_* = {r}_{\ast,\text{lat}}$, it's the GW power spectrum of the simulation.
        """
        return self.pow_gw_r_star_scale * self._pow(Quantity.GW_TT)

    @property
    def pow_gw_r_star_scale(self) -> float:
        r"""Scaling factor of the flat-space GW power spectrum from the simulation to $r_*$.

        :py:attr:`pow_gw_flat` is the GW power spectrum of the simulation multiplied by this factor:
        $$\mathcal{P}_\text{gw,flat} = \left( \frac{{r}_*}{{r}_{\ast,\text{lat}}} \right)^2
        \mathcal{P}_{\text{gw},\text{lat}}.$$
        It's 1 if the spectrum has not been scaled to another $r_*$.
        """
        return (self.r_star / self.run.r_star)**2

    @property
    def pow_gw_scale(self) -> float:
        r"""Scaling factor of the GW power spectrum from the simulation to $r_*$ in the expanding universe.

        :py:attr:`pow_gw` is the GW power spectrum of the simulation multiplied by this factor:
        $$\mathcal{P}_\text{gw} = \left( \frac{{r}_*}{{r}_{\ast,\text{lat}}} \right)^2
        \frac{\eta_\text{v}}{\Delta \eta_\text{v}} \left( \frac{a_*}{a_r} \right)^\frac{2\nu}{1+\nu}
        \mathcal{P}_{\text{gw},\text{lat}},$$
        see :py:attr:`pow_gw_r_star_scale`, :py:attr:`expansion_factor` and :py:attr:`dilution_of_e`.
        """
        return self.pow_gw_r_star_scale * self.expansion_factor * self.dilution_of_e

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

    @functools.cached_property
    @tp.override
    def valid_slice(self) -> slice:
        r"""Slice of the $z$ array in which the spectrum is valid, i.e. without the lattice artifact.

        The GW power spectrum is cut at the minimum between the main peak and the lattice artifact at high $z$,
        see :py:func:`pttools.ssm.suppression.fit.lattice_artifact_start`.
        If there is no artifact, the whole spectrum is valid.
        """
        return slice(None, lattice_artifact_start(self.y, self.pow_gw))

    @property
    def mean_adiabatic_index(self) -> float:
        r"""$\Gamma = \bar{w} / \bar{\epsilon}_0$, mean adiabatic index.

        See :py:meth:`pttools.lattice.run_base.LatticeRunBase.mean_adiabatic_index`.
        """
        return float(self.run.mean_adiabatic_index(self.step))

    def omega_gw_asymptote_ratio(self, **kwargs: tp.Any) -> float:
        r"""$\Omega_\text{gw}^\infty / \Omega_\text{gw}(t)$, ratio of the asymptotic and current GW energy density.

        $\Omega_\text{gw}^\infty$ is extrapolated by :py:meth:`pttools.lattice.run.LatticeRun.omega_gw_asymptote`,
        and $\Omega_\text{gw}(t)$ is that of the run at the time step of this spectrum.
        The ratio does not depend on :py:attr:`r_star`, as both are scaled by the same factor.
        It gives how much the GW power would still grow if the simulation were continued,
        and can therefore be used to correct the GW power spectrum of a finite simulation.

        :param kwargs: arguments for :py:meth:`pttools.lattice.run.LatticeRun.omega_gw_asymptote`
        :return: the ratio
        """
        return self.run.omega_gw_asymptote(**kwargs).value / self.run.omega_gw(self.step)

    def pow_gw_asymptote(self, asymptote_duration: bool = False, **kwargs: tp.Any) -> th.FloatArr1D:
        r"""$\mathcal{P}_\text{gw}^\infty(z)$, GW power spectrum scaled to the asymptote.

        This estimates the GW power spectrum at late times by scaling the GW power spectrum of the simulation
        with :py:meth:`omega_gw_asymptote_ratio`:
        $$\mathcal{P}_\text{gw}^\infty(z) \approx \frac{\Omega_\text{gw}^\infty}{\Omega_\text{gw}(t)}
        S \mathcal{P}_{\text{gw},\text{lat}}(z, t),$$
        where $S$ is :py:meth:`pow_gw_asymptote_scale`.
        All the wavenumbers are scaled by the same factor, i.e. the shape of the spectrum is assumed not to change.
        For the extrapolation of each wavenumber separately,
        see :py:meth:`pttools.lattice.run.LatticeRun.pow_gw_asymptote`.

        By default, $S$ is :py:attr:`pow_gw_scale`, i.e. the expansion of the universe is taken into account
        with the source duration of the time step, and therefore this is :py:attr:`pow_gw` scaled by the ratio.
        This neglects the additional reduction by the expansion of the GW power that is produced after the time step.
        With ``asymptote_duration=True``, the expansion is taken into account with the duration of the asymptote,
        see :py:meth:`pow_gw_asymptote_scale`.

        :param asymptote_duration: whether to use the duration of the asymptote for the expansion factor
        :param kwargs: arguments for :py:meth:`pttools.lattice.run.LatticeRun.omega_gw_asymptote`
        :return: the scaled GW power spectrum
        """
        return self.pow_gw_asymptote_ratio(asymptote_duration=asymptote_duration, **kwargs) * self.pow_gw

    def pow_gw_asymptote_ratio(self, asymptote_duration: bool = False, **kwargs: tp.Any) -> float:
        r"""$\mathcal{P}_\text{gw}^\infty / \mathcal{P}_\text{gw}(t)$, ratio of the asymptotic and current GW power.

        $$\frac{\mathcal{P}_\text{gw}^\infty(z)}{\mathcal{P}_\text{gw}(z, t)}
        = \frac{\Omega_\text{gw}^\infty}{\Omega_\text{gw}(t)} \frac{S}{S_t},$$
        where $S$ is :py:meth:`pow_gw_asymptote_scale` and $S_t$ is :py:attr:`pow_gw_scale`.
        This is the factor by which :py:meth:`pow_gw_asymptote` is larger than :py:attr:`pow_gw`,
        and therefore also the ratio of their integrals over $\ln z$ in the expanding universe.
        By default, $S = S_t$, and this is the flat-space ratio :py:meth:`omega_gw_asymptote_ratio`.
        With ``asymptote_duration=True``, it's smaller, as the GW power produced after the time step
        is reduced more by the expansion of the universe.

        :param asymptote_duration: whether to use the duration of the asymptote for the expansion factor
        :param kwargs: arguments for :py:meth:`pttools.lattice.run.LatticeRun.omega_gw_asymptote`
        :return: the ratio
        """
        ratio = self.omega_gw_asymptote_ratio(**kwargs)
        if not asymptote_duration:
            return ratio
        return ratio * self.pow_gw_asymptote_scale(asymptote_duration=True, **kwargs) / self.pow_gw_scale

    def pow_gw_asymptote_scale(self, asymptote_duration: bool = False, **kwargs: tp.Any) -> float:
        r"""Scaling factor of the asymptotic GW power spectrum from the simulation to $r_*$.

        This is the factor $S$ by which an asymptotic GW power spectrum of the simulation in flat space,
        e.g. :py:attr:`pttools.lattice.asymptote.PowGwAsymptoteFit.pow_gw_inf`
        or $\frac{\Omega_\text{gw}^\infty}{\Omega_\text{gw}(t)} \mathcal{P}_{\text{gw},\text{lat}}(z, t)$,
        is multiplied to scale it to $r_*$ in the expanding universe, see :py:meth:`pow_gw_asymptote`.

        By default, this is :py:attr:`pow_gw_scale`,
        i.e. the expansion factor is that of the source duration $\Delta \tau_\text{v} = t / R_*$ of the time step.
        With ``asymptote_duration=True``, the asymptote is considered as a source with the same mean growth rate
        as up to the time step (see :py:class:`LatticeSpectrum`), but with the duration
        $$\Delta \tau_\text{v}^\infty = \frac{\Omega_\text{gw}^\infty}{\Omega_\text{gw}(t)}
        \Delta \tau_\text{v},$$
        for which this source produces $\Omega_\text{gw}^\infty$ in flat space.
        The expansion factor $\eta_\text{v} / \Delta \eta_\text{v}$
        (:py:func:`pttools.ssm.barotropic.expansion_factor`) is then evaluated with $\Delta \tau_\text{v}^\infty$.
        This assumes that the remaining GW power is produced at the mean growth rate right after the time step.
        Neither option takes into account the actual time dependence of the growth rate.

        :param asymptote_duration: whether to use the duration of the asymptote for the expansion factor
        :param kwargs: arguments for :py:meth:`pttools.lattice.run.LatticeRun.omega_gw_asymptote`,
            used only if ``asymptote_duration=True``
        :return: the scaling factor
        """
        if not asymptote_duration:
            return self.pow_gw_scale
        eta_ratio = self.omega_gw_asymptote_ratio(**kwargs) * self.eta_ratio
        return self.pow_gw_r_star_scale * expansion_factor(eta_ratio=eta_ratio, nu=self.nu_gdh2024) \
            * self.dilution_of_e
