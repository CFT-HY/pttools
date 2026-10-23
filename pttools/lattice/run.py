r"""A run of a hydrodynamic lattice simulation.

:py:class:`LatticeRun` is split into classes, each of which extends the previous one:

- :py:class:`pttools.lattice.run_base.LatticeRunBase`: the parameters of the run
  and the global quantities of the ``.out`` file, see :py:mod:`pttools.lattice.run_base` for the data and units
- :py:class:`pttools.lattice.run_spectra.LatticeRunSpectra`: the power spectra and their time series
- :py:class:`LatticeRun`: the spectra at a given time step, the extrapolation to late times
  and the comparison with the Sound Shell Model
"""

import functools
import logging
import typing as tp

import numpy as np

from pttools.bubble import Bubble
from pttools.lattice.asymptote import (
    DEFAULT_RUNNING_MEAN_WINDOW,
    PowGwAsymptoteFit,
    fit_pow_gw_modes,
    gw_growth_index_prediction,
)
from pttools.lattice.quantities import Quantity
from pttools.lattice.run_spectra import LatticeRunSpectra
from pttools.lattice.spectrum import LatticeSpectrum
from pttools.math.asymptote import AsymptoteFit, fit_asymptote
from pttools.math.numerics import running_mean
from pttools.math.power_law import PowerLawFit, fit_power_law
from pttools.models import BagModel, Model
from pttools.ssm.spectrum import SSMSpectrum
from pttools.ssm.suppression import SuppressionMethod
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)

#: Default $z = kR_*$ points of the Sound Shell Model spectra for comparisons with simulations.
#: The range is wide enough to cover the simulation data and to converge the integrals over $z$.
DEFAULT_Y_LATTICE: th.FloatArr1D = np.logspace(-1, 4, 1000)


class LatticeRun(LatticeRunSpectra):
    r"""A run of a hydrodynamic lattice simulation.

    In addition to the parameters, the global quantities (:py:class:`pttools.lattice.run_base.LatticeRunBase`)
    and the power spectra and their time series (:py:class:`pttools.lattice.run_spectra.LatticeRunSpectra`),
    this provides the spectra at a given time step, the extrapolation to late times
    and the comparison with the Sound Shell Model.
    See :py:class:`pttools.lattice.run_base.LatticeRunBase` for the parameters.
    """

    @functools.cached_property
    def _fit_cache(self) -> dict[tuple[tp.Any, ...], tp.Any]:
        """The fits of the extrapolation to late times by their arguments."""
        return {}

    # -----
    # Spectra
    # -----

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
    # Extrapolation to late times
    # -----

    def _fit_steps(self, quantity: Quantity, step_end: int | None, after_max: bool = False) -> th.IntArr1D:
        r"""The time steps of the fits, optionally only those after the maximum of $\bar{U}_\parallel$."""
        steps = self.steps(quantity)
        if step_end is not None:
            steps = steps[steps <= step_end]
        if after_max:
            steps = steps[steps >= self.ubar_par_max[0]]
        return steps

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

    def gw_growth_index_prediction(self, step_end: int | None = None) -> tuple[float, float]:
        r"""$2\zeta + 2\lambda - 1$, predicted index of the approach of the GW power to its asymptote.

        See :py:func:`pttools.lattice.asymptote.gw_growth_index_prediction`.

        :param step_end: last time step of the fits, defaults to the last available step
        :return: the prediction and its uncertainty
        """
        return gw_growth_index_prediction(
            self.kinetic_energy_decay_index(step_end), self.integral_scale_growth_index(step_end))

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

    @property
    def t_max(self) -> float:
        r"""${t}_\text{max}$, time of the maximum of $\bar{U}_\parallel$, see :py:attr:`ubar_par_max`."""
        return float(self.time(self.ubar_par_max[0]))

    # -----
    # Comparison with the Sound Shell Model
    # -----

    def ssm_spectrum(
            self,
            model: Model | None = None,
            y: th.FloatArr1D = DEFAULT_Y_LATTICE,
            **kwargs: tp.Any) -> SSMSpectrum:
        r"""Compute the Sound Shell Model spectrum for the parameters of the run, without suppression.

        The nucleation type is that of the run, :py:attr:`nuc_type`.

        :param model: the equation of state, defaults to the bag model as in the simulations
        :param y: $z = kR_*$ points of the spectrum
        :param kwargs: additional arguments for :py:class:`pttools.ssm.spectrum.SSMSpectrum`
        :return: the spectrum
        """
        if model is None:
            model = BagModel(alpha_n_min=min(0.001, self.alpha_n / 2))
        bubble = Bubble(model, v_wall=self.v_wall, alpha_n=self.alpha_n)
        return SSMSpectrum(
            bubble, y=y, nuc_type=self.nuc_type, suppression_method=SuppressionMethod.NONE,
            label_latex=f"SSM, {self.nuc_type}", label_unicode=f"SSM, {self.nuc_type}",
            **kwargs
        )
