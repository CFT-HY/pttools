r"""Comparison of the Sound Shell Model with hydrodynamic lattice simulations.

The simulation data is imported with :py:mod:`pttools.lattice`,
and the suppression factor is fitted with :py:mod:`pttools.ssm.suppression.fit`.
"""

from dataclasses import dataclass
import logging
import typing as tp

import numpy as np

from pttools.bubble import Bubble
from pttools.lattice import LatticeRun, LatticeSpectrum, Quantity
from pttools.models import BagModel, Model
from pttools.ssm.nucleation import NucType
from pttools.ssm.spectrum import SSMSpectrum
from pttools.ssm.suppression import SuppressionFit, SuppressionFitMethod, SuppressionMethod, fit_suppression_gw
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)

#: Default $z = kR_*$ points of the Sound Shell Model spectra for comparisons with simulations.
#: The range is wide enough to cover the simulation data and to converge the integrals over $z$.
DEFAULT_Y_LATTICE: th.FloatArr1D = np.logspace(-1, 4, 1000)


@dataclass(frozen=True, slots=True)
class LatticeComparison:
    """Comparison of a simulation spectrum with the corresponding Sound Shell Model spectrum."""

    #: The simulation run
    run: LatticeRun
    #: The simulation spectrum
    spectrum: LatticeSpectrum
    #: The Sound Shell Model spectrum, computed without suppression
    ssm: SSMSpectrum
    #: The fitted suppression factor of the GW power growth rate
    fit: SuppressionFit

    @property
    def suppression(self) -> float:
        r"""$\Sigma$, the fitted suppression factor."""
        return self.fit.factor


def ssm_spectrum_for_run(
        run: LatticeRun,
        model: Model | None = None,
        nuc_type: NucType = NucType.SIMULTANEOUS,
        y: th.FloatArr1D = DEFAULT_Y_LATTICE,
        **kwargs: tp.Any) -> SSMSpectrum:
    r"""Compute the Sound Shell Model spectrum for the parameters of a simulation run, without suppression.

    :param run: the simulation run
    :param model: the equation of state, defaults to the bag model as in the simulations
    :param nuc_type: nucleation type. The simulations of :correia_2025:`\ ` nucleate all bubbles simultaneously.
    :param y: $z = kR_*$ points of the spectrum
    :param kwargs: additional arguments for :py:class:`pttools.ssm.spectrum.SSMSpectrum`
    :return: the spectrum
    """
    if model is None:
        model = BagModel(alpha_n_min=min(0.001, run.alpha_n / 2))
    bubble = Bubble(model, v_wall=run.v_wall, alpha_n=run.alpha_n)
    return SSMSpectrum(
        bubble, y=y, nuc_type=nuc_type, suppression_method=SuppressionMethod.NONE,
        label_latex=f"SSM, {nuc_type}", label_unicode=f"SSM, {nuc_type}",
        **kwargs
    )


def compare_lattice_run(
        run: LatticeRun,
        step: int,
        step_prev: int,
        ssm: SSMSpectrum | None = None,
        nuc_type: NucType = NucType.SIMULTANEOUS,
        method: SuppressionFitMethod = SuppressionFitMethod.DEFAULT,
        y_min: float | None = None,
        y_max: float | None = None,
        r_star: float | None = None,
        **kwargs: tp.Any) -> LatticeComparison:
    r"""Compare a simulation run with the Sound Shell Model and fit the suppression factor.

    The suppression factor is fitted to the GW power growth rate between the two time steps,
    excluding the lattice artifact at high $k$, see :py:func:`pttools.ssm.suppression.fit.fit_suppression`.

    :param run: the simulation run
    :param step: the time step
    :param step_prev: the earlier time step for the GW power growth rate
    :param ssm: the Sound Shell Model spectrum, computed with :py:func:`ssm_spectrum_for_run` if not given
    :param nuc_type: nucleation type for :py:func:`ssm_spectrum_for_run`
    :param method: fitting method
    :param y_min: smallest $z$ to include in the fit
    :param y_max: largest $z$ to include in the fit
    :param r_star: $r_*$, Hubble-scaled mean bubble spacing of both spectra.
        The compared velocity spectra and GW power growth rates do not depend on it,
        but the GW power spectra and their conversion to $\Omega_{\text{gw},0}$ do.
        If not given, the simulation spectrum has the $r_*$ of the simulation in lattice units,
        and the SSM spectrum the default of :py:class:`pttools.ssm.spectrum.SSMSpectrum`.
    :param kwargs: additional arguments for :py:func:`pttools.ssm.suppression.fit.fit_suppression`
    :return: the comparison
    """
    if ssm is None:
        ssm = ssm_spectrum_for_run(run, nuc_type=nuc_type) if r_star is None \
            else ssm_spectrum_for_run(run, nuc_type=nuc_type, r_star=r_star)
    if ssm.suppression_method != SuppressionMethod.NONE:
        logger.warning("The SSM spectrum includes a suppression factor, which will affect the fit.")
    spectrum = run.spectrum(step, step_prev, r_star=r_star)
    fit = fit_suppression_gw(spectrum, ssm, method=method, y_min=y_min, y_max=y_max, **kwargs)
    return LatticeComparison(run=run, spectrum=spectrum, ssm=ssm, fit=fit)


def suppression_series(
        run: LatticeRun,
        ssm: SSMSpectrum,
        delta_steps: int,
        steps: tp.Iterable[int] | None = None,
        **kwargs: tp.Any) -> tuple[th.FloatArr1D, th.FloatArr1D]:
    r"""Fit the suppression factor as a function of time.

    The simulated flow decays, and therefore the suppression factor depends on the time at which
    the GW power growth rate is measured.

    :param run: the simulation run
    :param ssm: the Sound Shell Model spectrum, computed without suppression
    :param delta_steps: the difference of the time steps for the GW power growth rate
    :param steps: the later time steps, defaults to all the available steps for which
        the earlier step is also available
    :param kwargs: additional arguments for :py:func:`pttools.ssm.suppression.fit.fit_suppression`
    :return: the midpoint times of the time step pairs and the suppression factors
    """
    available = set(run.steps(Quantity.GW_TT).tolist())
    if steps is None:
        steps = [step for step in sorted(available) if step - delta_steps in available and step - delta_steps > 0]
    times = []
    factors = []
    for step in steps:
        spectrum = run.spectrum(step, step - delta_steps)
        try:
            factors.append(fit_suppression_gw(spectrum, ssm, **kwargs).factor)
        except ValueError:
            factors.append(np.nan)
        times.append(0.5 * (spectrum.t + spectrum.t_prev))
    return np.array(times), np.array(factors)


def gw_rate_series(
        run: LatticeRun,
        delta_steps: int,
        steps: tp.Iterable[int] | None = None) -> tuple[th.FloatArr1D, th.FloatArr1D]:
    r"""Growth rate of the GW energy density fraction as a function of time.

    $$\frac{1}{H_n R_*} \frac{\Delta \Omega_\text{gw}}{H_n \Delta t}$$

    :param run: the simulation run
    :param delta_steps: the difference of the time steps of the finite difference
    :param steps: the later time steps, defaults to all the available steps for which
        the earlier step is also available
    :return: the midpoint times of the time step pairs and the growth rates
    """
    available = set(run.steps(Quantity.GW_TT).tolist())
    if steps is None:
        steps = [step for step in sorted(available) if step - delta_steps in available and step - delta_steps >= 0]
    steps_arr = np.asarray(list(steps), dtype=np.int_)
    t1 = np.asarray(run.time(steps_arr))
    t0 = np.asarray(run.time(steps_arr - delta_steps))
    omega1 = np.array([run.omega_gw(int(step)) for step in steps_arr])
    omega0 = np.array([run.omega_gw(int(step) - delta_steps) for step in steps_arr])
    return 0.5 * (t0 + t1), (omega1 - omega0) / (run.H_star * (t1 - t0)) / run.r_star


def scalar_comparison(comparison: LatticeComparison) -> dict[str, tuple[float, float]]:
    r"""Compare the scalar quantities that can be computed both from the simulation and from PTtools.

    The simulation values are evaluated at the time step of the comparison, except for
    $\bar{U}_{\parallel,*}$, $K_{\parallel,*}$ and ${t}_\text{sh}$,
    which are evaluated at the maximum of $\bar{U}_\parallel$, as in :correia_2025:`\ ` table IV.
    For PTtools, $\bar{U}_\parallel$ and $\xi$ are computed from the spectrum,
    and $\Gamma$ and $K$ from the single-bubble fluid profile.

    :param comparison: the comparison
    :return: dict of the quantity names and the (simulation, PTtools) values
    """
    run = comparison.run
    spectrum = comparison.spectrum
    ssm = comparison.ssm
    bubble = ssm.bubble
    step_max, ubar_max = run.ubar_par_max
    ubar_ssm = ssm.ubar_par
    return {
        "Ubar_par (t)": (spectrum.ubar_par, ubar_ssm),
        "Ubar_par,* (max)": (ubar_max, ubar_ssm),
        "Ubar_f (single bubble)": (ubar_max, bubble.ubarf),
        "Gamma": (spectrum.mean_adiabatic_index, bubble.mean_adiabatic_index),
        "K_par,* (max)": (run.kinetic_energy_fraction_par(step_max), bubble.mean_adiabatic_index * ubar_ssm**2),
        "K (single bubble)": (run.kinetic_energy_fraction_par(step_max), bubble.kinetic_energy_fraction),
        "xi_par / R_* (t)": (spectrum.integral_scale_par, ssm.integral_scale_par),
        "t_sh / R_*": (run.t_sh / run.R_star, 1 / ubar_ssm),
        "dOmega_gw/d(Ht) / (HR_*)": (spectrum.omega_gw_rate, ssm.omega_gw_rate),
    }


def gw_asymptote_scale(run: LatticeRun) -> float:
    r"""Scale of the GW power in the plots of its extrapolation, :correia_2025:`\ ` fig. 6.

    $$(\Gamma \bar{U}_\parallel^2)_*^2 (H_n R_*) (H_n {t}_\text{sh}),$$
    where $(\Gamma \bar{U}_\parallel^2)_*$ is :py:meth:`pttools.lattice.run.LatticeRun.kinetic_energy_fraction_par`
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

    :param run: the simulation run
    :return: the scale
    """
    return run.kinetic_energy_fraction_par(run.ubar_par_max[0])**2 * run.r_star * run.H_star * run.t_sh
