r"""Comparison of the Sound Shell Model with hydrodynamic lattice simulations.

The simulation data is imported with :py:mod:`pttools.lattice`,
and the suppression factor is fitted with :py:mod:`pttools.ssm.suppression.fit`.
"""

import logging
import typing as tp

import numpy as np

from pttools.lattice import LatticeRun, LatticeSpectrum, Quantity
from pttools.ssm.nucleation import NucType
from pttools.ssm.spectrum import SSMSpectrum
from pttools.ssm.suppression import SuppressionFit, SuppressionFitMethod, SuppressionMethod, fit_suppression_gw
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)


class LatticeComparison:
    r"""Comparison of a simulation spectrum with the corresponding Sound Shell Model spectrum.

    The suppression factor is fitted to the GW power growth rate between the two time steps,
    excluding the lattice artifact at high $k$, see :py:func:`pttools.ssm.suppression.fit.fit_suppression`.
    """

    def __init__(
            self,
            run: LatticeRun,
            step: int,
            step_prev: int,
            ssm: SSMSpectrum | None = None,
            nuc_type: NucType = NucType.SIMULTANEOUS,
            method: SuppressionFitMethod = SuppressionFitMethod.DEFAULT,
            y_min: float | None = None,
            y_max: float | None = None,
            r_star: float | None = None,
            **kwargs: tp.Any):
        r"""Compare a simulation run with the Sound Shell Model and fit the suppression factor.

        :param run: the simulation run
        :param step: the time step
        :param step_prev: the earlier time step for the GW power growth rate
        :param ssm: the Sound Shell Model spectrum, computed without suppression.
            If not given, it's computed with :py:meth:`pttools.lattice.run.LatticeRun.ssm_spectrum`.
        :param nuc_type: nucleation type for :py:meth:`pttools.lattice.run.LatticeRun.ssm_spectrum`
        :param method: fitting method
        :param y_min: smallest $z$ to include in the fit
        :param y_max: largest $z$ to include in the fit
        :param r_star: $r_*$, Hubble-scaled mean bubble spacing of both spectra.
            The compared velocity spectra and GW power growth rates do not depend on it,
            but the GW power spectra and their conversion to $\Omega_{\text{gw},0}$ do.
            If not given, the simulation spectrum has the $r_*$ of the simulation in lattice units,
            and the SSM spectrum the default of :py:class:`pttools.ssm.spectrum.SSMSpectrum`.
        :param kwargs: additional arguments for :py:func:`pttools.ssm.suppression.fit.fit_suppression`
        """
        if ssm is None:
            ssm = run.ssm_spectrum(nuc_type=nuc_type) if r_star is None \
                else run.ssm_spectrum(nuc_type=nuc_type, r_star=r_star)
        if ssm.suppression_method != SuppressionMethod.NONE:
            logger.warning("The SSM spectrum includes a suppression factor, which will affect the fit.")
        #: The simulation run
        self.run: LatticeRun = run
        #: The simulation spectrum
        self.spectrum: LatticeSpectrum = run.spectrum(step, step_prev, r_star=r_star)
        #: The Sound Shell Model spectrum, computed without suppression
        self.ssm: SSMSpectrum = ssm
        #: The fitted suppression factor of the GW power growth rate
        self.fit: SuppressionFit = fit_suppression_gw(
            self.spectrum, ssm, method=method, y_min=y_min, y_max=y_max, **kwargs)

    @property
    def suppression(self) -> float:
        r"""$\Sigma$, the fitted suppression factor."""
        return self.fit.factor

    def scalar_comparison(self) -> dict[str, tuple[float, float]]:
        r"""Compare the scalar quantities that can be computed both from the simulation and from PTtools.

        The simulation values are evaluated at the time step of the comparison, except for
        $\bar{U}_{\parallel,*}$, $K_{\parallel,*}$ and ${t}_\text{sh}$,
        which are evaluated at the maximum of $\bar{U}_\parallel$, as in :correia_2025:`\ ` table IV.
        For PTtools, $\bar{U}_\parallel$ and $\xi$ are computed from the spectrum,
        and $\Gamma$ and $K$ from the single-bubble fluid profile.

        :return: dict of the quantity names and the (simulation, PTtools) values
        """
        run = self.run
        spectrum = self.spectrum
        ssm = self.ssm
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
