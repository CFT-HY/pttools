r"""Comparison of hydrodynamic lattice simulations with the Sound Shell Model.

The suppression factor is fitted with :py:mod:`pttools.ssm.suppression.fit`.
"""

from __future__ import annotations

import logging
import typing as tp

import numpy as np
import pandas as pd

from pttools.omgw0.spectrum import Spectrum
from pttools.ssm.suppression import SuppressionFit, SuppressionFitMethod, SuppressionMethod, fit_suppression_gw
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from pttools.lattice.run import LatticeRun
    from pttools.lattice.spectrum import LatticeSpectrum
    from pttools.ssm.spectrum import SSMSpectrum

logger: logging.Logger = logging.getLogger(__name__)


class LatticeComparison:
    r"""Comparison of a simulation spectrum with the corresponding Sound Shell Model spectrum.

    The suppression factor is fitted to the GW power growth rate between the two time steps,
    excluding the lattice artifact at high $k$, see :py:func:`pttools.ssm.suppression.fit.fit_suppression`.
    """

    def __init__(
            self,
            spectrum: LatticeSpectrum,
            ssm: SSMSpectrum | None = None,
            method: SuppressionFitMethod = SuppressionFitMethod.DEFAULT,
            y_min: float | None = None,
            y_max: float | None = None,
            **kwargs: tp.Any):
        r"""Compare a simulation spectrum with the Sound Shell Model and fit the suppression factor.

        :param spectrum: the simulation spectrum, which must have the earlier time step ``step_prev``
            for the GW power growth rate, see :py:meth:`pttools.lattice.run.LatticeRun.spectrum`
        :param ssm: the Sound Shell Model spectrum, computed without suppression.
            If not given, it's computed with :py:meth:`pttools.lattice.run.LatticeRun.ssm_spectrum`
            with the $r_*$ of the simulation spectrum, if the spectrum has been scaled to it.
            Otherwise, the SSM spectrum has the default $r_*$ of :py:class:`pttools.ssm.spectrum.SSMSpectrum`,
            as the $r_*$ of the simulation in lattice units is unphysically large.
            The compared velocity spectra and GW power growth rates do not depend on $r_*$,
            but the GW power spectra and their conversion to $\Omega_{\text{gw},0}$ do.
            A warning is logged if the SSM spectrum has a different nucleation type than the run
            (:py:attr:`pttools.lattice.run_base.LatticeRunBase.nuc_type`),
            or a different $r_*$ than the simulation spectrum.
            The latter is always the case if the simulation spectrum has not been scaled to a physical $r_*$.
            If it's a :py:class:`pttools.omgw0.spectrum.Spectrum`,
            a warning is also logged if its $T_*$ or $g_*$ differ from those of the simulation spectrum.
        :param method: fitting method
        :param y_min: smallest $z$ to include in the fit
        :param y_max: largest $z$ to include in the fit
        :param kwargs: additional arguments for :py:func:`pttools.ssm.suppression.fit.fit_suppression`
        :raises ValueError: if the simulation spectrum does not have ``step_prev``
        """
        if spectrum.step_prev is None:
            raise ValueError("The spectrum must have step_prev for the GW power growth rate.")
        run = spectrum.run
        if ssm is None:
            # The r_star of the simulation in lattice units is unphysically large for the SSM spectrum.
            unscaled = not np.isfinite(spectrum.r_star) or spectrum.r_star == run.r_star
            ssm = run.ssm_spectrum() if unscaled else run.ssm_spectrum(r_star=spectrum.r_star)
        self._check_ssm(spectrum, ssm)
        if ssm.suppression_method != SuppressionMethod.NONE:
            logger.warning("The SSM spectrum includes a suppression factor, which will affect the fit.")
        #: The simulation run
        self.run: LatticeRun = run
        #: The simulation spectrum
        self.spectrum: LatticeSpectrum = spectrum
        #: The Sound Shell Model spectrum, computed without suppression
        self.ssm: SSMSpectrum = ssm
        #: The fitted suppression factor of the GW power growth rate
        self.fit: SuppressionFit = fit_suppression_gw(spectrum, ssm, method=method, y_min=y_min, y_max=y_max, **kwargs)

    @staticmethod
    def _check_ssm(spectrum: LatticeSpectrum, ssm: SSMSpectrum) -> None:
        """Log warnings if the parameters of the SSM spectrum do not match those of the simulation spectrum."""
        name = spectrum.run.name
        if ssm.nuc_type != spectrum.run.nuc_type:
            logger.warning(
                "The nucleation type of the SSM spectrum (%s) differs from that of the run %s (%s).",
                ssm.nuc_type, name, spectrum.run.nuc_type)
        if not np.isclose(ssm.r_star, spectrum.r_star):
            logger.warning(
                "The r_star of the SSM spectrum (%s) differs from that of the spectrum of the run %s (%s).",
                ssm.r_star, name, spectrum.r_star)
        if isinstance(ssm, Spectrum):
            for attr in ("T_star", "g_star"):
                if not np.isclose(getattr(ssm, attr), getattr(spectrum, attr)):
                    logger.warning(
                        "The %s of the SSM spectrum (%s) differs from that of the spectrum of the run %s (%s).",
                        attr, getattr(ssm, attr), name, getattr(spectrum, attr))

    def omega(self, asymptote_duration: bool = False, **kwargs: tp.Any) -> dict[str, float]:
        r"""$\Omega_\text{gw}$, total GW power of the simulation and of the SSM.

        The values are integrated over the $z = kR_*$ points of each spectrum.
        The asymptotic value of the simulation is that of
        :py:meth:`pttools.lattice.spectrum.LatticeSpectrum.pow_gw_asymptote`.

        :param asymptote_duration: whether to take into account the expansion of the universe
            with the duration of the asymptote, see
            :py:meth:`pttools.lattice.spectrum.LatticeSpectrum.pow_gw_asymptote_scale`
        :param kwargs: arguments for :py:meth:`pttools.lattice.run.LatticeRun.omega_gw_asymptote`
        :return: $\Omega_\text{gw}(t)$ and $\Omega_\text{gw}^\infty$ of the simulation,
            and $\Omega_\text{gw}$ of the SSM with and without the fitted suppression factor
        """
        spectrum = self.spectrum
        ssm = self.ssm
        ln_y = np.log(spectrum.y)
        omega_ssm = float(np.trapezoid(ssm.pow_gw, np.log(ssm.y)))
        return {
            "Omega_gw(t)": float(np.trapezoid(spectrum.pow_gw, ln_y)),
            "Omega_gw_inf": float(np.trapezoid(
                spectrum.pow_gw_asymptote(asymptote_duration=asymptote_duration, **kwargs), ln_y)),
            "Omega_gw SSM, suppressed": self.fit.factor * omega_ssm,
            "Omega_gw SSM, unsuppressed": omega_ssm,
        }

    @property
    def suppression(self) -> float:
        r"""$\Sigma$, the fitted suppression factor."""
        return self.fit.factor

    def scalars(self) -> pd.DataFrame:
        r"""Compare the scalar quantities that can be computed both from the simulation and from PTtools.

        The simulation values are evaluated at the time step of the comparison, except for
        $\bar{U}_{\parallel,*}$, $K_{\parallel,*}$ and ${t}_\text{sh}$,
        which are evaluated at the maximum of $\bar{U}_\parallel$, as in :correia_2025:`\ ` table IV.
        For PTtools, $\bar{U}_\parallel$ and $\xi$ are computed from the spectrum,
        and $\Gamma$ and $K$ from the single-bubble fluid profile.

        :return: the table of the quantities, with the columns ``Simulation``, ``PTtools`` and their ``Ratio``.
            It can be formatted e.g. as Markdown with :py:meth:`pandas.DataFrame.to_markdown`.
        """
        run = self.run
        spectrum = self.spectrum
        ssm = self.ssm
        bubble = ssm.bubble
        step_max, ubar_max = run.ubar_par_max
        ubar_ssm = ssm.ubar_par
        values = {
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
        table = pd.DataFrame.from_dict(values, orient="index", columns=["Simulation", "PTtools"])
        table.index.name = "Quantity"
        table["Ratio"] = table["Simulation"] / table["PTtools"]
        return table
