r"""Analysis of the suppression factor of the Sound Shell Model with hydrodynamic lattice simulations.

For the comparison of a simulation with the Sound Shell Model,
see :py:class:`pttools.lattice.comparison.LatticeComparison`.
"""

from __future__ import annotations

import typing as tp

import numpy as np

from pttools.lattice import LatticeRun, Quantity
from pttools.ssm.suppression import fit_suppression_gw
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from pttools.ssm.spectrum import SSMSpectrum


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
