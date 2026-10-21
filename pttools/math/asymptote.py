r"""Fits of power law approaches to asymptotes to time series.

The approach to an asymptote is fitted with
$$X(t) = X^\infty \left[ 1 - \left( \frac{t}{{t}_\text{decay}} \right)^{-d} \right],$$
as in :correia_2025:`\ ` eqs. 46 and 47.
For the applications to the GW power of lattice simulations, see :py:mod:`pttools.lattice.asymptote`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from pttools.speedup import njit
import pttools.type_hints as th


@dataclass(frozen=True, slots=True)
class AsymptoteFit:
    r"""Fit of :func:`asymptotic_growth` to a time series.

    The uncertainties are the square roots of the diagonal of the covariance matrix of the fit.
    """

    #: $X^\infty$, asymptotic value
    value: float
    #: Uncertainty of :py:attr:`value`
    value_err: float
    #: ${t}_\text{decay}$, time scale of the approach
    t_decay: float
    #: Uncertainty of :py:attr:`t_decay`
    t_decay_err: float
    #: $d$, power law index of the approach
    index: float
    #: Uncertainty of :py:attr:`index`
    index_err: float
    #: Start of the time range of the fit
    t_start: float
    #: End of the time range of the fit
    t_end: float

    def __call__[T: th.FloatOrArr](self, t: T) -> T:
        """Evaluate the fitted function at the given times."""
        return asymptotic_growth(t, value_inf=self.value, t_decay=self.t_decay, index=self.index)


@njit(cache=True, nogil=True)
def asymptotic_growth[T: th.FloatOrArr](t: T, value_inf: float, t_decay: float, index: float) -> T:
    r"""Power law approach to an asymptote.

    $$X(t) = X^\infty \left[ 1 - \left( \frac{t}{{t}_\text{decay}} \right)^{-d} \right]$$
    :correia_2025:`\ ` eqs. 46 and 47.

    :param t: $t$, time
    :param value_inf: $X^\infty$, asymptotic value
    :param t_decay: ${t}_\text{decay}$, time scale of the approach
    :param index: $d$, power law index of the approach
    :return: $X(t)$
    """
    return value_inf * (1 - (t / t_decay)**(-index))  # pyrefly: ignore[bad-return]


@njit(cache=True, nogil=True)
def _asymptotic_growth_fit(t: th.FloatArr1D, index: float, t_decay: float, value_inf: float) -> th.FloatArr1D:
    """:func:`asymptotic_growth` with the parameters in the order of the fit."""
    return asymptotic_growth(t, value_inf=value_inf, t_decay=t_decay, index=index)


@njit(cache=True, nogil=True)
def _asymptotic_growth_jac(t: th.FloatArr1D, index: float, t_decay: float, value_inf: float) -> th.FloatArr2D:
    """Jacobian of :func:`_asymptotic_growth_fit` with respect to its parameters."""
    power = (t / t_decay)**(-index)
    return np.column_stack((
        value_inf * np.log(t / t_decay) * power,
        -index * value_inf * power / t_decay,
        1 - power,
    ))


def fit_asymptote(
        t: th.FloatArr1D,
        values: th.FloatArr1D,
        index_guess: float = 1.,
        t_decay_guess: float | None = None,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None) -> AsymptoteFit:
    r"""Fit :func:`asymptotic_growth` to a time series.

    The values are scaled by their last value for the fit, so that the fit is independent of their magnitude.

    :param t: $t$, times of the time series
    :param values: $X(t)$, values of the time series
    :param index_guess: initial guess of the index $d$
    :param t_decay_guess: initial guess of ${t}_\text{decay}$, defaults to the first time of the time series
    :param bounds: bounds of the parameters $(d, {t}_\text{decay}, X^\infty / X({t}_\text{end}))$ as (lower, upper).
        If given, the fit is done with the trust region reflective algorithm instead of Levenberg-Marquardt.
    :return: the fit
    :raises RuntimeError: if the fit does not converge
    :raises ValueError: if the time series has fewer points than there are parameters
    """
    scale = float(values[-1])
    p0 = (index_guess, float(t[0]) if t_decay_guess is None else t_decay_guess, 1.)
    if t.size < len(p0):
        raise ValueError(f"The fit requires at least {len(p0)} points, but the time series has {t.size}.")
    # The trial parameters of the iterations may be outside the domain of the function, e.g. a negative t_decay,
    # for which the power is undefined. The fit handles the resulting nan values.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if bounds is None:
            popt, pcov = curve_fit(_asymptotic_growth_fit, t, values / scale, p0=p0, jac=_asymptotic_growth_jac)
        else:
            popt, pcov = curve_fit(_asymptotic_growth_fit, t, values / scale, p0=p0, bounds=bounds)
    err = np.sqrt(np.diag(pcov))
    return AsymptoteFit(
        value=float(popt[2] * scale), value_err=float(err[2] * abs(scale)),
        t_decay=float(popt[1]), t_decay_err=float(err[1]),
        index=float(popt[0]), index_err=float(err[0]),
        t_start=float(t[0]), t_end=float(t[-1]),
    )


def fit_start_index(values: th.FloatArr1D, fraction: float) -> int:
    """Index of the start of the interval at the end of the time series where the values exceed a fraction of the last.

    :param values: the time series
    :param fraction: the fraction of the last value
    :return: the index after the last value below the threshold, or 0 if all the values exceed it
    """
    below = np.flatnonzero(values < fraction * values[-1])
    return int(below[-1]) + 1 if below.size else 0
