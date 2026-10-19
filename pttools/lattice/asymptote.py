r"""Extrapolation of the time development of the quantities of lattice simulations to late times.

The GW power of the simulations of :correia_2025:`\ ` approaches an asymptote as the fluid flow decays.
It's extrapolated with fits to the power law approach
$$X(t) = X^\infty \left[ 1 - \left( \frac{t}{{t}_\text{decay}} \right)^{-d} \right],$$
:correia_2025:`\ ` eqs. 46 and 47, where $X$ is either $\mathcal{P}_\text{gw}(k)$ or $\Omega_\text{gw}$.
The decay of the kinetic energy and the growth of the integral scale are fitted with
$$\frac{\bar{U}_\parallel(t)}{\bar{U}_\parallel({t}_\text{max})}
= \left( 1 + \frac{t - {t}_\text{max}}{t_U} \right)^{-\zeta/2}, \quad
\frac{\xi(t)}{\xi({t}_\text{max})} = \left( 1 + \frac{t - {t}_\text{max}}{{t}_\xi} \right)^\lambda,$$
where ${t}_\text{max}$ is the time of the maximum of $\bar{U}_\parallel$,
so that $K^2 \propto t^{-2\zeta}$ and $\xi \propto t^\lambda$ at late times,
which predict the index of the approach of the GW power to its asymptote to be $d = 2\zeta + 2\lambda - 1$,
:correia_2025:`\ ` sec. VI.
These follow the scripts of the Zenodo release of the article.
"""

from dataclasses import dataclass
import logging

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)

#: Default window length of the running mean of $\Omega_\text{gw}(t)$ before fitting,
#: as in the scripts of :correia_2025:`\ `
DEFAULT_RUNNING_MEAN_WINDOW: int = 23


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


def _asymptotic_growth_fit(t: th.FloatArr1D, index: float, t_decay: float, value_inf: float) -> th.FloatArr1D:
    """:func:`asymptotic_growth` with the parameters in the order of the fit."""
    return asymptotic_growth(t, value_inf=value_inf, t_decay=t_decay, index=index)


def _asymptotic_growth_jac(t: th.FloatArr1D, index: float, t_decay: float, value_inf: float) -> th.FloatArr2D:
    """Jacobian of :func:`_asymptotic_growth_fit` with respect to its parameters."""
    power = (t / t_decay)**(-index)
    return np.column_stack((
        value_inf * np.log(t / t_decay) * power,
        -index * value_inf * power / t_decay,
        1 - power,
    ))


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
    """
    scale = float(values[-1])
    p0 = (index_guess, float(t[0]) if t_decay_guess is None else t_decay_guess, 1.)
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


def running_mean(values: th.FloatArr1D, window: int = DEFAULT_RUNNING_MEAN_WINDOW) -> th.FloatArr1D:
    r"""Running mean with mirrored boundaries, as in the scripts of :correia_2025:`\ `.

    :param values: the values
    :param window: number of points in the window, which must be odd
    :return: the running mean
    """
    return savgol_filter(values, window_length=window, polyorder=0, mode="mirror")


@dataclass(frozen=True, slots=True)
class PowerLawFit:
    r"""Fit of the power law $(1 + (t - t_0) / {t}_\text{scale})^{p}$ to a normalised time series.

    The uncertainties are the square roots of the diagonal of the covariance matrix of the fit.
    """

    #: The power law index, e.g. $\zeta$ or $\lambda$
    index: float
    #: Uncertainty of :py:attr:`index`
    index_err: float
    #: ${t}_\text{scale}$, time scale of the power law
    t_scale: float
    #: $t_0$, start of the fit
    t_0: float
    #: The exponent $p$ is ``factor * index``
    factor: float

    def __call__[T: th.FloatOrArr](self, t: T) -> T:
        """Evaluate the fitted function, normalised to 1 at $t_0$, at the given times."""
        return (1 + (t - self.t_0) / self.t_scale)**(self.factor * self.index)  # pyrefly: ignore[bad-return]


def fit_power_law(t: th.FloatArr1D, values: th.FloatArr1D, factor: float) -> PowerLawFit:
    r"""Fit $X(t) / X(t_0) = (1 + (t - t_0) / {t}_\text{scale})^{p}$ with $p = $ ``factor * index``.

    :param t: times, the first of which is $t_0$
    :param values: values, which are normalised by the first value
    :param factor: factor of the exponent, e.g. $-1/2$ for $\zeta$ and 1 for $\lambda$
    :return: the fit
    """
    t_0 = float(t[0])

    def func(t2: th.FloatArr1D, index: float, t_scale: float) -> th.FloatArr1D:
        return (1 + (t2 - t_0) / t_scale)**(factor * index)

    # See fit_asymptote() for the error state.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        popt, pcov = curve_fit(func, t, values / values[0], p0=(1., float(t[-1] - t_0) / 10 or 1.))
    err = np.sqrt(np.diag(pcov))
    return PowerLawFit(index=float(popt[0]), index_err=float(err[0]), t_scale=float(popt[1]), t_0=t_0, factor=factor)


def fit_start_index(values: th.FloatArr1D, fraction: float) -> int:
    """Index of the start of the interval at the end of the time series where the values exceed a fraction of the last.

    :param values: the time series
    :param fraction: the fraction of the last value
    :return: the index after the last value below the threshold, or 0 if all the values exceed it
    """
    below = np.flatnonzero(values < fraction * values[-1])
    return int(below[-1]) + 1 if below.size else 0


@dataclass(frozen=True, slots=True)
class PowGwAsymptoteFit:
    r"""Fits of :func:`asymptotic_growth` to the time series of $\mathcal{P}_\text{gw}(k, t)$ of each wavenumber."""

    #: $z = kR_*$ of the fitted wavenumbers
    y: th.FloatArr1D
    #: $k$ of the fitted wavenumbers in lattice units
    k: th.FloatArr1D
    #: $t$, the times of the time series
    t: th.FloatArr1D
    #: $\mathcal{P}_\text{gw}(k, t)$ with the shape (times, wavenumbers)
    pow_gw: th.FloatArr2D
    #: The fits of each wavenumber, or None if the fit failed
    fits: tuple[AsymptoteFit | None, ...]

    def _param(self, name: str) -> th.FloatArr1D:
        return np.array([np.nan if fit is None else getattr(fit, name) for fit in self.fits])

    @property
    def pow_gw_inf(self) -> th.FloatArr1D:
        r"""$\mathcal{P}_\text{gw}^\infty(k)$, extrapolated late-time GW power spectrum."""
        return self._param("value")

    @property
    def pow_gw_inf_err(self) -> th.FloatArr1D:
        r"""Uncertainty of $\mathcal{P}_\text{gw}^\infty(k)$."""
        return self._param("value_err")

    @property
    def index(self) -> th.FloatArr1D:
        r"""$d(k)$, power law indices of the approach to the asymptote."""
        return self._param("index")

    @property
    def index_err(self) -> th.FloatArr1D:
        r"""Uncertainties of $d(k)$."""
        return self._param("index_err")

    @property
    def t_decay(self) -> th.FloatArr1D:
        r"""${t}_\text{decay}(k)$, time scales of the approach to the asymptote."""
        return self._param("t_decay")

    def omega_gw_partial(self, t: th.FloatArr1D | None = None) -> th.FloatArr1D:
        r"""$\Omega_\text{par}(t)$, GW energy density fraction of the fitted wavenumbers.

        $$\Omega_\text{par}(t) = \int_{{k}_\text{min}}^{{k}_\text{max}} \frac{dk}{k} \mathcal{P}_\text{gw}(k, t)$$

        :param t: times at which the fits are evaluated. If not given, the data is integrated at :py:attr:`t`.
        :return: $\Omega_\text{par}(t)$
        """
        ln_k = np.log(self.k)
        if t is None:
            return np.trapezoid(self.pow_gw, ln_k, axis=1)
        fitted = np.column_stack([np.full_like(t, np.nan) if fit is None else fit(t) for fit in self.fits])
        return np.trapezoid(fitted, ln_k, axis=1)


def fit_pow_gw_modes(
        t: th.FloatArr1D,
        k: th.FloatArr1D,
        y: th.FloatArr1D,
        pow_gw: th.FloatArr2D,
        start_fraction: float,
        t_decay_guess: float,
        fallback_fraction: float = 0.7) -> PowGwAsymptoteFit:
    r"""Fit :func:`asymptotic_growth` to the time series of $\mathcal{P}_\text{gw}(k, t)$ of each wavenumber.

    The fit of each wavenumber is done over the interval at the end of the time series
    where its GW power exceeds ``start_fraction`` of its final value.
    The fitted parameters of each wavenumber are used as the initial guess for the next one.
    If a fit does not converge, it's retried over the interval where the power exceeds ``fallback_fraction``
    of its final value, with bounds on the parameters as in the scripts of :correia_2025:`\ `.

    :param t: $t$, times of the time series
    :param k: $k$, wavenumbers
    :param y: $z = kR_*$ of the wavenumbers
    :param pow_gw: $\mathcal{P}_\text{gw}(k, t)$ with the shape (times, wavenumbers)
    :param start_fraction: fraction of the final value at which the fits start
    :param t_decay_guess: initial guess of ${t}_\text{decay}$ for the first wavenumber
    :param fallback_fraction: fraction of the final value at which the fallback fits start
    :return: the fits
    """
    fits: list[AsymptoteFit | None] = []
    index_guess = 1.
    for i in range(k.size):
        series = pow_gw[:, i]
        start = fit_start_index(series, start_fraction)
        try:
            fit = fit_asymptote(t[start:], series[start:], index_guess=index_guess, t_decay_guess=t_decay_guess)
        except RuntimeError:
            logger.debug("The asymptote fit failed at kR_*=%s. Retrying with bounds.", y[i])
            start = fit_start_index(series, fallback_fraction)
            try:
                fit = fit_asymptote(
                    t[start:], series[start:], index_guess=index_guess, t_decay_guess=t_decay_guess,
                    bounds=((0., t_decay_guess, 0.9), (5., np.inf, 4.)))
            except (RuntimeError, ValueError) as e:
                logger.warning("The asymptote fit failed at kR_*=%s: %s", y[i], e)
                fit = None
        if fit is not None:
            index_guess, t_decay_guess = fit.index, fit.t_decay
        fits.append(fit)
    return PowGwAsymptoteFit(y=y, k=k, t=t, pow_gw=pow_gw, fits=tuple(fits))


def gw_growth_index_prediction(zeta: PowerLawFit, lambda_: PowerLawFit) -> tuple[float, float]:
    r"""$2\zeta + 2\lambda - 1$, predicted index of the approach of the GW power to its asymptote.

    :correia_2025:`\ ` sec. VI.
    The uncertainty is the sum of the uncertainties of the terms, as in the scripts of the article.

    :param zeta: fit of the decay index $\zeta$ of the kinetic energy
    :param lambda_: fit of the growth index $\lambda$ of the integral scale
    :return: the prediction and its uncertainty
    """
    return 2 * zeta.index + 2 * lambda_.index - 1, 2 * zeta.index_err + 2 * lambda_.index_err

