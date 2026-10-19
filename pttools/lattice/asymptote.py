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
The generic fitting functions are in :py:mod:`pttools.math.asymptote` and :py:mod:`pttools.math.power_law`.
"""

from dataclasses import dataclass
import logging

import numpy as np

from pttools.math.asymptote import AsymptoteFit, fit_asymptote, fit_start_index
from pttools.math.power_law import PowerLawFit
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)

#: Default window length of the running mean of $\Omega_\text{gw}(t)$ before fitting,
#: as in the scripts of :correia_2025:`\ `
DEFAULT_RUNNING_MEAN_WINDOW: int = 23


@dataclass(frozen=True, slots=True)
class PowGwAsymptoteFit:
    r"""Fits of the approach to an asymptote to the time series of $\mathcal{P}_\text{gw}(k, t)$ of each wavenumber.

    The fitted function is :py:func:`pttools.math.asymptote.asymptotic_growth`.
    """

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
    r"""Fit the approach to an asymptote to the time series of $\mathcal{P}_\text{gw}(k, t)$ of each wavenumber.

    The fitted function is :py:func:`pttools.math.asymptote.asymptotic_growth`.

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
