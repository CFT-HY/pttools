r"""Fits of power laws to time series.

The power laws are of the form
$$\frac{X(t)}{X(t_0)} = \left( 1 + \frac{t - t_0}{{t}_\text{scale}} \right)^p.$$
For the applications to the decay of the kinetic energy and the growth of the integral scale
of lattice simulations, see :py:mod:`pttools.lattice.asymptote`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

import pttools.type_hints as th


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

    # The trial parameters of the iterations may be outside the domain of the function,
    # e.g. a negative t_scale, for which the power is undefined. The fit handles the resulting nan values.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        popt, pcov = curve_fit(func, t, values / values[0], p0=(1., float(t[-1] - t_0) / 10 or 1.))
    err = np.sqrt(np.diag(pcov))
    return PowerLawFit(index=float(popt[0]), index_err=float(err[0]), t_scale=float(popt[1]), t_0=t_0, factor=factor)
