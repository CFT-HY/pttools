r"""Fit the suppression factor that matches a PTtools spectrum with a spectrum from external data.

The suppression factor $\Sigma$ is the constant by which a Sound Shell Model spectrum has to be multiplied
to match a reference spectrum, e.g. the GW power spectrum of a hydrodynamic lattice simulation.
For the GW power spectrum the factor is applied to the amplitude, i.e.
$$\mathcal{P}_\text{gw,ref}(z) \approx \Sigma \mathcal{P}_\text{gw,SSM}(z).$$

The spectra of lattice simulations often have a separate peak at high $k$, close to the lattice cutoff.
It is a lattice artifact, and therefore it is detected by :py:func:`lattice_artifact_start`
and excluded from the fit by default.
"""

from dataclasses import dataclass
import enum
import logging
import typing as tp

import numpy as np

from pttools.ssm.spectrum_base import SpectrumBase
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)


class SuppressionFitMethod(enum.StrEnum):
    """Methods for fitting the suppression factor."""

    #: Ratio of the integrals over $\ln z$, i.e. of the total powers within the fit range.
    #: This is consistent with the definition of the suppression datasets of :gowling_2021:`\ `,
    #: which compare the total GW powers.
    INTEGRAL = DEFAULT = "integral"
    #: Least-squares fit in log space, weighted by $\Delta \ln z$,
    #: i.e. the geometric mean of the ratio of the spectra.
    LOG_LSQ = "log_lsq"


@dataclass(frozen=True)
class SuppressionFit:
    """Result of a suppression factor fit."""

    #: $\Sigma$, the fitted suppression factor
    factor: float
    #: The method used for the fit
    method: SuppressionFitMethod
    #: Smallest $z$ included in the fit
    y_min: float
    #: Largest $z$ included in the fit
    y_max: float
    #: Number of data points included in the fit
    n_points: int
    #: $z$ at which the lattice artifact starts, or NaN if no artifact was detected or excluded
    y_artifact: float
    #: RMS of $\log_{10}$ of the ratio of the reference and the scaled model spectra over the fit range,
    #: i.e. a measure of how well the shapes of the spectra agree
    rms_log10_residual: float


def lattice_artifact_start(
        y: th.FloatArr1D,
        p: th.FloatArr1D,
        rise_factor: float = 10.,
        smoothing: int = 5) -> int | None:
    r"""Find the start of a separate high-frequency peak in a power spectrum.

    The spectrum is smoothed with a running mean of $\log p$ over ``smoothing`` points.
    Starting from the main peak, the running minimum of the smoothed spectrum is tracked towards higher $z$.
    If the smoothed spectrum rises above the running minimum by more than ``rise_factor``,
    the spectrum has a separate peak, which is considered to be a lattice artifact.
    The artifact is defined to start at the running minimum.
    Non-positive and non-finite values are ignored.

    It's assumed that the main peak is the global maximum of the spectrum.

    :param y: $z = kR_*$, in increasing order
    :param p: power spectrum
    :param rise_factor: how much the spectrum has to rise above the minimum after the main peak
        to be considered a separate peak
    :param smoothing: number of points in the running mean of $\log p$
    :return: index of ``y`` at which the artifact starts, or None if there is no separate peak
    """
    if y.shape != p.shape:
        raise ValueError(f"y and p must have the same shape. Got: {y.shape}, {p.shape}")
    valid = np.flatnonzero(np.isfinite(p) & (p > 0))
    if valid.size < 2 * smoothing:
        return None
    log_p = np.log(p[valid])
    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        # Pad with the edge values so that the smoothed array has the same length.
        padded = np.pad(log_p, (smoothing // 2, smoothing - 1 - smoothing // 2), mode="edge")
        log_p = np.convolve(padded, kernel, mode="valid")

    i_peak = int(np.argmax(log_p))
    log_rise = np.log(rise_factor)
    i_min = i_peak
    for i in range(i_peak + 1, log_p.size):
        if log_p[i] < log_p[i_min]:
            i_min = i
        elif log_p[i] - log_p[i_min] > log_rise:
            return int(valid[i_min])
    return None


def fit_mask(
        y: th.FloatArr1D,
        p: th.FloatArr1D,
        y_min: float | None = None,
        y_max: float | None = None,
        exclude_artifact: bool = True,
        rise_factor: float = 10.) -> tuple[th.BoolArr1D, float]:
    r"""Select the data points of a reference spectrum to be included in a fit.

    :param y: $z = kR_*$, in increasing order
    :param p: reference power spectrum
    :param y_min: smallest $z$ to include
    :param y_max: largest $z$ to include
    :param exclude_artifact: whether to exclude a separate high-frequency peak,
        see :py:func:`lattice_artifact_start`
    :param rise_factor: see :py:func:`lattice_artifact_start`
    :return: boolean mask of the included points, and $z$ at which the artifact starts (NaN if none)
    """
    mask = np.isfinite(p) & (p > 0) & (y > 0)
    if y_min is not None:
        mask &= y >= y_min
    if y_max is not None:
        mask &= y <= y_max
    y_artifact = np.nan
    if exclude_artifact:
        i_artifact = lattice_artifact_start(y, p, rise_factor=rise_factor)
        if i_artifact is not None:
            y_artifact = float(y[i_artifact])
            mask &= y < y_artifact
            logger.debug("Excluding a lattice artifact starting at z=%s from the fit.", y_artifact)
    return mask, y_artifact


def fit_suppression(
        y_ref: th.FloatArr1D,
        p_ref: th.FloatArr1D,
        y_model: th.FloatArr1D,
        p_model: th.FloatArr1D,
        y_min: float | None = None,
        y_max: float | None = None,
        method: SuppressionFitMethod = SuppressionFitMethod.DEFAULT,
        exclude_artifact: bool = True,
        rise_factor: float = 10.) -> SuppressionFit:
    r"""Fit the suppression factor $\Sigma$ so that $\Sigma p_{\text{model}} \approx p_{\text{ref}}$.

    The model spectrum is interpolated in log-log space to the points of the reference spectrum.
    Only the reference points within the range of the model are used.

    With :py:attr:`SuppressionFitMethod.INTEGRAL`,
    $$\Sigma = \frac{\int d\ln z \, p_{\text{ref}}(z)}{\int d\ln z \, p_{\text{model}}(z)},$$
    and with :py:attr:`SuppressionFitMethod.LOG_LSQ`,
    $$\ln \Sigma = \frac{\int d\ln z \, \ln \frac{p_{\text{ref}}(z)}{p_{\text{model}}(z)}}{\int d\ln z},$$
    where the integrals are over the fit range and computed with the trapezoidal rule.

    :param y_ref: $z = kR_*$ of the reference spectrum, in increasing order
    :param p_ref: reference spectrum, e.g. from a lattice simulation
    :param y_model: $z = kR_*$ of the model spectrum, in increasing order
    :param p_model: model spectrum, e.g. from PTtools, computed without suppression
    :param y_min: smallest $z$ to include in the fit
    :param y_max: largest $z$ to include in the fit
    :param method: fitting method
    :param exclude_artifact: whether to exclude a separate high-frequency peak of the reference spectrum,
        see :py:func:`lattice_artifact_start`
    :param rise_factor: see :py:func:`lattice_artifact_start`
    :return: the fit result
    :raises ValueError: if there are fewer than two points in the fit range
    """
    mask, y_artifact = fit_mask(
        y_ref, p_ref, y_min=y_min, y_max=y_max, exclude_artifact=exclude_artifact, rise_factor=rise_factor)
    model_valid = np.isfinite(p_model) & (p_model > 0)
    y_model_valid = y_model[model_valid]
    mask &= (y_ref >= y_model_valid.min()) & (y_ref <= y_model_valid.max())
    if np.count_nonzero(mask) < 2:  # noqa: PLR2004
        raise ValueError("There must be at least two points in the fit range.")

    y = y_ref[mask]
    ref = p_ref[mask]
    model = np.exp(np.interp(np.log(y), np.log(y_model_valid), np.log(p_model[model_valid])))
    ln_y = np.log(y)

    if method == SuppressionFitMethod.INTEGRAL:
        factor = np.trapezoid(ref, ln_y) / np.trapezoid(model, ln_y)
    elif method == SuppressionFitMethod.LOG_LSQ:
        factor = np.exp(np.trapezoid(np.log(ref / model), ln_y) / (ln_y[-1] - ln_y[0]))
    else:
        raise ValueError(f"Unknown suppression fit method: {method}")

    residual = np.log10(ref / (factor * model))
    return SuppressionFit(
        factor=float(factor),
        method=method,
        y_min=float(y[0]),
        y_max=float(y[-1]),
        n_points=y.size,
        y_artifact=y_artifact,
        rms_log10_residual=float(np.sqrt(np.trapezoid(residual**2, ln_y) / (ln_y[-1] - ln_y[0])))
    )


def fit_suppression_gw(
        reference: SpectrumBase,
        model: SpectrumBase,
        **kwargs: tp.Any) -> SuppressionFit:
    r"""Fit the suppression factor of the GW power growth rate of a model spectrum to a reference spectrum.

    This compares :py:attr:`pttools.ssm.spectrum_base.SpectrumBase.pow_gw_rate` of the spectra.
    The model spectrum should be computed without suppression,
    i.e. with :py:attr:`pttools.ssm.suppression.suppression.SuppressionMethod.NONE`.

    :param reference: reference spectrum, e.g. :py:class:`pttools.lattice.spectrum.LatticeSpectrum`
    :param model: model spectrum, e.g. :py:class:`pttools.ssm.spectrum.SSMSpectrum`
    :param kwargs: arguments for :py:func:`fit_suppression`
    :return: the fit result
    """
    return fit_suppression(
        y_ref=reference.y, p_ref=reference.pow_gw_rate,
        y_model=model.y, p_model=model.pow_gw_rate,
        **kwargs
    )
