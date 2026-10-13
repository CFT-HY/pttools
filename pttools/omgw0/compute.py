r"""Functions for organizing the computation of the GW spectra today and their signal-to-noise ratios.

These extend :py:mod:`pttools.ssm.compute` with the conversion to the GW power spectrum today
$\Omega_{\text{gw},0} h^2$ and the signal-to-noise ratio,
so that all of them can be computed in one Numba-compiled function without acquiring the GIL in between.
"""

import typing as tp

import numpy as np

from pttools.omgw0 import const
from pttools.omgw0.const import LISA_OBS_TIME
from pttools.omgw0.noise import signal_to_noise_ratio
from pttools.speedup import njit
from pttools.ssm.compute import SSMBaseResults, SSMGWResults, SSMInput
from pttools.ssm.compute import compute as compute_ssm
from pttools.ssm.compute import compute_gw as compute_gw_ssm
from pttools.ssm.pow_spec import pow_spec
from pttools.ssm.spec_den_gw import spec_den_gw_scaling
from pttools.type_hints import FloatArr1D


class SNRSettings(tp.NamedTuple):
    """Settings for the signal-to-noise ratio, see :py:meth:`pttools.omgw0.spectrum.Spectrum.snr`."""

    #: $\Omega_\text{noise} h^2$, or ``None`` to generate the LISA noise
    noise: FloatArr1D | None = None
    #: frequencies for the noise (assumed to be the same as for the signal, if not provided)
    f_noise: FloatArr1D | None = None
    #: observation time (s)
    obs_time: float = LISA_OBS_TIME
    #: $g_{e\gamma 0}$, degrees of freedom for energy density of photons today
    ge0_photon: float = const.GE0_PHOTON
    #: $g_{s,0}$, degrees of freedom today for entropy
    gs0: float = const.GS0
    #: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
    om_gamma0_h2: float = const.OMEGA_PHOTON_H2
    #: whether to generate extragalactic compact binary noise when noise is not provided
    noise_eb: bool = True
    #: whether to generate galactic compact binary noise when noise is not provided
    noise_gb: bool = True
    #: whether to generate instrument noise when noise is not provided
    noise_ins: bool = True

    def matches(self, other: "SNRSettings") -> bool:
        """Whether the settings are the same, where the arrays are compared by identity."""
        return all(
            a is b if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) else a == b
            for a, b in zip(self, other, strict=True)
        )


class OmGw0Input(tp.NamedTuple):
    r"""Input parameters for the conversion to $\Omega_{\text{gw},0} h^2$ and the signal-to-noise ratio."""

    #: $f$, frequencies today corresponding to the $z$ array
    f: FloatArr1D
    #: $F_{\text{gw},0} h^2$, power attenuation following the end of the radiation era
    F_gw0_h2: float
    #: $\Gamma$, mean adiabatic index
    mean_adiabatic_index: float
    #: $r_*$, Hubble-scaled mean bubble spacing
    r_star: float
    #: $\nu_\text{gdh2024}$
    nu_gdh2024: float
    #: dilution of the background energy density
    dilution_of_e: float
    #: $\Sigma$, suppression factor
    suppression_factor: float
    #: whether to use the low-$k$ approximation
    low_k: bool
    #: whether to compute the signal-to-noise ratio
    compute_snr: bool
    #: settings for the signal-to-noise ratio
    snr: SNRSettings


class OmGw0Results(tp.NamedTuple):
    r"""Results of :py:func:`compute_omgw0`."""

    #: $\mathcal{P}_\text{gw}$, GW power spectrum at the time of production
    pow_gw: FloatArr1D
    #: $\Omega_{\text{gw},0} h^2$, GW power spectrum today
    omgw0_h2: FloatArr1D
    #: $\rho$, signal-to-noise ratio, or nan if it was not computed
    snr: float
    #: frequencies of the noise, or an empty array if the signal-to-noise ratio was not computed
    f_noise: FloatArr1D
    #: $\Omega_\text{noise} h^2$, or an empty array if the signal-to-noise ratio was not computed
    noise: FloatArr1D


@njit(nogil=True)
def compute(
        ssm_inp: SSMInput,
        omgw0_inp: OmGw0Input,
        parallel: bool = True) -> tuple[SSMBaseResults, SSMGWResults, OmGw0Results]:
    r"""Compute the Sound Shell Model spectra, $\Omega_{\text{gw},0} h^2$ and optionally the signal-to-noise ratio.

    This is in one Numba-compiled function so that the GIL is released for the entire computation.

    :param ssm_inp: input parameters of the Sound Shell Model
    :param omgw0_inp: input parameters of the conversion to $\Omega_{\text{gw},0} h^2$
    :param parallel: whether to use multiple CPU cores
    :return: results of :py:func:`pttools.ssm.compute.compute_base`, :py:func:`pttools.ssm.compute.compute_gw`
        and :py:func:`compute_omgw0`
    """
    base_res, gw_res = compute_ssm(inp=ssm_inp, parallel=parallel)
    omgw0_res = compute_omgw0(y=ssm_inp.y, inp=omgw0_inp, ubarf2=base_res.ubarf2, gw_res=gw_res)
    return base_res, gw_res, omgw0_res


@njit(nogil=True)
def compute_gw(
        ssm_inp: SSMInput,
        omgw0_inp: OmGw0Input,
        base_res: SSMBaseResults) -> tuple[SSMGWResults, OmGw0Results]:
    r"""Compute the GW spectra, $\Omega_{\text{gw},0} h^2$ and optionally the signal-to-noise ratio.

    This reuses the results of :py:func:`pttools.ssm.compute.compute_base`.

    :param ssm_inp: input parameters of the Sound Shell Model
    :param omgw0_inp: input parameters of the conversion to $\Omega_{\text{gw},0} h^2$
    :param base_res: results of :py:func:`pttools.ssm.compute.compute_base`
    :return: results of :py:func:`pttools.ssm.compute.compute_gw` and :py:func:`compute_omgw0`
    """
    gw_res = compute_gw_ssm(inp=ssm_inp, base_res=base_res)
    omgw0_res = compute_omgw0(y=ssm_inp.y, inp=omgw0_inp, ubarf2=base_res.ubarf2, gw_res=gw_res)
    return gw_res, omgw0_res


@njit(nogil=True)
def compute_omgw0(y: FloatArr1D, inp: OmGw0Input, ubarf2: float, gw_res: SSMGWResults) -> OmGw0Results:
    r"""Compute $\mathcal{P}_\text{gw}$, $\Omega_{\text{gw},0} h^2$ and optionally the signal-to-noise ratio.

    $$\Omega_{\text{gw},0} h^2 = F_{\text{gw},0} h^2 \mathcal{P}_\text{gw}$$
    See :py:func:`pttools.ssm.spec_den_gw.spec_den_gw_scaling`, :py:func:`pttools.ssm.pow_spec.pow_spec`,
    :py:func:`pttools.omgw0.factors.F_gw0_h2` and :py:func:`pttools.omgw0.noise.signal_to_noise_ratio`.

    :param y: $z = kR_*$ array
    :param inp: input parameters
    :param ubarf2: $\bar{U}_f^2$
    :param gw_res: results of :py:func:`pttools.ssm.compute.compute_gw`
    :return: $\mathcal{P}_\text{gw}$, $\Omega_{\text{gw},0} h^2$ and the signal-to-noise ratio
    """
    spec_den_gw = gw_res.spec_den_gw_expanded if inp.low_k else gw_res.spec_den_gw_ssm
    scaling = spec_den_gw_scaling(
        ubarf2=ubarf2,
        mean_adiabatic_index=inp.mean_adiabatic_index,
        r_star=inp.r_star,
        nu=inp.nu_gdh2024,
        dilution_of_e=inp.dilution_of_e,
        suppression_factor=inp.suppression_factor
    )
    pow_gw = scaling * pow_spec(z=y, spec_den=spec_den_gw)
    omgw0_h2 = inp.F_gw0_h2 * pow_gw
    if inp.compute_snr:
        snr, f_noise, noise = signal_to_noise_ratio(
            f=inp.f, signal=omgw0_h2, obs_time=inp.snr.obs_time,
            noise=inp.snr.noise, f_noise=inp.snr.f_noise,
            noise_eb=inp.snr.noise_eb, noise_gb=inp.snr.noise_gb, noise_ins=inp.snr.noise_ins
        )
        return OmGw0Results(pow_gw=pow_gw, omgw0_h2=omgw0_h2, snr=snr, f_noise=f_noise, noise=noise)
    empty = np.empty(0, dtype=np.float64)
    return OmGw0Results(pow_gw=pow_gw, omgw0_h2=omgw0_h2, snr=np.nan, f_noise=empty, noise=empty)
