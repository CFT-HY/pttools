"""Gravitational wave power spectrum as observed today."""

from __future__ import annotations

import functools
import os
import typing as tp

import numpy as np
from numpy.typing import NDArray

from pttools.omgw0 import const
from pttools.omgw0.base import Omgw0Base
from pttools.omgw0.compute import OmGw0Input, OmGw0Results, SNRSettings, compute, compute_gw
from pttools.omgw0.const import LISA_OBS_TIME
from pttools.omgw0.export import SPECTRUM_FIELDS, SPECTRUM_FIELDS_F, SPECTRUM_FIELDS_Y
from pttools.omgw0.factors import H
from pttools.ssm.const import (
    DEFAULT_A_STAR_A_R_RATIO,
    DEFAULT_EPS_LOOKUP,
    DEFAULT_LIFETIME_DISTRIBUTION_A,
    DEFAULT_N_SH,
    DEFAULT_N_T,
    DEFAULT_N_Z_LOOKUP,
    DEFAULT_Y,
    T_TILDE_MAX,
    T_TILDE_MIN,
    Z_ST_THRESH,
)
from pttools.ssm.nucleation import DEFAULT_NUC_TYPE, NucType
from pttools.ssm.spectrum import SSMSpectrum
from pttools.ssm.ssm import DEFAULT_LAMBDA_CORRECTION, LambdaCorrection
from pttools.ssm.suppression import DEFAULT_SUPPRESSION, Suppression, SuppressionMethod
import pttools.type_hints as th
from pttools.type_hints import FloatArr1D
from pttools.utils.decorators import cached_property

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.figure import Figure

    from pttools.bubble import Bubble
    from pttools.utils.fields import Fields


class Spectrum(Omgw0Base, SSMSpectrum):
    r"""A spectrum object that includes the conversion to the GW power spectrum today $\Omega_{\text{gw},0}$."""

    FIELDS: tp.ClassVar[Fields] = SPECTRUM_FIELDS
    #: The exportable fields of the spectra that have been given the $y$ array.
    #: For these, $y$ is shared by the spectra of a file.
    #: This is an alias of :py:attr:`FIELDS` for consistency with :py:attr:`FIELDS_F`.
    #: Subclasses that change :py:attr:`FIELDS` should also change this and :py:attr:`FIELDS_F`.
    FIELDS_Y: tp.ClassVar[Fields] = SPECTRUM_FIELDS_Y
    #: The exportable fields of the spectra that have been given the frequencies $f$ instead of $y$.
    #: For these, $f$ is shared by the spectra of a file instead of $y$.
    FIELDS_F: tp.ClassVar[Fields] = SPECTRUM_FIELDS_F

    def __init__(
            self,
            bubble: Bubble,
            # Input parameters
            beta_tilde: float | None = None,
            r_star: float | None = None,
            y: th.FloatArr1D | None = None,
            f: th.FloatArr1D | None = None,
            a_star_a_r_ratio: float = DEFAULT_A_STAR_A_R_RATIO,
            N_sh: float = DEFAULT_N_SH,
            nuc_type: NucType = DEFAULT_NUC_TYPE,
            omega_barotropic: float | None = None,
            # Suppression
            suppression: Suppression = DEFAULT_SUPPRESSION,
            suppression_method: SuppressionMethod = SuppressionMethod.DEFAULT,
            # Omega_gw_0 input parameters
            T_star: float | None = None,
            g_star: float | None = None,
            gs_star: float | None = None,
            # Accuracy settings
            nT: int = DEFAULT_N_T,
            nx_P_tilde_gw: int | None = None,
            n_z_lookup: int = DEFAULT_N_Z_LOOKUP,
            T_tilde_min: float = T_TILDE_MIN,
            T_tilde_max: float = T_TILDE_MAX,
            z_st_thresh: float = Z_ST_THRESH,
            # Switches
            compute: bool = True,
            compute_snr: bool = False,
            low_k: bool = True,
            low_k_cs_friction: bool = False,
            parallel: bool = True,
            # Labels
            label_latex: str | None = None,
            label_unicode: str | None = None):
        r"""Create the spectrum, and compute it unless ``compute=False``.

        :param bubble: the Bubble object
        :param beta_tilde: nucleation rate parameter $\tilde{\beta} \equiv \frac{\beta}{H_*}$
        :param r_star: Hubble-scaled mean bubble spacing $r_*$
        :param y: $z = k R_*$ array.
            If neither $y$ nor $f$ is given, the default of :py:mod:`pttools.ssm.const` is used.
        :param f: $f$, frequencies today, which are converted to the $y$ array with :py:meth:`z_from_f`.
            Cannot be given together with $y$.
        :param N_sh: $N_\text{sh}$, number of shock formation times
        :param nuc_type: nucleation type
        :param omega_barotropic: $\omega$, barotropic equation of state parameter.
            If not given, it's computed from Bubble.
        :param T_star: $T_*$, temperature at the time of GW production in GeV.
            If not given, it's taken from the bubble if the temperature of the model is in physical units.
        :param g_star: $g_*$, degrees of freedom override for pressure at the time of GW production
        :param gs_star: $g_{s,*}$ degrees of freedom override for entropy at the time of GW production.
            If only $g_*$ is given, $g_{s,*} = g_*$ is assumed.
        :param nT: number of points in the t array
        :param n_z_lookup: number of points in the lookup arrays
        :param z_st_thresh: for $z$ values above z_sh_tresh,
            use approximation rather than doing the sine transform integral.
        :param compute: whether to compute the spectrum immediately
        :param compute_snr: whether to compute the signal-to-noise ratio for LISA together with the spectrum
            in the same Numba-compiled function, so that :py:meth:`snr` returns the precomputed result
            when called with the default arguments. See :py:meth:`compute` for custom settings.
        :param low_k: whether to use the :giombi_2024_cs: approximation for low $k$
        :param low_k_cs_friction: whether to compute the acoustic friction of the low-$k$ approximation
            from $c_s$ instead of $\nu$, see :py:func:`pttools.ssm.low_k.kernel.kernel_low`.
            This generalises the articles to $c_s^2 \neq \omega$, and is disabled by default.
            It should be considered an estimate, see the caveat in
            :py:func:`pttools.ssm.low_k.kernel.acoustic_friction_exponent`.
        :param parallel: whether to use multiple CPU cores
        """
        if f is not None:
            if y is not None:
                raise ValueError("Either y or f can be provided, but not both.")
            if np.isnan(f).any():
                raise ValueError("f must not contain nan values.")
        # The spectrum is computed only after the y array has been obtained,
        # since converting f to y requires T_star, g_star and r_star.
        super().__init__(
            bubble=bubble,
            beta_tilde=beta_tilde,
            r_star=r_star,
            y=DEFAULT_Y if y is None else y,  # When f is set, this is a placeholder that will be replaced below.
            z_st_thresh=z_st_thresh,
            nuc_type=nuc_type,
            omega_barotropic=omega_barotropic,
            suppression=suppression,
            suppression_method=suppression_method,
            a_star_a_r_ratio=a_star_a_r_ratio,
            N_sh=N_sh,
            nT=nT,
            nx_P_tilde_gw=nx_P_tilde_gw,
            n_z_lookup=n_z_lookup,
            T_tilde_min=T_tilde_min,
            T_tilde_max=T_tilde_max,
            compute=False,
            low_k=low_k,
            low_k_cs_friction=low_k_cs_friction,
            label_latex=label_latex,
            label_unicode=label_unicode
        )
        # This is needed for T_star, g_star and gs_star, and beta_tilde -> r_star conversion
        if not self.bubble.solved:
            self.bubble.solve()

        bubble_temp_physical = bubble.model.temperature_is_physical
        self.T_star: float = T_star if T_star is not None \
            else bubble.T_star * bubble.model.temperature_unit_gev if bubble_temp_physical \
            else const.DEFAULT_T_STAR
        self.g_star: float = g_star if g_star is not None \
            else bubble.g_star if bubble_temp_physical \
            else const.DEFAULT_G_STAR
        self.gs_star: float = gs_star if gs_star is not None \
            else g_star if g_star is not None \
            else bubble.gs_star if bubble_temp_physical \
            else const.DEFAULT_G_STAR

        # The frequencies given as an argument are returned by f() as is, instead of converting them back from y.
        self._f_given: th.FloatArr1D | None = f
        if f is not None:
            self.y: th.FloatArr1D = self.z_from_f(f)

        #: Whether the signal-to-noise ratio is computed together with the spectrum
        self.compute_snr: bool = compute_snr
        #: Settings of the signal-to-noise ratio that is computed together with the spectrum
        self.snr_settings: SNRSettings = SNRSettings()
        # The signal-to-noise ratio computed together with the spectrum
        self._omgw0_res: OmGw0Results | None = None

        if compute:
            self.compute(parallel=parallel)

    # =====
    # Properties
    # =====

    @property
    def f_given(self) -> bool:
        """Whether the frequencies $f$ were given as an argument instead of the $y$ array."""
        return self._f_given is not None

    @cached_property
    @tp.override
    def _f_y(self) -> th.FloatArr1D:
        # The frequencies given as an argument are returned as is, instead of converting them back from y.
        if self._f_given is not None:
            return self._f_given
        return self.f(self.y)

    @cached_property
    @tp.override
    def H_star(self) -> float:
        return float(H(self.e_star))

    @cached_property
    @tp.override
    def R_star(self) -> float:
        return self.r_star / self.H_star

    @functools.cached_property
    def R_star_m(self) -> th.FloatOrArr:
        r"""Mean bubble separation $R_*$, in meters, presuming that $T$ is in GeV."""
        return self.R_star * const.GEV_IN_J * const.PLANCK_LENGTH

    # =====
    # Methods
    # =====

    @tp.override
    def compute(
            self,
            eps_lookup: float = DEFAULT_EPS_LOOKUP,
            lifetime_distribution_a: float = DEFAULT_LIFETIME_DISTRIBUTION_A,
            lambda_correction: LambdaCorrection = DEFAULT_LAMBDA_CORRECTION,
            parallel: bool = True,
            compute_snr: bool | None = None,
            snr_settings: SNRSettings | None = None) -> None:
        r"""Compute the spectra and optionally the signal-to-noise ratio, and store them as attributes.

        The spectra and the signal-to-noise ratio are computed in one Numba-compiled function
        :py:func:`pttools.omgw0.compute.compute`, so that the GIL is not acquired in between.

        :param eps_lookup: small offset used when generating the lookup table of $z$ values for the GW spectrum
        :param lifetime_distribution_a: $a$, parameter of the bubble lifetime distribution,
            see :py:func:`pttools.ssm.nucleation.lifetime_distribution`
        :param lambda_correction: kinetic energy term of $\lambda$, see :py:class:`pttools.ssm.ssm.LambdaCorrection`
        :param parallel: whether to use multiple CPU cores
        :param compute_snr: whether to compute the signal-to-noise ratio.
            If not given, the value given to the constructor is used.
        :param snr_settings: settings for the signal-to-noise ratio.
            :py:meth:`snr` returns the precomputed result when called with the same settings.
            If not given, the previous settings are used, which are the defaults unless changed.
        """
        if compute_snr is not None:
            self.compute_snr = compute_snr
        if snr_settings is not None:
            self.snr_settings = snr_settings
        self._prepare_compute(
            eps_lookup=eps_lookup, lifetime_distribution_a=lifetime_distribution_a, lambda_correction=lambda_correction)
        base_res, gw_res, omgw0_res = compute(
            ssm_inp=self._ssm_input(), omgw0_inp=self._omgw0_input(), parallel=parallel)
        self._store_base(base_res)
        self._store_gw(gw_res)
        self._store_omgw0(omgw0_res)

    @tp.override
    def _compute_gw(self) -> None:
        if self.base_res is None:
            raise RuntimeError("The quantities that do not depend on r_star have not been computed.")
        gw_res, omgw0_res = compute_gw(
            ssm_inp=self._ssm_input(), omgw0_inp=self._omgw0_input(), base_res=self.base_res)
        self._store_gw(gw_res)
        self._store_omgw0(omgw0_res)

    def _omgw0_input(self) -> OmGw0Input:
        """Input parameters for :py:mod:`pttools.omgw0.compute`."""
        return OmGw0Input(
            f=self.f(),
            F_gw0_h2=self.F_gw0_h2(
                ge0_photon=self.snr_settings.ge0_photon,
                gs0=self.snr_settings.gs0,
                om_gamma0_h2=self.snr_settings.om_gamma0_h2
            ),
            # The scalars are converted to float, so that the compiled computation is not compiled separately
            # for each combination of int and float arguments.
            mean_adiabatic_index=float(self.bubble.mean_adiabatic_index),
            r_star=float(self.r_star),
            nu_gdh2024=float(self.nu_gdh2024),
            dilution_of_e=float(self.dilution_of_e),
            suppression_factor=float(self.suppression_factor),
            low_k=self.low_k,
            compute_snr=self.compute_snr,
            snr=self.snr_settings
        )

    def _store_omgw0(self, omgw0_res: OmGw0Results) -> None:
        """Store the results of :py:func:`pttools.omgw0.compute.compute_omgw0`."""
        self.pow_gw = omgw0_res.pow_gw
        self._omgw0_res = omgw0_res if self.compute_snr else None

    @tp.override
    def with_params(  # pyrefly: ignore[bad-override]
            self,
            r_star: float | None = None,
            beta_tilde: float | None = None,
            a_star_a_r_ratio: float | None = None,
            N_sh: float | None = None,
            omega_barotropic: float | None = None,
            suppression: Suppression | None = None,
            suppression_method: SuppressionMethod | None = None,
            label_latex: str | None = None,
            label_unicode: str | None = None,
            T_star: float | None = None,
            g_star: float | None = None,
            gs_star: float | None = None,
            compute_snr: bool | None = None,
            y: FloatArr1D | None = None,
            f: FloatArr1D | None = None) -> tp.Self:
        r"""Create a new spectrum with different parameters, reusing the computations of this spectrum.

        See :py:meth:`pttools.ssm.spectrum.SSMSpectrum.with_params`.
        Unless $y$ or $f$ is given, the new spectrum has the same $z$ array as this spectrum.
        Therefore, if the frequencies $f$ were given to this spectrum,
        the frequencies of the new spectrum differ from them when $r_*$, $T_*$, $g_*$ or $g_{s,*}$ is changed.
        To keep the frequencies, give them as ``f``.
        They are then converted to the $y$ array of the new spectrum with the parameters of the new spectrum,
        as in the constructor, and the spectrum is interpolated to it.
        The signal-to-noise ratio is computed with the same settings as for this spectrum.

        :param r_star: $r_*$, Hubble-scaled mean bubble spacing
        :param beta_tilde: $\tilde{\beta} \equiv \frac{\beta}{H_*}$, nucleation rate parameter
        :param a_star_a_r_ratio: $\frac{a_*}{a_r}$
        :param N_sh: $N_\text{sh}$, number of shock formation times
        :param omega_barotropic: $\omega$, barotropic equation of state parameter
        :param suppression: suppression data
        :param suppression_method: suppression method
        :param label_latex: label of the new spectrum in LaTeX format
        :param label_unicode: label of the new spectrum in Unicode format
        :param T_star: $T_*$, temperature at the time of GW production in GeV
        :param g_star: $g_*$, degrees of freedom for pressure at the time of GW production.
            If $g_*$ is given without $g_{s,*}$, $g_{s,*} = g_*$ is assumed, as in the constructor.
        :param gs_star: $g_{s,*}$, degrees of freedom for entropy at the time of GW production
        :param compute_snr: whether to compute the signal-to-noise ratio.
            If not given, the value of this spectrum is used.
        :param y: $z = kR_*$ array of the new spectrum, to which this spectrum is interpolated.
            Cannot be given together with $f$.
        :param f: $f$, frequencies today of the new spectrum, to which this spectrum is interpolated.
            Cannot be given together with $y$.
        :return: the new spectrum
        :raises RuntimeError: if this spectrum has not been computed
        :raises ValueError: if both $y$ and $f$ are given,
            or if the $y$ array of the new spectrum is not within the range of the $y$ array of this spectrum
        """
        if y is not None and f is not None:
            raise ValueError("Either y or f can be provided, but not both.")
        if g_star is not None and gs_star is None:
            gs_star = g_star
        return self._with_params(
            {
                "r_star": r_star, "beta_tilde": beta_tilde,
                "a_star_a_r_ratio": a_star_a_r_ratio, "N_sh": N_sh, "omega_barotropic": omega_barotropic,
                "suppression": suppression, "suppression_method": suppression_method,
                "label_latex": label_latex, "label_unicode": label_unicode,
                "T_star": T_star, "g_star": g_star, "gs_star": gs_star, "compute_snr": compute_snr,
                "y": y, "f": f
            }
        )

    @tp.override
    def _init_kwargs(self) -> dict[str, tp.Any]:
        return {
            **super()._init_kwargs(),
            "T_star": self.T_star,
            "g_star": self.g_star,
            "gs_star": self.gs_star,
            "compute_snr": self.compute_snr,
        }

    @tp.override
    def _reuse(self, other: SSMSpectrum) -> None:
        if isinstance(other, Spectrum):
            self.snr_settings = other.snr_settings
        super()._reuse(other)

    @tp.override
    def snr(
            self,
            # Arrays
            noise: FloatArr1D | None = None,
            f_noise: FloatArr1D | None = None,
            # Scalars
            ge0_photon: float = const.GE0_PHOTON,
            gs0: float = const.GS0,
            obs_time: float = LISA_OBS_TIME,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2,
            # Switches
            noise_eb: bool = True,
            noise_gb: bool = True,
            noise_ins: bool = True,
        ) -> tuple[float, FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D]:
        """Signal-to-noise ratio for LISA, see :py:meth:`pttools.omgw0.base.Omgw0Base.snr`.

        If the signal-to-noise ratio was computed together with the spectrum with the same settings
        (see :py:meth:`compute`), the precomputed result is returned.
        """
        settings = SNRSettings(
            noise=noise, f_noise=f_noise, obs_time=obs_time,
            ge0_photon=ge0_photon, gs0=gs0, om_gamma0_h2=om_gamma0_h2,
            noise_eb=noise_eb, noise_gb=noise_gb, noise_ins=noise_ins
        )
        res = self._omgw0_res
        if res is not None and settings.matches(self.snr_settings):
            return res.snr, self.f(), res.omgw0_h2, res.f_noise, res.noise
        return super().snr(
            noise=noise, f_noise=f_noise, ge0_photon=ge0_photon, gs0=gs0, obs_time=obs_time,
            om_gamma0_h2=om_gamma0_h2, noise_eb=noise_eb, noise_gb=noise_gb, noise_ins=noise_ins
        )

    # -----
    # Plotting
    # -----

    def plot_multi(
            self,
            fig: Figure | None = None,
            path: str | os.PathLike[str] | None = None,
            **kwargs: tp.Any) -> tuple[Figure, th.AxesArr2D]:
        """Plot the fluid velocity profile, the velocity and GW power spectra, and the GW power spectrum today.

        The plots are in a 2x2 grid.

        :param fig: the figure to plot on
        :param path: if given, save the figure to this path
        :param kwargs: additional arguments for :py:func:`pttools.plot.spectra.plot_spectra_multi`
        :return: the figure and the 2D array of axes
        """
        from pttools.plot.spectra import plot_spectra_multi  # noqa: PLC0415
        return plot_spectra_multi([self], fig, path, **kwargs)

    def plot_multi_flat(
            self,
            fig: Figure | None = None,
            path: str | os.PathLike[str] | None = None,
            label: str | None = None,
            legend: bool = False,
            **kwargs: tp.Any) -> tuple[Figure, th.AxesArr1D]:
        """Plot the fluid velocity profile, the velocity and GW power spectra, and the GW power spectrum today.

        The plots are in a single row.

        :param fig: the figure to plot on
        :param path: if given, save the figure to this path
        :param label: label of the spectrum
        :param legend: whether to show the legend
        :param kwargs: additional arguments for :py:func:`pttools.plot.spectra.plot_spectra_multi_flat`
        :return: the figure and the 1D array of axes
        """
        from pttools.plot.spectra import plot_spectra_multi_flat  # noqa: PLC0415
        return plot_spectra_multi_flat([self], fig=fig, path=path, labels=[label], legend=legend, **kwargs)


# These are object arrays. Numpy typing has no way of expressing the element type of an object array,
# but declaring the element type here does give the correct types when the arrays are indexed.
type SpectrumArr = NDArray[Spectrum]  # pyrefly: ignore[bad-specialization]
type SpectrumArr2D = np.ndarray[tuple[int, int], np.dtype[Spectrum]]  # pyrefly: ignore[bad-specialization]
type SpectrumArr3D = np.ndarray[tuple[int, int, int], np.dtype[Spectrum]]  # pyrefly: ignore[bad-specialization]
