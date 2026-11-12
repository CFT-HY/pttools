"""Gravitational wave power spectrum as observed today."""

from __future__ import annotations

import dataclasses
import functools
import os
import typing as tp

import numpy as np
from numpy.typing import NDArray

from pttools.omgw0 import const, freq
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
from pttools.ssm.settings import SSMSettings
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


#: Default density of the $z$ arrays of :py:meth:`Spectrum.z_covering` relative to the density of the frequencies.
#: The spectra created from a spectrum computed for the $z$ array with :py:meth:`Spectrum.with_params`
#: are interpolated to the frequencies, and the interpolation error decreases with the square of the density.
#: With the density of the frequencies, the interpolation error near the peak can be a few percent,
#: and with four times the density, it is at most a few tenths of a percent, which is less than the error
#: of about 1 % of the accuracy settings of :ref:`sphx_glr_auto_examples_dataset_save.py`.
#: The computation time of the spectrum depends only weakly on the number of points of the $z$ array.
Z_COVERING_DENSITY: float = 4

#: Default margin of the $z$ arrays of :py:meth:`Spectrum.z_covering` beyond the $z$ values of the frequencies,
#: in decades at each end.
#: The spectra are inaccurate near the ends of their $z$ arrays, especially at the upper end, where the error
#: can be over 100 % within a few tenths of a decade from the end when the $z$ array extends above the peak.
#: A margin of half a decade removes this edge effect,
#: and larger margins do not improve the accuracy further.
Z_COVERING_MARGIN: float = 0.5


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
            allow_beta_tilde_for_simultaneous: bool = False,
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
        :param allow_beta_tilde_for_simultaneous:
            allow $\tilde{\beta}$ as an input parameter for simultaneous nucleation
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
            allow_beta_tilde_for_simultaneous=allow_beta_tilde_for_simultaneous,
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
    # z arrays covering frequencies
    # -----

    @classmethod
    def covering(
            cls,
            bubble: Bubble,
            f: FloatArr1D,
            T_star: th.FloatOrArr1D,
            g_star: th.FloatOrArr1D,
            beta_tilde: th.FloatOrArr1D | None = None,
            r_star: th.FloatOrArr1D | None = None,
            settings: SSMSettings | None = None,
            f_star0_factor: float = 1.,
            points_per_decade: float | None = None,
            margin: float = Z_COVERING_MARGIN,
            **kwargs: tp.Any) -> tp.Self:
        r"""A spectrum that covers the frequencies ``f`` for all the given parameter values.

        The spectrum is computed for the $z$ array of :py:meth:`z_covering`,
        with the accuracy settings of :py:meth:`settings_covering`.
        The spectra of the individual parameter values can then be created from it with :py:meth:`with_params`,
        e.g. ``spectrum.with_params(beta_tilde=..., T_star=..., g_star=..., f=f)``,
        which reuses the computationally heavy part, and interpolates the spectrum to the frequencies.
        This is much faster than computing each of them separately,
        but the results differ from the separately computed spectra by the interpolation error,
        see :py:meth:`pttools.ssm.spectrum.SSMSpectrum.with_params`.
        The parameters that cannot be changed with :py:meth:`with_params`, e.g. the nucleation type,
        are given as ``kwargs`` and are the same for all the spectra created from this spectrum.

        The spectrum itself is created with the first valid value of $\tilde{\beta}$ or $r_*$,
        and the first values of $T_*$ and $g_*$.

        :param bubble: fluid shell of the spectra
        :param f: frequencies $f$ to be covered
        :param T_star: $T_*$, temperatures at which the GWs were produced, a scalar or an array
        :param g_star: $g_*$, degrees of freedom, a scalar or an array
        :param beta_tilde: $\tilde{\beta}$ values of the spectra
        :param r_star: $r_*$ values of the spectra
        :param settings: accuracy settings for the frequencies,
            defaults to :py:class:`~pttools.ssm.settings.SSMSettings`
        :param f_star0_factor: correction factor for $f_{\ast,0}$, see :py:meth:`z_covering`
        :param points_per_decade: number of $z$ points per decade, see :py:meth:`z_covering`
        :param margin: margin of the $z$ array in decades, see :py:meth:`z_covering`
        :param kwargs: other arguments of the spectrum, e.g. ``nuc_type`` or ``parallel``
        :return: the spectrum
        :raises ValueError: if not exactly one of $\tilde{\beta}$ and $r_*$ is given,
            or if none of their values are valid
        """
        beta_tildes = None if beta_tilde is None else np.atleast_1d(np.asarray(beta_tilde, dtype=float))
        r_stars = None if r_star is None else np.atleast_1d(np.asarray(r_star, dtype=float))
        z = cls.z_covering(
            f=f, bubble=bubble, T_star=T_star, g_star=g_star, beta_tilde=beta_tildes, r_star=r_stars,
            f_star0_factor=f_star0_factor, points_per_decade=points_per_decade, margin=margin)
        # Any valid value can be used, as with_params() changes it.
        value = cls._valid_r_stars(bubble, beta_tilde=beta_tildes, r_star=r_stars)[0][0]
        return cls(
            bubble,
            y=z,
            beta_tilde=None if beta_tildes is None else value,
            r_star=value if beta_tildes is None else None,
            T_star=float(np.atleast_1d(T_star)[0]),
            g_star=float(np.atleast_1d(g_star)[0]),
            **cls.settings_covering(z=z, f=f, settings=settings).spectrum_kwargs(),
            **kwargs
        )

    @staticmethod
    def n_z_lookup_covering(z: FloatArr1D, f: FloatArr1D, n_z_lookup: int = DEFAULT_N_Z_LOOKUP) -> int:
        """Number of points in the lookup arrays for a $z$ array, giving the same density as for the frequencies.

        The lookup arrays are logarithmically spaced over the range of the $z$ array.
        This scales ``n_z_lookup`` by the ratio of the numbers of decades spanned by ``z`` and ``f``,
        so that a spectrum computed for ``z`` is as accurate as one computed for the frequencies.

        :param z: $z$ array, e.g. from :py:meth:`z_covering`
        :param f: frequencies $f$ for which the spectra would otherwise be computed
        :param n_z_lookup: number of points in the lookup arrays for the frequencies
        :return: number of points in the lookup arrays for ``z``, at least ``n_z_lookup``
        """
        ratio = np.log10(z.max() / z.min()) / np.log10(f.max() / f.min())
        return max(n_z_lookup, int(np.ceil(n_z_lookup * ratio)))

    @classmethod
    def settings_covering(cls, z: FloatArr1D, f: FloatArr1D, settings: SSMSettings | None = None) -> SSMSettings:
        """Accuracy settings for a $z$ array, with ``n_z_lookup`` scaled by :py:meth:`n_z_lookup_covering`.

        :param z: $z$ array, e.g. from :py:meth:`z_covering`
        :param f: frequencies $f$ for which the spectra would otherwise be computed
        :param settings: accuracy settings for the frequencies,
            defaults to :py:class:`~pttools.ssm.settings.SSMSettings`
        :return: accuracy settings for ``z``
        """
        if settings is None:
            settings = SSMSettings()
        return dataclasses.replace(
            settings, n_z_lookup=cls.n_z_lookup_covering(z=z, f=f, n_z_lookup=settings.n_z_lookup))

    @staticmethod
    def z_covering(
            f: FloatArr1D,
            bubble: Bubble,
            T_star: th.FloatOrArr1D,
            g_star: th.FloatOrArr1D,
            beta_tilde: FloatArr1D | None = None,
            r_star: FloatArr1D | None = None,
            f_star0_factor: float = 1.,
            points_per_decade: float | None = None,
            margin: float = Z_COVERING_MARGIN) -> FloatArr1D:
        r"""$z$ array that covers the frequencies ``f`` for all the given parameter values.

        The $z$ values corresponding to the frequencies are
        $$z = \frac{f}{f_{\ast,0}} r_*,$$
        inverted from :gowling_2021:`\ ` eq. 2.12,
        where $f_{\ast,0}$ is that of :py:func:`pttools.omgw0.freq.f_star0` multiplied by ``f_star0_factor``,
        and $r_*$ is computed from $\tilde{\beta}$ in the same way as for the spectra,
        with :py:meth:`pttools.ssm.spectrum.SSMSpectrum.validate_nucleation`.
        The array covers all the combinations of the given values of $\tilde{\beta}$ or $r_*$, $T_*$ and $g_*$,
        with $g_{s,*} = g_*$.
        The returned array is logarithmically spaced from the smallest to the largest $z$,
        extended by ``margin`` decades at each end, as the spectra are inaccurate near the ends of their $z$ arrays.
        The values of $\tilde{\beta}$ and $r_*$ that are not finite and positive are ignored,
        as are those for which $r_*$ cannot be computed.

        :param f: frequencies $f$ to be covered
        :param bubble: fluid shell of the spectra
        :param T_star: $T_*$, temperatures at which the GWs were produced, a scalar or an array
        :param g_star: $g_*$, degrees of freedom, a scalar or an array
        :param beta_tilde: $\tilde{\beta}$ values of the spectra
        :param r_star: $r_*$ values of the spectra
        :param f_star0_factor: correction factor for $f_{\ast,0}$
        :param points_per_decade: number of $z$ points per decade.
            Defaults to :py:data:`Z_COVERING_DENSITY` times the average number of points per decade of ``f``.
        :param margin: number of decades by which the array is extended at each end,
            see :py:data:`Z_COVERING_MARGIN`
        :return: $z$ array
        :raises ValueError: if not exactly one of $\tilde{\beta}$ and $r_*$ is given,
            or if none of their values are valid
        """
        r_stars = [r for _value, r in Spectrum._valid_r_stars(bubble, beta_tilde=beta_tilde, r_star=r_star)]
        # f_*,0 is computed for all the combinations of T_* and g_*.
        T_stars = np.asarray(T_star, dtype=float).reshape(-1, 1)
        g_stars = np.asarray(g_star, dtype=float).reshape(1, -1)
        f_star0 = freq.f_star0(T_star=T_stars, ge_star=g_stars, gs_star=g_stars) * f_star0_factor
        z_min = freq.z(f=float(f.min()), r_star=min(r_stars), f_star0=float(np.max(f_star0))) / 10**margin
        z_max = freq.z(f=float(f.max()), r_star=max(r_stars), f_star0=float(np.min(f_star0))) * 10**margin
        if points_per_decade is None:
            points_per_decade = Z_COVERING_DENSITY * (f.size - 1) / np.log10(f.max() / f.min())
        n = max(int(np.ceil(points_per_decade * np.log10(z_max / z_min))) + 1, 2)
        z = np.logspace(np.log10(z_min), np.log10(z_max), n)
        # Ensure that the endpoints are exact despite the rounding errors of logspace.
        z[0] = z_min
        z[-1] = z_max
        return z

    @staticmethod
    def _valid_r_stars(
            bubble: Bubble,
            beta_tilde: FloatArr1D | None,
            r_star: FloatArr1D | None) -> list[tuple[float, float]]:
        r"""The valid values of $\tilde{\beta}$ or $r_*$, and the corresponding values of $r_*$.

        See :py:meth:`z_covering`.

        :raises ValueError: if not exactly one of $\tilde{\beta}$ and $r_*$ is given,
            or if none of their values are valid
        """
        if (beta_tilde is None) == (r_star is None):
            raise ValueError("Exactly one of beta_tilde and r_star must be given.")
        valid: list[tuple[float, float]] = []
        for value in np.atleast_1d(tp.cast(FloatArr1D, r_star if beta_tilde is None else beta_tilde)):
            if not (np.isfinite(value) and value > 0):
                continue
            try:
                # r_* does not depend on the nucleation type.
                valid.append((float(value), (
                    Spectrum.validate_nucleation(
                        bubble=bubble, nuc_type=DEFAULT_NUC_TYPE, r_star=float(value), beta_tilde=None)
                    if beta_tilde is None
                    else Spectrum.validate_nucleation(
                        bubble=bubble, nuc_type=DEFAULT_NUC_TYPE, r_star=None, beta_tilde=float(value))
                )))
            except ValueError:
                continue
        if not valid:
            raise ValueError("None of the beta_tilde or r_star values are valid.")
        return valid

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
            eb: bool = True,
            gb: bool = True,
            **kwargs: tp.Any) -> tuple[Figure, th.AxesArr1D]:
        """Plot the fluid velocity profile, the velocity and GW power spectra, and the GW power spectrum today.

        The plots are in a single row.

        :param fig: the figure to plot on
        :param path: if given, save the figure to this path
        :param label: label of the spectrum
        :param legend: whether to show the legend
        :param eb: whether to include the extragalactic compact binary noise in the noise and the SNR
        :param gb: whether to include the galactic compact binary noise in the noise and the SNR
        :param kwargs: additional arguments for :py:func:`pttools.plot.spectra.plot_spectra_multi_flat`
        :return: the figure and the 1D array of axes
        """
        from pttools.plot.spectra import plot_spectra_multi_flat  # noqa: PLC0415
        return plot_spectra_multi_flat(
            [self], fig=fig, path=path, labels=[label], legend=legend, eb=eb, gb=gb, **kwargs)


# These are object arrays. Numpy typing has no way of expressing the element type of an object array,
# but declaring the element type here does give the correct types when the arrays are indexed.
type SpectrumArr = NDArray[Spectrum]  # pyrefly: ignore[bad-specialization]
type SpectrumArr2D = np.ndarray[th.Shape2D, np.dtype[Spectrum]]  # pyrefly: ignore[bad-specialization]
type SpectrumArr3D = np.ndarray[th.Shape3D, np.dtype[Spectrum]]  # pyrefly: ignore[bad-specialization]
