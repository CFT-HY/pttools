"""Gravitational wave power spectrum as observed today."""

import functools
import math
import os
import typing as tp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from pttools.bubble import Bubble
from pttools.omgw0 import const, freq
from pttools.omgw0.compute import OmGw0Input, OmGw0Results, SNRSettings, compute, compute_gw
from pttools.omgw0.const import H2, LISA_OBS_TIME, OMEGA_PHOTON_H2
from pttools.omgw0.export import SPECTRUM_FIELDS, SPECTRUM_FIELDS_F, SPECTRUM_FIELDS_Y
from pttools.omgw0.factors import F_gw0_h2
from pttools.omgw0.noise import omega_ins_h2, omega_noise_h2, signal_to_noise_ratio
from pttools.ssm.calculators import trapezoid_loglog
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
from pttools.ssm.suppression import DEFAULT_SUPPRESSION, Suppression, SuppressionMethod
import pttools.type_hints as th
from pttools.type_hints import FloatArr1D, FloatOrArr
from pttools.utils.docstrings import copy_docstring_dec
from pttools.utils.fields import Fields

if tp.TYPE_CHECKING:
    from pttools.analysis.utils import FigAndAxes


class Spectrum(SSMSpectrum):
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

    @functools.cached_property
    def ge_star(self) -> float:
        r"""$g_{e,*}$, degrees of freedom for energy density at the time the GWs were produced.

        $$g_{e,*} = \frac{1}{3}(4 g_s - g_p)$$
        :maki_msc:`\ ` eq. 2.108.
        """
        return (4 * self.gs_star - self.g_star) / 3

    @functools.cached_property
    def e_star(self) -> float:
        r"""$e_*$, energy density at GW formation.

        $$e_* = \frac{\pi^2}{30} g_e(T_*) T_*^4$$
        :maki_msc:`\ ` eq. 2.105
        This presumes that $V(T_*, \phi_b) = 0$.
        """
        return np.pi**2 / 30 * self.ge_star * self.T_star ** 4

    @property
    def f_given(self) -> bool:
        """Whether the frequencies $f$ were given as an argument instead of the $y$ array."""
        return self._f_given is not None

    @functools.cached_property
    def _f_y(self) -> th.FloatArr1D:
        r"""Frequencies today $f(y)$ corresponding to the $y$ array of the spectrum.

        This is cached, so that :py:meth:`f`, :py:attr:`f_min` and :py:attr:`f_max` compute the frequencies only once.
        If the frequencies were given as an argument, they are returned as is.
        """
        if self._f_given is not None:
            return self._f_given
        return freq.f(z=self.y, r_star=self.r_star, f_star0=self.f_star0)

    @functools.cached_property
    def f_max(self) -> float:
        r"""$f_\text{max}$, maximum frequency today corresponding to the $y$ array of the spectrum."""
        return float(self._f_y.max())

    @functools.cached_property
    def f_min(self) -> float:
        r"""$f_\text{min}$, minimum frequency today corresponding to the $y$ array of the spectrum."""
        return float(self._f_y.min())

    @functools.cached_property
    @copy_docstring_dec(freq.f_star0, without_params=True)
    def f_star0(self) -> float:
        return freq.f_star0(
            T_star=self.T_star,
            ge_star=self.ge_star,
            gs_star=self.gs_star
        )

    @functools.cached_property
    def H_star(self) -> float:
        r"""Hubble rate $H_*$ at GW formation, in units of $T^2$.

        $$H = \sqrt{8 \pi \frac{e_*}{3}} \frac{1}{m_{\text{pl}}}$$
        This is a direct consequence of the Friedmann equation
        $$H^2 + \frac{K}{a^2} = \frac{8 \pi G}{3} e$$
        with $K = 0$ and $m_{\text{pl}} = \frac{1}{G}$.
        Note that here $m_pl = 1$ in natural units.
        """
        return math.sqrt(8 * math.pi * self.e_star / 3)

    @functools.cached_property
    def R_star(self) -> th.FloatOrArr:
        r"""Mean bubble separation $R_*$, in units of $T^{-2}$.

        $$R_* = \frac{r_*}{H_*}$$
        :gowling_2021:`\ ` eq. 2.2
        """
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
            lambda_correction: bool = False,
            parallel: bool = True,
            compute_snr: bool | None = None,
            snr_settings: SNRSettings | None = None) -> None:
        r"""Compute the spectra and optionally the signal-to-noise ratio, and store them as attributes.

        The spectra and the signal-to-noise ratio are computed in one Numba-compiled function
        :py:func:`pttools.omgw0.compute.compute`, so that the GIL is not acquired in between.

        :param eps_lookup: small offset used when generating the lookup table of $z$ values for the GW spectrum
        :param lifetime_distribution_a: $a$, parameter of the bubble lifetime distribution,
            see :py:func:`pttools.ssm.nucleation.lifetime_distribution`
        :param lambda_correction: whether to enable a non-linear correction for $\lambda$,
            see :py:func:`pttools.ssm.ssm.A2_e_conserving`
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
            mean_adiabatic_index=self.bubble.mean_adiabatic_index,
            r_star=self.r_star,
            nu_gdh2024=self.nu_gdh2024,
            dilution_of_e=self.dilution_of_e,
            suppression_factor=self.suppression_factor,
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
            compute_snr: bool | None = None) -> tp.Self:
        r"""Create a new spectrum with different parameters, reusing the computations of this spectrum.

        See :py:meth:`pttools.ssm.spectrum.SSMSpectrum.with_params`.
        The new spectrum has the same $z$ array as this spectrum.
        Therefore, if the frequencies $f$ were given to this spectrum,
        the frequencies of the new spectrum differ from them when $r_*$, $T_*$, $g_*$ or $g_{s,*}$ is changed.
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
        :return: the new spectrum
        :raises RuntimeError: if this spectrum has not been computed
        """
        if g_star is not None and gs_star is None:
            gs_star = g_star
        return self._with_params(
            {
                "r_star": r_star, "beta_tilde": beta_tilde,
                "a_star_a_r_ratio": a_star_a_r_ratio, "N_sh": N_sh, "omega_barotropic": omega_barotropic,
                "suppression": suppression, "suppression_method": suppression_method,
                "label_latex": label_latex, "label_unicode": label_unicode,
                "T_star": T_star, "g_star": g_star, "gs_star": gs_star, "compute_snr": compute_snr
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

    @copy_docstring_dec(freq.f, without_params=True)
    def f(self, z: th.FloatArr1D | None = None) -> th.FloatArr1D:
        # The frequencies of the y array are cached, and the frequencies of a custom z are not.
        if z is None:
            return self._f_y
        return freq.f(z=z, r_star=self.r_star, f_star0=self.f_star0)

    def F_gw0[T: FloatOrArr](
            self,
            ge0_photon: T = const.GE0_PHOTON,
            gs0: th.FloatOrArr = const.GS0,
            h2: th.FloatOrArr = H2) -> T:
        r"""$F_{\text{gw},0}$, power attenuation following the end of the radiation era."""
        return self.F_gw0_h2(ge0_photon=ge0_photon, gs0=gs0) / h2  # pyrefly: ignore[bad-return]

    @copy_docstring_dec(F_gw0_h2, without_params=True)
    def F_gw0_h2[T: FloatOrArr](
            self,
            ge0_photon: T = const.GE0_PHOTON,
            gs0: th.FloatOrArr = const.GS0,
            om_gamma0_h2: th.FloatOrArr = OMEGA_PHOTON_H2) -> T:
        return F_gw0_h2(  # pyrefly: ignore[bad-return]
            ge_star=self.ge_star,
            ge0_photon=ge0_photon,
            gs0=gs0,
            gs_star=self.gs_star,
            om_gamma0_h2=om_gamma0_h2
        )

    def noise(self, eb: bool = True, gb: bool = True, ins: bool = True, h2: th.FloatOrArr1D = H2) -> th.FloatArr1D:
        r"""Total LISA noise $\Omega_\text{noise}$."""
        return self.noise_h2(eb=eb, gb=gb, ins=ins) / h2

    @copy_docstring_dec(omega_noise_h2, without_params=True)
    def noise_h2(self, eb: bool = True, gb: bool = True, ins: bool = True) -> th.FloatArr1D:
        return omega_noise_h2(f=self.f(), eb=eb, gb=gb, ins=ins)

    def noise_ins(self, h2: th.FloatOrArr1D) -> th.FloatArr1D:
        r"""LISA instrument noise $\Omega_\text{ins}$."""
        return self.noise_ins_h2() / h2

    @copy_docstring_dec(omega_ins_h2, without_params=True)
    def noise_ins_h2(self) -> th.FloatArr1D:
        return omega_ins_h2(f=self.f())

    def omgw0(
            self,
            ge0_photon: float = const.GE0_PHOTON,
            gs0: float = const.GS0,
            h2: th.FloatOrArr1D = H2) -> th.FloatArr1D:
        r"""Gravitational wave power spectrum today $\Omega_{\text{gw},0}$.

        :param ge0_photon: $g_{e\gamma 0}$, degrees of freedom for energy density of photons today
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param h2: $h^2$, dimensionless reduced Hubble constant squared
        """
        return self.omgw0_h2(ge0_photon=ge0_photon, gs0=gs0) / h2

    def omgw0_h2(
            self,
            ge0_photon: float = const.GE0_PHOTON,
            gs0: float = const.GS0,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2) -> th.FloatArr1D:
        r"""Gravitational wave power spectrum today $\Omega_{\text{gw},0} h^2$.

        :param ge0_photon: $g_{e\gamma 0}$, degrees of freedom for energy density of photons today
        :param gs0: $g_{s,0}$, degrees of freedom for entropy today
        :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
        """
        return self.F_gw0_h2(ge0_photon=ge0_photon, gs0=gs0, om_gamma0_h2=om_gamma0_h2) * self.pow_gw

    def omgw0_h2_peak(
            self,
            ge0_photon: float = const.GE0_PHOTON,
            gs0: float = const.GS0,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2) -> tuple[float, float]:
        r"""Peak $\Omega_{\text{gw},0} h^2$.

        :param ge0_photon: $g_{e\gamma 0}$, degrees of freedom for energy density of photons today
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
        """
        omgw0_h2 = self.omgw0_h2(ge0_photon=ge0_photon, gs0=gs0, om_gamma0_h2=om_gamma0_h2)
        i_max = np.argmax(omgw0_h2)
        return self.f()[i_max], omgw0_h2[i_max]

    def omgw0_h2_total(self, omgw0_h2: th.FloatArr1D | None = None) -> float:
        r"""Total $\Omega_{\text{gw},0} h^2$ integrated over all frequencies.

        :param omgw0_h2: $\Omega_{\text{gw},0} h^2$
        """
        return trapezoid_loglog(
            x=self.f(),
            # This doesn't take arguments, as the entire omgw0_h2 can be overridden with an argument.
            y=self.omgw0_h2() if omgw0_h2 is None else omgw0_h2
        )

    def omgw0_peak[T: FloatOrArr](
            self,
            ge0_photon: float = const.GE0_PHOTON,
            gs0: float = const.GS0,
            h2: T = H2,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2) -> tuple[float, T]:
        r"""Peak $\Omega_{\text{gw},0}$.

        :param ge0_photon: $g_{e\gamma 0}$, degrees of freedom for energy density of photons today
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param h2: $h^2$, dimensionless reduced Hubble constant squared
        :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
        """
        f_peak, omgw0_h2_peak = self.omgw0_h2_peak(ge0_photon=ge0_photon, gs0=gs0, om_gamma0_h2=om_gamma0_h2)
        return f_peak, tp.cast(T, omgw0_h2_peak / h2)

    def omgw0_total[T: FloatOrArr](
            self,
            omgw0_h2: th.FloatArr1D | None = None,
            h2: T = H2) -> T:
        r"""Total $\Omega_{\text{gw},0}$ integrated over all frequencies.

        :param omgw0_h2: $\Omega_{\text{gw},0} h^2$
        :param h2: $h^2$, dimensionless reduced Hubble constant squared
        """
        return tp.cast(T, self.omgw0_h2_total(omgw0_h2=omgw0_h2) / h2)

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
        r"""Signal-to-noise ratio for LISA, taking into account all noise sources.

        :param noise: $\Omega_\text{noise} h^2$
        :param f_noise: frequencies for the noise (assumed to be the same as for the signal, if not provided)
        :param ge0_photon: $g_{e\gamma 0}$, degrees of freedom for energy density of photons today
        :param gs0: $g_{s,0}$, degrees of freedom today for entropy
        :param obs_time: observation time (s)
        :param om_gamma0_h2: $\Omega_{\gamma,0} h^2$, the photon density parameter today, multiplied by $h^2$
        :param noise_eb: whether to generate extragalactic compact binary noise when noise is not provided
        :param noise_gb: whether to generate galactic compact binary noise when noise is not provided
        :param noise_ins: whether to generate instrument noise when noise is not provided
        :return: the signal-to-noise ratio, the frequencies of the signal, $\Omega_{\text{gw},0} h^2$,
            the frequencies of the noise and $\Omega_\text{noise} h^2$.
            If the signal-to-noise ratio was computed together with the spectrum with the same settings
            (see :py:meth:`compute`), the precomputed result is returned.
        """
        f: FloatArr1D = self.f()
        settings = SNRSettings(
            noise=noise, f_noise=f_noise, obs_time=obs_time,
            ge0_photon=ge0_photon, gs0=gs0, om_gamma0_h2=om_gamma0_h2,
            noise_eb=noise_eb, noise_gb=noise_gb, noise_ins=noise_ins
        )
        res = self._omgw0_res
        if res is not None and settings.matches(self.snr_settings):
            return res.snr, f, res.omgw0_h2, res.f_noise, res.noise
        omgw0_h2 = self.omgw0_h2(ge0_photon=ge0_photon, gs0=gs0, om_gamma0_h2=om_gamma0_h2)
        snr, f_noise, noise = signal_to_noise_ratio(
            f=f, signal=omgw0_h2, obs_time=obs_time,
            noise=noise, f_noise=f_noise,
            noise_eb=noise_eb, noise_gb=noise_gb, noise_ins=noise_ins
        )
        return snr, f, omgw0_h2, f_noise, noise

    def snr_ins(
            self,
            ge0_photon: float = const.GE0_PHOTON,
            gs0: float = const.GS0,
            obs_time: float = LISA_OBS_TIME,
            om_gamma0_h2: float = const.OMEGA_PHOTON_H2
        ) -> tuple[float, FloatArr1D, FloatArr1D, FloatArr1D, FloatArr1D]:
        """Signal-to-noise ratio for LISA, taking into account only the instrument noise."""
        return self.snr(
            ge0_photon=ge0_photon, gs0=gs0, obs_time=obs_time, om_gamma0_h2=om_gamma0_h2,
            noise_eb=False, noise_gb=False, noise_ins=True
        )

    @copy_docstring_dec(freq.z, without_params=True)
    def z_from_f[T: FloatOrArr](self, f: T) -> T:
        return freq.z(f=f, r_star=self.r_star, f_star0=self.f_star0)

    # -----
    # Plotting
    # -----

    def plot(
            self,
            fig: Figure | None = None,
            ax: Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            **kwargs: tp.Any) -> "FigAndAxes":
        r"""Plot the GW power spectrum today $\Omega_{\text{gw},0} h^2(f)$ together with the LISA noise.

        :param fig: the figure to plot on
        :param ax: the axes to plot on
        :param path: if given, save the figure to this path
        :param kwargs: additional arguments for :py:func:`pttools.analysis.plot_spectra.plot_spectra`
        :return: the figure and the axes
        """
        from pttools.analysis.plot_spectra import plot_spectra  # noqa: PLC0415
        return plot_spectra([self], fig, ax, path, **kwargs)

    def plot_multi(
            self,
            fig: Figure | None = None,
            path: str | os.PathLike[str] | None = None,
            **kwargs: tp.Any) -> tuple[Figure, th.AxesArr2D]:
        """Plot the fluid velocity profile, the velocity and GW power spectra, and the GW power spectrum today.

        The plots are in a 2x2 grid.

        :param fig: the figure to plot on
        :param path: if given, save the figure to this path
        :param kwargs: additional arguments for :py:func:`pttools.analysis.plot_spectra.plot_spectra_multi`
        :return: the figure and the 2D array of axes
        """
        from pttools.analysis.plot_spectra import plot_spectra_multi  # noqa: PLC0415
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
        :param kwargs: additional arguments for :py:func:`pttools.analysis.plot_spectra.plot_spectra_multi_flat`
        :return: the figure and the 1D array of axes
        """
        from pttools.analysis.plot_spectra import plot_spectra_multi_flat  # noqa: PLC0415
        return plot_spectra_multi_flat([self], fig=fig, path=path, labels=[label], legend=legend, **kwargs)


# These are object arrays. Numpy typing has no way of expressing the element type of an object array,
# but declaring the element type here does give the correct types when the arrays are indexed.
type SpectrumArr = NDArray[Spectrum]  # pyrefly: ignore[bad-specialization]
type SpectrumArr2D = np.ndarray[tuple[int, int], np.dtype[Spectrum]]  # pyrefly: ignore[bad-specialization]
type SpectrumArr3D = np.ndarray[tuple[int, int, int], np.dtype[Spectrum]]  # pyrefly: ignore[bad-specialization]
