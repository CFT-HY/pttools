"""Functions for computing GW power spectra."""

from __future__ import annotations

import logging
from math import sqrt
import os
import typing as tp
import uuid

import numpy as np

from pttools.bubble import Bubble, Phase, nu_gdh2024
from pttools.bubble.thermo import warn_if_ubarf_relativistic
from pttools.math.interpolation import interpolate_log_log
from pttools.speedup import NAN_ARR
from pttools.ssm.compute import SSMBaseResults, SSMGWResults, SSMInput, compute, compute_gw
from pttools.ssm.const import (
    BETA_TILDE_MIN,
    DEFAULT_A_STAR_A_R_RATIO,
    DEFAULT_EPS_LOOKUP,
    DEFAULT_LIFETIME_DISTRIBUTION_A,
    DEFAULT_N_SH,
    DEFAULT_N_T,
    DEFAULT_N_Z_LOOKUP,
    DEFAULT_R_STAR,
    DEFAULT_Y,
    T_TILDE_MAX,
    T_TILDE_MIN,
    Z_ST_THRESH,
)
from pttools.ssm.export import SSM_SPECTRUM_FIELDS
from pttools.ssm.nucleation import DEFAULT_NUC_TYPE, NucType
from pttools.ssm.nucleation import r_star as r_star_func
from pttools.ssm.spectrum_base_ssm import BaseSSMSpectrum
from pttools.ssm.ssm import DEFAULT_LAMBDA_CORRECTION, LambdaCorrection
from pttools.ssm.suppression import DEFAULT_SUPPRESSION, Suppression, SuppressionMethod
import pttools.type_hints as th
from pttools.type_hints import FloatArr1D
from pttools.utils.cache import Cacheable
from pttools.utils.fields import Fields, FieldSpec, Preset
from pttools.utils.json import export_json

if tp.TYPE_CHECKING:
    pass

if th.TYPE_CHECKING_OR_DOCS:
    pass

logger: logging.Logger = logging.getLogger(__name__)


class SSMSpectrum(Cacheable, BaseSSMSpectrum):
    """Gravitational wave power spectrum computed with the Sound Shell Model.

    The flat-space quantities of :py:class:`pttools.ssm.spectrum_base.BaseSpectrum`
    allow comparing the spectrum with external data, e.g. :py:class:`pttools.lattice.spectrum.LatticeSpectrum`.
    The spectra can be cached with :py:meth:`pttools.utils.cache.Cacheable.cached`.
    """

    #: The exportable fields of the spectrum
    FIELDS: tp.ClassVar[Fields] = SSM_SPECTRUM_FIELDS

    def __init__(  # noqa: PLR0915
            self,
            bubble: Bubble,
            # Input parameters
            beta_tilde: float | None = None,
            r_star: float | None = None,
            y: FloatArr1D = DEFAULT_Y,
            a_star_a_r_ratio: float = DEFAULT_A_STAR_A_R_RATIO,
            N_sh: float = DEFAULT_N_SH,
            nuc_type: NucType = DEFAULT_NUC_TYPE,
            omega_barotropic: float | None = None,
            # Suppression
            suppression: Suppression = DEFAULT_SUPPRESSION,
            suppression_method: SuppressionMethod = SuppressionMethod.DEFAULT,
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
        :param y: $z = k R_*$ array
        :param N_sh: $N_{\text{sh}}$, number of shock formation times
        :param nuc_type: nucleation type
        :param omega_barotropic: $\omega$, barotropic equation of state parameter.
            If not given, it's computed from Bubble.
        :param nT: number of points in the t array
        :param n_z_lookup: number of points in the lookup arrays
        :param z_st_thresh: for $z$ values above z_sh_tresh,
            use approximation rather than doing the sine transform integral.
        :param allow_beta_tilde_for_simultaneous:
            allow $\tilde{\beta}$ as an input parameter for simultaneous nucleation
        :param compute: whether to compute the spectrum immediately
        :param low_k: whether to use the :giombi_2024_cs: approximation for low $k$
        :param low_k_cs_friction: whether to compute the acoustic friction of the low-$k$ approximation
            from $c_s$ instead of $\nu$, see :py:func:`pttools.ssm.low_k.kernel.kernel_low`.
            This generalises the articles to $c_s^2 \neq \omega$, and is disabled by default.
            It should be considered an estimate, see the caveat in
            :py:func:`pttools.ssm.low_k.kernel.acoustic_friction_exponent`.
        :param parallel: whether to use multiple CPU cores
        """
        if np.isnan(y).any():
            raise ValueError("y must not contain nan values.")

        self.id: str = uuid.uuid4().hex

        # -----
        # Parameters
        # -----
        self._nu_gdh2024 = None if omega_barotropic is None else nu_gdh2024(omega_barotropic)
        self._omega_barotropic = omega_barotropic
        self._bubble: Bubble = bubble
        self.beta_tilde: float | None = beta_tilde
        self.a_star_a_r_ratio: float = a_star_a_r_ratio
        self.N_sh: float = N_sh
        self.nuc_type: NucType = nuc_type
        self.allow_beta_tilde_for_simultaneous: bool = allow_beta_tilde_for_simultaneous
        self.r_star: float = self.validate_nucleation(
            bubble=bubble, nuc_type=nuc_type, r_star=r_star, beta_tilde=beta_tilde,
            allow_beta_tilde_for_simultaneous=allow_beta_tilde_for_simultaneous
        )
        self.y: FloatArr1D = y
        # Suppression
        self.suppression: Suppression = suppression
        self.suppression_method: SuppressionMethod = suppression_method
        # Accuracy
        self.z_st_thresh: float = z_st_thresh
        self.nT: int = nT
        self.nx_P_tilde_gw: int | None = nx_P_tilde_gw
        self.n_z_lookup: int = n_z_lookup
        self.T_tilde_min: float = T_tilde_min
        self.T_tilde_max: float = T_tilde_max
        # Switches
        self.low_k: bool = low_k
        self.low_k_cs_friction: bool = low_k_cs_friction
        # Labels
        default_latex, default_unicode = self.default_labels(bubble, self.r_star)
        self.label_latex: str = default_latex if label_latex is None else label_latex
        self.label_unicode: str = default_unicode if label_unicode is None else label_unicode

        # -----
        # Values generated by compute()
        # -----
        self.a2: FloatArr1D = NAN_ARR
        self.a2_lookup: FloatArr1D = NAN_ARR
        self.bubble_spacing_enlargement_factor: float = np.nan
        self.cs: float = np.nan
        self.cs2: float = np.nan
        self.nucleation_f: float = np.nan
        self.hx: float = np.nan
        self.qT_lookup: FloatArr1D = NAN_ARR
        self.qT_gw_lookup: FloatArr1D = NAN_ARR
        self.spec_den_v: FloatArr1D = NAN_ARR
        self.spec_den_v_lookup: FloatArr1D = NAN_ARR
        self.spec_den_gw: FloatArr1D = NAN_ARR
        self.spec_den_gw_expanded: FloatArr1D = NAN_ARR
        self.spec_den_gw_ssm: FloatArr1D = NAN_ARR
        self.spec_den_gw_int: FloatArr1D = NAN_ARR
        self.spec_den_gw_low: FloatArr1D = NAN_ARR
        self.T_tilde: FloatArr1D = NAN_ARR
        self.ubarf2: float = np.nan
        self.z_lookup: FloatArr1D = NAN_ARR
        self.spec_den_gw_ssm_unscaled: FloatArr1D = NAN_ARR
        self.Iv: float = np.nan
        # Arguments of compute()
        #: Small offset used when generating the lookup table of $z$ values for the GW spectrum
        self.eps_lookup: float = DEFAULT_EPS_LOOKUP
        #: $a$, parameter of the bubble lifetime distribution
        self.lifetime_distribution_a: float = DEFAULT_LIFETIME_DISTRIBUTION_A
        #: Kinetic energy term of $\lambda$, see :py:class:`pttools.ssm.ssm.LambdaCorrection`
        self.lambda_correction: LambdaCorrection = DEFAULT_LAMBDA_CORRECTION
        #: Results of :py:func:`pttools.ssm.compute.compute_base`, which do not depend on $r_*$,
        #: and are therefore reused by :py:meth:`with_params`
        self.base_res: SSMBaseResults | None = None

        if compute:
            self.compute(parallel=parallel)

    @property
    @tp.override
    def bubble(self) -> Bubble:
        """The bubble of the spectrum."""
        return self._bubble

    def compute(
            self,
            eps_lookup: float = DEFAULT_EPS_LOOKUP,
            lifetime_distribution_a: float = DEFAULT_LIFETIME_DISTRIBUTION_A,
            lambda_correction: LambdaCorrection = DEFAULT_LAMBDA_CORRECTION,
            parallel: bool = True) -> None:
        r"""Compute the velocity and GW spectra and the related quantities, and store them as attributes.

        The bubble is solved first, if it has not been solved yet.

        :param eps_lookup: small offset used when generating the lookup table of $z$ values for the GW spectrum
        :param lifetime_distribution_a: $a$, parameter of the bubble lifetime distribution,
            see :py:func:`pttools.ssm.nucleation.lifetime_distribution`
        :param lambda_correction: kinetic energy term of $\lambda$, see :py:class:`pttools.ssm.ssm.LambdaCorrection`
        :param parallel: whether to use multiple CPU cores
        """
        self._prepare_compute(
            eps_lookup=eps_lookup, lifetime_distribution_a=lifetime_distribution_a, lambda_correction=lambda_correction)
        base_res, gw_res = compute(inp=self._ssm_input(), parallel=parallel)
        self._store_base(base_res)
        self._store_gw(gw_res)

    def _compute_gw(self) -> None:
        """Compute only the quantities that depend on $r_*$, reusing the results of the r_*-independent stage."""
        if self.base_res is None:
            raise RuntimeError("The quantities that do not depend on r_star have not been computed.")
        self._store_gw(compute_gw(inp=self._ssm_input(), base_res=self.base_res))

    def _prepare_compute(
            self,
            eps_lookup: float,
            lifetime_distribution_a: float,
            lambda_correction: LambdaCorrection) -> None:
        """Store the arguments of :py:meth:`compute`, and compute the input values that require the bubble."""
        if not self.bubble.solved:
            self.bubble.solve()
        self.eps_lookup = eps_lookup
        self.lifetime_distribution_a = lifetime_distribution_a
        self.lambda_correction = lambda_correction
        self.cs2 = self.bubble.model.cs2(self.bubble.va_enthalpy_density, Phase.BROKEN)
        self.cs = sqrt(self.cs2)

    def _ssm_input(self) -> SSMInput:
        """Input parameters for :py:mod:`pttools.ssm.compute`."""
        return SSMInput(
            # Arrays
            e=self.bubble.e,
            T_profile=self.bubble.T,
            v=self.bubble.v,
            w=self.bubble.w,
            xi=self.bubble.xi,
            y=self.y,
            # Scalars
            # The scalars are converted to float, so that the compiled computation is not compiled separately
            # for each combination of int, float and None arguments.
            beta_tilde=np.nan if self.beta_tilde is None else float(self.beta_tilde),
            cs=self.cs,
            lifetime_distribution_a=self.lifetime_distribution_a,
            nu_gdh2024=self.nu_gdh2024,
            source_lifetime_factor=self.source_lifetime_factor,
            tau_end=self.tau_end,
            tau_star=self.tau_star,
            v_sh=self.bubble.v_sh,
            v_wall=self.bubble.v_wall,
            # The average enthalpy density is reused if the bubble has already computed it,
            # and otherwise it's computed by the compiled computation.
            w_bar=self.bubble.__dict__.get("w_bar", np.nan),
            # Accuracy
            eps_lookup=self.eps_lookup,
            nT=self.nT,
            n_z_lookup=self.n_z_lookup,
            # None is converted to 0, which means the size of the lookup array in spec_den_gw(),
            # as the compiled computation would otherwise be compiled separately for None.
            nx_P_tilde_gw=0 if self.nx_P_tilde_gw is None else self.nx_P_tilde_gw,
            T_tilde_min=self.T_tilde_min,
            T_tilde_max=self.T_tilde_max,
            z_st_thresh=self.z_st_thresh,
            # Other
            nuc_type=self.nuc_type,
            sol_type=self.bubble.sol_type,
            lambda_correction=self.lambda_correction,
            low_k_cs_friction=self.low_k_cs_friction
        )

    def _store_gw(self, gw_res: SSMGWResults) -> None:
        """Store the results of :py:func:`pttools.ssm.compute.compute_gw` as attributes."""
        self.spec_den_gw_ssm = gw_res.spec_den_gw_ssm
        self.spec_den_gw_low = gw_res.spec_den_gw_low
        self.spec_den_gw_int = gw_res.spec_den_gw_int
        self.spec_den_gw_expanded = gw_res.spec_den_gw_expanded
        self.nucleation_f = gw_res.nucleation_f
        self.hx = gw_res.hx
        self.bubble_spacing_enlargement_factor = gw_res.bubble_spacing_enlargement_factor
        self.spec_den_gw = self.spec_den_gw_expanded if self.low_k else self.spec_den_gw_ssm

    def _store_base(self, base_res: SSMBaseResults) -> None:
        """Store the results of :py:func:`pttools.ssm.compute.compute_base` as attributes."""
        self.base_res = base_res
        self.a2 = base_res.A2
        self.a2_lookup = base_res.A2_lookup
        self.qT_lookup = base_res.qT_lookup
        self.qT_gw_lookup = base_res.qT_gw_lookup
        self.spec_den_v = base_res.spec_den_v
        self.spec_den_v_lookup = base_res.spec_den_v_lookup
        self.T_tilde = base_res.T_tilde
        self.ubarf2 = base_res.ubarf2
        self.z_lookup = base_res.z_lookup
        self.spec_den_gw_ssm_unscaled = base_res.spec_den_gw_ssm_unscaled
        self.Iv = base_res.Iv
        warn_if_ubarf_relativistic(self.ubarf2, source=self.label_unicode)

    def export(
            self,
            path: str | os.PathLike[str] | None = None,
            fields: FieldSpec = Preset.FULL,
            bubble_fields: FieldSpec = Preset.FULL,
            model_fields: FieldSpec = Preset.FULL) -> dict[str, tp.Any]:
        """Export the spectrum data to a dictionary, and optionally save it as a JSON file.

        :param path: path of the JSON file
        :param fields: the fields of the spectrum to export, see :py:data:`pttools.utils.fields.FieldSpec`
        :param bubble_fields: the fields of the bubble to export
        :param model_fields: the fields of the model to export
        :return: the exported data, where the bubble data is under the key ``"bubble"``
        """
        data = {
            "bubble": self.bubble.export(fields=bubble_fields, model_fields=model_fields),
            **self.extract(fields)
        }
        if path is not None:
            export_json(data, path)
        return data

    def with_params(
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
            y: FloatArr1D | None = None) -> tp.Self:
        r"""Create a new spectrum with different parameters, reusing the computations of this spectrum.

        The velocity spectra, ${\lvert A \rvert}^2$, $\bar{U}_f^2$,
        the GW spectral density without the source lifetime factor $\tilde{P}_\text{gw,ssm} / \Upsilon_\ell$
        and the source integral $\mathcal{I}_v$ do not depend on the parameters given here,
        and therefore the computationally heavy :py:func:`pttools.ssm.compute.compute_base` is not run again.
        Only the computationally light :py:func:`pttools.ssm.compute.compute_gw` is run,
        which applies the source lifetime factor and computes the low-$k$ approximation.
        The result is the same as when creating a new spectrum with the same arguments as this one,
        except for the changed parameters.

        The parameters that are not given are taken from this spectrum.
        If either $r_*$ or $\tilde{\beta}$ is given, the other one is not taken from this spectrum.
        The default labels are generated for the new spectrum unless labels are given,
        as they contain $r_*$.

        If a new $y$ array is given, it must be within the range of the $y$ array of this spectrum,
        see :py:meth:`covers`.
        Then the velocity spectrum $\tilde{P}_v$ and the GW spectral density without the source lifetime factor
        $\tilde{P}_\text{gw,ssm} / \Upsilon_\ell$ are interpolated to it linearly in log-log scale
        with :py:func:`pttools.math.interpolation.interpolate_log_log`,
        and the rest of the spectrum is computed for it by :py:func:`pttools.ssm.compute.compute_gw`,
        including the low-$k$ approximation.
        As the power spectra are $\mathcal{P} \propto z^3 \tilde{P}$,
        this is the same as interpolating the power spectra in log-log scale.
        The lookup arrays, ${\lvert A \rvert}^2$, $\bar{U}_f^2$ and $\mathcal{I}_v$ do not depend on the $y$ array,
        and are therefore the same as those of this spectrum.
        The result therefore differs from a spectrum computed for the new $y$ array by the interpolation error,
        and by the dependence of the computation on the range of the $y$ array,
        as the lookup arrays span the range of the $y$ array.
        As for any $y$ array, the junction of the low-$k$ approximation and $\tilde{P}_\text{gw,ssm}$
        depends on the points of the $y$ array, see :py:func:`pttools.ssm.low_k.join.gw_junction`.
        Therefore, the result differs from this spectrum near the junction even at the points of this spectrum.
        This is useful for computing the spectra with different parameters for the same frequencies,
        as their $y$ arrays then differ, see :py:meth:`pttools.omgw0.spectrum.Spectrum.with_params`.

        :param r_star: $r_*$, Hubble-scaled mean bubble spacing
        :param beta_tilde: $\tilde{\beta} \equiv \frac{\beta}{H_*}$, nucleation rate parameter
        :param a_star_a_r_ratio: $\frac{a_*}{a_r}$
        :param N_sh: $N_\text{sh}$, number of shock formation times
        :param omega_barotropic: $\omega$, barotropic equation of state parameter
        :param suppression: suppression data
        :param suppression_method: suppression method
        :param label_latex: label of the new spectrum in LaTeX format
        :param label_unicode: label of the new spectrum in Unicode format
        :param y: $z = kR_*$ array of the new spectrum. If not given, the $y$ array of this spectrum is used.
        :return: the new spectrum
        :raises RuntimeError: if this spectrum has not been computed
        :raises ValueError: if the new $y$ array is not within the range of the $y$ array of this spectrum
        """
        return self._with_params(
            {
                "r_star": r_star, "beta_tilde": beta_tilde,
                "a_star_a_r_ratio": a_star_a_r_ratio, "N_sh": N_sh, "omega_barotropic": omega_barotropic,
                "suppression": suppression, "suppression_method": suppression_method,
                "label_latex": label_latex, "label_unicode": label_unicode, "y": y
            }
        )

    def _init_kwargs(self) -> dict[str, tp.Any]:
        """Arguments of the constructor for creating a copy of this spectrum.

        The labels are not included, as the default labels depend on $r_*$.
        """
        return {
            "bubble": self.bubble,
            "beta_tilde": self.beta_tilde,
            "r_star": self.r_star if self.beta_tilde is None else None,
            "y": self.y,
            "a_star_a_r_ratio": self.a_star_a_r_ratio,
            "N_sh": self.N_sh,
            "nuc_type": self.nuc_type,
            "omega_barotropic": self._omega_barotropic,
            "suppression": self.suppression,
            "suppression_method": self.suppression_method,
            "nT": self.nT,
            "nx_P_tilde_gw": self.nx_P_tilde_gw,
            "n_z_lookup": self.n_z_lookup,
            "T_tilde_min": self.T_tilde_min,
            "T_tilde_max": self.T_tilde_max,
            "z_st_thresh": self.z_st_thresh,
            "low_k": self.low_k,
            "low_k_cs_friction": self.low_k_cs_friction,
            "allow_beta_tilde_for_simultaneous": self.allow_beta_tilde_for_simultaneous,
        }

    def _with_params(self, params: dict[str, tp.Any]) -> tp.Self:
        """Implementation of :py:meth:`with_params`, where the parameters that are ``None`` are not changed."""
        if self.base_res is None:
            raise RuntimeError("The spectrum has to be computed before it can be reused for a new spectrum.")
        kwargs = self._init_kwargs()
        if params.get("r_star") is not None or params.get("beta_tilde") is not None:
            kwargs["r_star"] = kwargs["beta_tilde"] = None
        # The frequencies of pttools.omgw0.spectrum.Spectrum replace the y array.
        if params.get("f") is not None:
            kwargs["y"] = None
        kwargs.update({key: value for key, value in params.items() if value is not None})
        spectrum = type(self)(**kwargs, compute=False)
        if not self.covers(spectrum.y):
            raise ValueError(
                f"The new spectrum has y = {spectrum.y.min():.3e} - {spectrum.y.max():.3e}, "
                f"which is not within the range y = {self.y.min():.3e} - {self.y.max():.3e} of this spectrum."
            )
        spectrum._reuse(self)  # noqa: SLF001
        return spectrum

    def _reuse(self, other: SSMSpectrum) -> None:
        """Compute this spectrum by reusing the r_*-independent results of another spectrum.

        If the $y$ arrays differ, the results along $y$ are interpolated to the $y$ array of this spectrum,
        see :py:meth:`with_params`.
        """
        if not self.bubble.solved:
            self.bubble.solve()
        self.eps_lookup = other.eps_lookup
        self.lifetime_distribution_a = other.lifetime_distribution_a
        self.lambda_correction = other.lambda_correction
        self.cs = other.cs
        self.cs2 = other.cs2
        base_res = tp.cast(SSMBaseResults, other.base_res)
        if not np.array_equal(self.y, other.y):
            base_res = base_res._replace(
                spec_den_v=interpolate_log_log(self.y, other.y, base_res.spec_den_v),
                spec_den_gw_ssm_unscaled=interpolate_log_log(self.y, other.y, base_res.spec_den_gw_ssm_unscaled)
            )
        self._store_base(base_res)
        self._compute_gw()

    @staticmethod
    def validate_nucleation(
            bubble: Bubble,
            nuc_type: NucType,
            r_star: float | None,
            beta_tilde: float | None = None,
            allow_beta_tilde_for_simultaneous: bool = False,
            warn_on_small_beta_tilde: bool = True) -> float:
        r"""Validate the nucleation parameters and determine $r_*$.

        Only one of $r_*$ and $\tilde{\beta}$ can be given.
        If neither is given, the default $r_*$ of :py:mod:`pttools.ssm.const` is used.
        If $\tilde{\beta}$ is given, $r_*$ is computed from it with :py:func:`pttools.ssm.nucleation.r_star`.
        Warnings are logged for $\tilde{\beta}$ values that are experimentally excluded and for $r_* \geq 1$.

        :param bubble: the bubble, which is solved if $\tilde{\beta}$ is given and it has not been solved yet
        :param nuc_type: nucleation type
        :param r_star: $r_*$, Hubble-scaled mean bubble spacing
        :param beta_tilde: $\tilde{\beta} \equiv \frac{\beta}{H_*}$, nucleation rate parameter
        :param allow_beta_tilde_for_simultaneous:
            allow $\tilde{\beta}$ as an input parameter for simultaneous nucleation
        :param warn_on_small_beta_tilde: warn if $\tilde{\beta}$ is experimentally excluded
        :return: $r_*$, Hubble-scaled mean bubble spacing
        :raises ValueError: if both $r_*$ and $\tilde{\beta}$ are given, if either of them is not positive,
            or if $\tilde{\beta}$ is given for simultaneous nucleation without ``allow_beta_tilde_for_simultaneous``
        """
        r_star_set: float
        if beta_tilde is None:
            r_star_set = DEFAULT_R_STAR if r_star is None else r_star
        # If beta_tilde is set, use it to obtain r_star.
        else:
            if r_star is not None:
                raise ValueError(
                    "Either beta_tilde or r_star must be provided, but not both. "
                    f"Got beta_tilde={beta_tilde}, r_star={r_star}."
                )
            if (not allow_beta_tilde_for_simultaneous) and nuc_type == NucType.SIMULTANEOUS:
                raise ValueError(
                    "beta_tilde -> r_star conversion is not defined for simultaneous nucleation. "
                    f"Got: nuc_type={nuc_type}, r_star={r_star}, beta_tilde={beta_tilde}"
                )
            if np.isnan(beta_tilde) or beta_tilde <= 0:
                raise ValueError(f"beta_tilde must be positive. Got beta_tilde={beta_tilde}.")
            if warn_on_small_beta_tilde and beta_tilde < BETA_TILDE_MIN:
                logger.warning(
                    "Got beta_tilde=%s < %s. "
                    "This is experimentally excluded due to primordial black hole formation. "
                    "Please see Lewicki et al. (2023) for details.",
                    beta_tilde, BETA_TILDE_MIN
                )
            if not bubble.solved:
                bubble.solve()
            r_star_set = r_star_func(
                beta_tilde=beta_tilde, v_wall=bubble.v_wall, xi=bubble.xi, T=bubble.T, sol_type=bubble.sol_type
            )

        if np.isnan(r_star_set) or r_star_set <= 0:
            raise ValueError(f"r_star must be positive. Got r_star={r_star_set}.")
        return r_star_set
