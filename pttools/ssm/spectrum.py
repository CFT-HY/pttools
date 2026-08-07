"""Functions for computing GW power spectra"""

import functools
import logging
import typing as tp

import matplotlib.pyplot as plt

from math import sqrt
import numpy as np

from pttools.bubble import Bubble, Phase
from pttools.ssm import const
from pttools.ssm.barotropic import dilution_of_e, eta_ratio, H_eta, source_lifetime_factor
from pttools.ssm.compute import compute
from pttools.ssm.nucleation import \
    DEFAULT_NUC_TYPE, NucType, beta, bubble_spacing_enlargement_factor, hx, nucleation_f, r_star as r_star_func
from pttools.ssm.pow_spec import pow_spec
from pttools.ssm.scaling import H_star_tau_sh, H_star_tau_v, H_star_tau_v_old, J
from pttools.ssm.spec_den_gw import spec_den_gw_scaling
from pttools.ssm.low_k.intersection import z_cross_approx
from pttools.ssm.ssm import ubarf2_from_a2
from pttools.ssm.suppression import DEFAULT_SUPPRESSION, Suppression, SuppressionMethod
from pttools.type_hints import FloatArr, FloatArr1D
from pttools.utils import copy_docstrings, export_json

if tp.TYPE_CHECKING:
    from pttools.analysis.utils import FigAndAxes

logger = logging.getLogger(__name__)


class SSMSpectrum:
    """Gravitational wave simulation object"""
    def __init__(
            self,
            bubble: Bubble,
            # Input parameters
            beta_tilde: float | None = None,
            r_star: float | None = None,
            y: FloatArr1D | None = None,
            a_star_a_r_ratio: float = const.DEFAULT_A_STAR_A_R_RATIO,
            N_sh: float = const.DEFAULT_N_SH,
            nuc_type: NucType = DEFAULT_NUC_TYPE,
            # Suppression
            suppression: Suppression = DEFAULT_SUPPRESSION,
            suppression_method: SuppressionMethod = SuppressionMethod.DEFAULT,
            # Accuracy settings
            nT: int = const.DEFAULT_N_T,
            n_z_lookup: int = const.DEFAULT_N_Z_LOOKUP,
            T_tilde_min: float = const.T_TILDE_MIN,
            T_tilde_max: float = const.T_TILDE_MAX,
            z_st_thresh: float = const.Z_ST_THRESH,
            # Switches
            compute: bool = True,
            low_k: bool = True,
            parallel: bool = True,
            # Labels
            label_latex: str | None = None,
            label_unicode: str | None = None):
        r"""
        :param bubble: the Bubble object
        :param beta_tilde: nucleation rate parameter $\tilde{\beta} \equiv \frac{\beta}{H_*}$
        :param r_star: Hubble-scaled mean bubble spacing $r_*$
        :param y: $z = k R_*$ array
        :param N_sh: $N_\text{sh}$, number of shock formation times
        :param nuc_type: nucleation type
        :param nT: number of points in the t array
        :param n_z_lookup: number of points in the lookup arrays
        :param z_st_thresh: for $z$ values above z_sh_tresh,
            use approximation rather than doing the sine transform integral.
        :param compute: whether to compute the spectrum immediately
        :param low_k: whether to use the :giombi_2024_cs: approximation for low $k$
        :param parallel: whether to use multiple CPU cores
        """
        if y is None:
            self.y = const.DEFAULT_Y
        elif np.isnan(y).any():
            raise ValueError("y must not contain nan values.")
        else:
            self.y = y

        # -----
        # r_star
        # -----
        # Todo: Move this to a separate function
        self.r_star: float
        if beta_tilde is None:
            if r_star is None:
                self.r_star = const.DEFAULT_R_STAR
            elif np.isnan(r_star):
                raise ValueError(f"Got invalid r_star={r_star}")
            else:
                self.r_star = r_star
        else:
            if r_star is not None:
                raise ValueError(
                    "Either beta_tilde or r_star must be provided, but not both. "
                    f"Got beta_tilde={beta_tilde}, r_star={r_star}."
                )
            if np.isnan(beta_tilde) or beta_tilde <= 0:
                raise ValueError(f"beta_tilde must be positive. Got beta_tilde={beta_tilde}.")
            if beta_tilde < const.BETA_TILDE_MIN:
                logger.warning(
                    "Got beta_tilde=%s < %s. "
                    "This is experimentally excluded due to primordial black hole formation. "
                    "Please see Lewicki et al. (2023) for details.",
                    beta_tilde, const.BETA_TILDE_MIN
                )
            if not bubble.solved:
                bubble.solve()
            self.r_star = r_star_func(
                beta_over_H=beta_tilde, v_wall=bubble.v_wall, xi=bubble.xi, T=bubble.T, sol_type=bubble.sol_type
            )

        if np.isnan(self.r_star) or self.r_star <= 0:
            raise ValueError(f"r_star must be positive. Got r_star={self.r_star}.")
        elif self.r_star >= 1:
            # Todo: Find a better reference for this.
            logger.warning(
                "r_star < 1 is required for the phase transition to complete. "
                "Got r_star=%s. See Hindmarsh & Hijazi 2019, p. 6.",
                self.r_star
            )

        # -----
        # Parameters
        # -----
        self.bubble = bubble
        self.beta_tilde = beta_tilde
        self.a_star_a_r_ratio = a_star_a_r_ratio
        self.N_sh = N_sh
        self.nuc_type = nuc_type
        # Suppression
        self.suppression = suppression
        self.suppression_method = suppression_method
        # Accuracy
        self.z_st_thresh = z_st_thresh
        self.nT = nT
        self.n_z_lookup = n_z_lookup
        self.T_tilde_min = T_tilde_min
        self.T_tilde_max = T_tilde_max
        # Switches
        self.low_k = low_k
        # Labels
        self.label_latex = self.bubble.label_latex[:-1] + f", r_*={self.r_star:.3f}$" \
            if label_latex is None else label_latex
        self.label_unicode = self.bubble.label_unicode + f", r⁎={self.r_star:.3f}" \
            if label_unicode is None else label_unicode

        # -----
        # Values generated by compute()
        # -----
        #: $|A(z)|^2$
        self.a2: FloatArr1D | None = None
        #: $|A_\text{lookup}(z)|^2$
        self.a2_lookup: FloatArr1D | None = None
        #: $c_s^2({T}_\text{gw})$
        self.cs2: float | None = None
        #: $qT_\text{lookup}$
        self.qT_lookup: FloatArr1D | None = None
        #: $qT_\text{gw,lookup}$
        self.qT_gw_lookup: FloatArr1D | None = None
        #: $\tilde{P}_v(z)$
        self.spec_den_v: FloatArr1D | None = None
        #: $\tilde{P}_v({z}_\text{lookup})$
        self.spec_den_v_lookup: FloatArr1D | None = None
        #: $\tilde{P}_\text{gw}$
        self.spec_den_gw: FloatArr1D | None = None
        #: $\tilde{P}_\text{gw,expanded}$
        self.spec_den_gw_expanded: FloatArr1D | None = None
        #: $\tilde{P}_\text{gw,ssm}$
        self.spec_den_gw_ssm: FloatArr1D | None = None
        #: $\tilde{P}_\text{gw,int}$
        self.spec_den_gw_int: FloatArr1D | None = None
        #: $\tilde{P}_\text{gw,low}$
        self.spec_den_gw_low: FloatArr1D | None = None
        #: $\tilde{T}$
        self.T_tilde: FloatArr1D | None = None
        #: $\bar{U}_f^2$
        self.ubarf2: float | None = None
        #: $z_\text{lookup}$
        self.z_lookup: FloatArr1D | None = None

        if compute:
            self.compute(parallel=parallel)

    def beta[T: (float, FloatArr)](self, H_n: T) -> T:  # pylint: disable=missing-function-docstring
        return self.beta_tilde * H_n

    def compute(
            self,
            eps_lookup: float = 1e-8,
            lifetime_distribution_a: float = 1.,
            lambda_correction: bool = False,
            parallel: bool = True):
        if not self.bubble.solved:
            self.bubble.solve()
        self.cs2 = self.bubble.model.cs2(self.bubble.va_enthalpy_density, Phase.BROKEN)
        self.spec_den_v, self.spec_den_v_lookup, self.spec_den_gw_ssm, \
            self.a2, self.a2_lookup, self.qT_lookup, self.qT_gw_lookup, self.T_tilde, self.ubarf2, self.z_lookup, \
            self.spec_den_gw_low, self.spec_den_gw_int, self.spec_den_gw_expanded = compute(
                # Arrays
                e=self.bubble.e,
                v=self.bubble.v,
                w=self.bubble.w,
                xi=self.bubble.xi,
                y=self.y,
                # Scalars
                bubble_spacing_enlargement_factor=self.bubble_spacing_enlargement_factor,
                cs=sqrt(self.cs2),
                lifetime_distribution_a=lifetime_distribution_a,
                nu_gdh2024=self.bubble.nu_gdh2024,
                r_star=self.r_star,
                source_lifetime_factor=self.source_lifetime_factor,
                tau_end=self.tau_end,
                tau_star=self.tau_star,
                v_wall=self.bubble.v_wall,
                v_sh=self.bubble.v_sh,
                # Accuracy
                eps_lookup=eps_lookup,
                nT=self.nT,
                n_z_lookup=self.n_z_lookup,
                T_tilde_min=self.T_tilde_min,
                T_tilde_max=self.T_tilde_max,
                z_st_thresh=self.z_st_thresh,
                # Other
                nuc_type=self.nuc_type,
                lambda_correction=lambda_correction,
                parallel=parallel
        )
        self.spec_den_gw = self.spec_den_gw_expanded if self.low_k else self.spec_den_gw_ssm

    def export(self, path: str | None = None) -> dict[str, tp.Any]:
        data = {
            "bubble": self.bubble.export(),
            # Input parameters
            "beta_tilde": self.beta_tilde,
            "r_star": self.r_star,
            "a_star_a_r_ratio": self.a_star_a_r_ratio,
            "low_k": self.low_k,
            "N_sh": self.N_sh,
            "nuc_type": self.nuc_type,
            "nT": self.nT,
            "n_z_lookup": self.n_z_lookup,
            "z_st_thresh": self.z_st_thresh,
            # Computed arrays
            "a2": self.a2,
            "a2_lookup": self.a2_lookup,
            "spec_den_gw_ssm": self.spec_den_gw_ssm,
            "spec_den_gw_expanded": self.spec_den_gw_expanded,
            "spec_den_gw_int": self.spec_den_gw_int,
            "spec_den_gw_low": self.spec_den_gw_low,
            "spec_den_v": self.spec_den_v,
            "spec_den_v_lookup": self.spec_den_v_lookup,
            "T_tilde": self.T_tilde,
            "y": self.y,
            "z_lookup": self.z_lookup,
            # Computed values
            "cs2": self.cs2,
            "delta_tau_v": self.delta_tau_v,
            "dilution_of_e": self.dilution_of_e,
            "H_star_eta_star": self.H_star_eta_star,
            "H_star_tau_nl": self.H_star_tau_nl,
            "H_star_tau_sh": self.H_star_tau_sh,
            "H_star_tau_v": self.H_star_tau_v,
            "H_star_tau_v_old": self.H_star_tau_v_old,
            "k_peak_eta_star": self.k_peak_eta_star,
            "J": self.J,
            "label_latex": self.label_latex,
            "label_unicode": self.label_unicode,
            "source_lifetime_factor": self.source_lifetime_factor,
            "suppression_factor": self.suppression_factor,
            "tau_end": self.tau_end,
            "tau_star": self.tau_star,
            "ubarf2": self.ubarf2
        }
        if path is not None:
            export_json(data, path)
        return data

    # -----
    # Properties
    # -----

    @functools.cached_property
    def bubble_spacing_enlargement_factor(self) -> float:
        return 1. if self.beta_tilde is None else bubble_spacing_enlargement_factor(hx=self.hx)

    @functools.cached_property
    def delta_tau_v(self) -> float:
        r"""$\Delta \tau_v$
        $$\Delta \tau_v \equiv \frac{\delta \eta_v}{R_*} = \frac{\eta_sh N_sh}{R_*} = \frac{N_sh}{\bar{U}_f}$$
        """
        return self.N_sh / self.bubble.ubarf

    @functools.cached_property
    def dilution_of_e(self) -> float:
        return dilution_of_e(a_star_a_r_ratio=self.a_star_a_r_ratio, nu=self.bubble.nu_gdh2024)

    @functools.cached_property
    def eta_ratio(self) -> float:
        return eta_ratio(ubarf=self.bubble.ubarf, r_star=self.r_star, N_sh=self.N_sh, nu=self.bubble.nu_gdh2024)

    @functools.cached_property
    def H_star_eta_star(self) -> float:
        r"""$H_* \eta_*$
        $$H_* \eta_* = 1 + \nu_\text{gdh2024}$$
        """
        return H_eta(nu=self.bubble.nu_gdh2024)

    @functools.cached_property
    def H_star_tau_nl(self) -> float:
        r"""Hubble-scaled timescale of non-linearities $H \tau_\text{nl}$
        $$H_* \tau_\text{nl} = \frac{r_*}{\bar{U}_f}$$,
        where $\bar{U}_f \equiv v_{\text{rms}}$
        :gw_pt_ssm:`\ ` p. 6, 13
        :notes:`\ ` p. 48
        :giombi_2024_cs:`\ ` p. 2

        Please note that $\tau_\text{nl}$ and $\tau_\text{v}$ are different quantities.
        If $H \tau_\text{nl} \gg 1$, then $H \tau_\text{v} \rightarrow 1$.
        :gw_pt_ssm:`\ ` p. 13
        """
        return self.r_star / self.bubble.ubarf

    @functools.cached_property
    def H_star_tau_sh(self) -> float:  # pylint: disable=missing-function-docstring
        return H_star_tau_sh(r_star=self.r_star, ubarf=self.bubble.ubarf)

    @functools.cached_property
    def H_star_tau_v(self) -> float:  # pylint: disable=missing-function-docstring
        return H_star_tau_v(source_lifetime_factor=self.source_lifetime_factor, nu=self.bubble.nu_gdh2024)

    @functools.cached_property
    def H_star_tau_v_old(self) -> float:  # pylint: disable=missing-function-docstring
        return H_star_tau_v_old(H_star_tau_sh=self.H_star_tau_sh)

    @functools.cached_property
    def hx(self):
        return hx(self.nucleation_f)

    @functools.cached_property
    def k_peak_eta_star(self) -> float:
        r"""Peak wavenumber, scaled by conformal time at GW formation $k_\text{peak} \eta_*$
        $$k_p = \frac{2 \pi}{R_*} \Rightarrow k_p \eta_* = (1 + \nu_\text{gdh2024}) \frac{2\pi}{r_*}$$
        :giombi_2024_cs:`\ ` p. 2
        """
        return (1 + self.bubble.nu_gdh2024) * 2 * np.pi / self.r_star

    @functools.cached_property
    def J(self) -> float:
        return J(r_star=self.r_star, H_star_tau_v=self.H_star_tau_v)

    @functools.cached_property
    def nucleation_f(self) -> float:
        return nucleation_f(
            xi=self.bubble.xi, T=self.bubble.T,
            beta_tilde=self.beta_tilde, v_wall=self.bubble.v_wall, v_sh=self.bubble.v_sh
        )

    @functools.cached_property
    def pow_gw(self) -> FloatArr1D:
        r"""$\mathcal{P}_\text{gw}$"""
        return self.spec_den_gw_scaling * pow_spec(z=self.y, spec_den=self.spec_den_gw)

    @functools.cached_property
    def pow_gw_expanded(self) -> FloatArr1D:
        r"""$\mathcal{P}_\text{gw,ext}$"""
        return self.spec_den_gw_scaling * pow_spec(z=self.y, spec_den=self.spec_den_gw_expanded)

    @functools.cached_property
    def pow_gw_int(self) -> FloatArr1D:
        r"""$\mathcal{P}_\text{gw,int}$"""
        return self.spec_den_gw_scaling * pow_spec(z=self.y, spec_den=self.spec_den_gw_int)

    @functools.cached_property
    def pow_gw_low(self) -> FloatArr1D:
        r"""$\mathcal{P}_\text{gw,low}$"""
        return self.spec_den_gw_scaling * pow_spec(z=self.y, spec_den=self.spec_den_gw_low)

    @functools.cached_property
    def pow_gw_ssm(self) -> FloatArr1D:
        r"""$\mathcal{P}_\text{gw,ssm}$"""
        return self.spec_den_gw_scaling * pow_spec(z=self.y, spec_den=self.spec_den_gw_ssm)

    @functools.cached_property
    def pow_v(self) -> FloatArr1D:
        r"""$\mathcal{P}_v"""
        return pow_spec(z=self.y, spec_den=self.spec_den_v)

    @functools.cached_property
    def pow_v_tilde(self) -> FloatArr1D:
        r"""$\mathcal{P}_{\tilde{v}}$"""
        return 2 * self.pow_v

    @functools.cached_property
    def source_lifetime_factor(self) -> float:
        return source_lifetime_factor(
            ubarf=self.bubble.ubarf,
            r_star=self.r_star,
            N_sh=self.N_sh,
            nu=self.bubble.nu_gdh2024
        )

    @functools.cached_property
    def spec_den_gw_scaling(self) -> float:
        return spec_den_gw_scaling(
            ubarf2=self.ubarf2,
            mean_adiabatic_index=self.bubble.mean_adiabatic_index,
            r_star=self.r_star,
            nu=self.bubble.nu_gdh2024,
            dilution_of_e=self.dilution_of_e,
            suppression_factor=self.suppression_factor
        )

    @functools.cached_property
    def spec_den_v_tilde(self) -> FloatArr1D:
        r"""Spectral density $\tilde{P}_{\tilde{v}}$ of the velocity field $v$
        This includes
        $$\tilde{P}_{\tilde{v}}(q) = 2 \tilde{P}_v(q)$$
        :gw_pt_ssm:`\ ` eq. 4.18
        """
        return 2 * self.spec_den_v

    @functools.cached_property
    def suppression_factor(self) -> float:
        return self.suppression.suppression(
            v_wall=self.bubble.v_wall,
            alpha_n=self.bubble.alpha_n,
            method=self.suppression_method
        )

    @functools.cached_property
    def tau_end(self) -> float:
        r"""
        Time $\tau_\text{end}$ when the anisotropic stress turns off
        $$\tau_\text{end} \equiv \frac{\eta_\text{end}}{R_*}$$
        :giombi_2024_cs:`\ ` p. 8
        """
        return self.tau_star + self.delta_tau_v

    @functools.cached_property
    def tau_star(self) -> float:
        r"""Time $\tau_*$ when the anisotropic stress turns on
        $$\tau_* \equiv \frac{\eta_*}{R_*} = \frac{1 + \nu_\text{gdh2024}}{r_*}$$
        :giombi_2024_cs:`\ ` p. 8
        """
        return (1 + self.bubble.nu_gdh2024) / self.r_star

    def ubarf(self) -> float:
        return sqrt(self.ubarf2)

    def ubarf_custom_nucleation(self, nuc_type: NucType | None = None) -> float:
        return sqrt(self.ubarf2_custom_nucleation(nuc_type=nuc_type))

    def ubarf2_custom_nucleation(self, nuc_type: NucType | None = None) -> float:
        r"""$\bar{U}_f^2$ using $z$ and $|A|^2$
        The arguments $z, |A|^2, v_\text{wall}$ and the bubble spacing enlargement factor $\Lambda$
        are not directly dependent on the nucleation type, and therefore it's an adjustable parameter.
        """
        # Todo: Think which z and A2 to use here and in compute_ssm()
        return ubarf2_from_a2(
            T_tilde=self.T_tilde, z=self.qT_lookup, A2=self.a2, v_wall=self.bubble.v_wall,
            nuc_type=self.nuc_type if nuc_type is None else nuc_type,
            bubble_spacing_enlargement_factor=self.bubble_spacing_enlargement_factor
        )

    @functools.cached_property
    def z_cross_approx(self) -> float:
        return z_cross_approx(cs=self.cs, eta_ratio=self.eta_ratio, nu=self.bubble.nu_gdh2024, r_star=self.r_star)

    # -----
    # Plotting
    # -----

    def plot(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | None = None,
            **kwargs) -> "FigAndAxes":
        r"""Plot GW power spectrum $\mathcal{P}_{\text{gw}}(k)$"""
        return self.plot_gw(fig, ax, path, **kwargs)

    def plot_gw(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | None = None,
            **kwargs) -> "FigAndAxes":
        r"""Plot GW power spectrum $\mathcal{P}_{\text{gw}}(k)$"""
        from pttools.analysis.plot_spectra import plot_spectra_gw
        return plot_spectra_gw([self], fig, ax, path, **kwargs)

    def plot_v(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | None = None,
            **kwargs) -> "FigAndAxes":
        r"""Plot velocity power spectrum $\mathcal{P}_{\tilde{v}}(q)$"""
        from pttools.analysis.plot_spectra import plot_spectra_v
        return plot_spectra_v([self], fig, ax, path, **kwargs)

    def plot_spec_den_gw(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | None = None,
            **kwargs) -> "FigAndAxes":
        """Plot spectral density of scaled GW power"""
        from pttools.analysis.plot_spectra import plot_spectra_spec_den_gw
        return plot_spectra_spec_den_gw([self], fig, ax, path, **kwargs)

    def plot_spec_den_v(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | None = None,
            **kwargs) -> "FigAndAxes":
        """Plot spectral density of the velocity field $P_v(y)$"""
        from pttools.analysis.plot_spectra import plot_spectra_spec_den_v
        return plot_spectra_spec_den_v([self], fig, ax, path, **kwargs)


copy_docstrings({
    SSMSpectrum.beta: beta,
    SSMSpectrum.bubble_spacing_enlargement_factor: bubble_spacing_enlargement_factor,
    SSMSpectrum.eta_ratio: eta_ratio,
    SSMSpectrum.H_star_tau_sh: H_star_tau_sh,
    SSMSpectrum.H_star_tau_v: H_star_tau_v,
    SSMSpectrum.hx: hx,
    SSMSpectrum.J: J,
    SSMSpectrum.nucleation_f: nucleation_f,
    SSMSpectrum.source_lifetime_factor: source_lifetime_factor,
    SSMSpectrum.spec_den_gw_scaling: spec_den_gw_scaling,
    SSMSpectrum.suppression_factor: Suppression.suppression,
    SSMSpectrum.z_cross_approx: z_cross_approx
}, without_params=True)
