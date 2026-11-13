"""Base class of the Sound Shell Model spectra, both computed and imported."""

from __future__ import annotations

import abc
import functools
from math import sqrt
import os
import typing as tp

import numpy as np

from pttools.bubble import BaseNucleatedBubble, nu_gdh2024
from pttools.bubble import omega_barotropic as omega_barotropic_func
from pttools.math.formatting import as_latex, as_unicode
from pttools.ssm.barotropic import eta_ratio
from pttools.ssm.export import SSM_SPECTRUM_FIELDS
from pttools.ssm.low_k.intersection import z_cross_approx
from pttools.ssm.nucleation import NucType, v_eff
from pttools.ssm.pow_spec import pow_spec
from pttools.ssm.scaling import H_star_eta_sh, H_star_eta_v_old, J
from pttools.ssm.spec_den_gw import spec_den_gw_scaling
from pttools.ssm.spectrum_base import BaseSpectrum
from pttools.ssm.ssm import ubarf2_from_a2
from pttools.ssm.suppression import Suppression, SuppressionMethod
import pttools.type_hints as th
from pttools.type_hints import FloatArr, FloatArr1D
from pttools.utils.decorators import cached_property
from pttools.utils.docstrings import copy_docstring_dec
from pttools.utils.fields import Extractable, Fields

if tp.TYPE_CHECKING:
    from pttools.plot.utils import FigAndAxes

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["BaseSSMSpectrum"]


class BaseSSMSpectrum(Extractable, BaseSpectrum, abc.ABC):
    """Common base class of the Sound Shell Model spectra, both computed and imported.

    This defines the attributes of the spectra and the quantities that are computed from them,
    so that the plotting and analysis functions can take any SSM spectrum.
    The attributes are set either by the computation of :py:class:`SSMSpectrum`,
    or from the data of a file by :py:class:`pttools.export.ssm.ImportedSSMSpectrum`.
    The quantities that need data that is not available, e.g. the spectral densities of a spectrum
    that has been exported without them, raise :py:class:`pttools.export.imported.NotImportedError`.
    """

    #: The exportable fields of the spectrum
    FIELDS: tp.ClassVar[Fields] = SSM_SPECTRUM_FIELDS

    #: Unique identifier of the spectrum, which is used for deduplication when exporting.
    #: This is preserved when the spectrum is pickled, e.g. when sent to another process.
    id: str

    # -----
    # Parameters
    # -----
    beta_tilde: float | None
    N_sh: float
    nuc_type: NucType
    #: Whether $\tilde{\beta}$ is allowed as an input parameter for simultaneous nucleation
    allow_beta_tilde_for_simultaneous: bool
    suppression: Suppression
    suppression_method: SuppressionMethod
    # Overrides of the equation of state parameters of the bubble
    _nu_gdh2024: float | None
    _omega_barotropic: float | None
    # Accuracy
    z_st_thresh: float
    nT: int  # noqa: N815
    nx_P_tilde_gw: int | None  # noqa: N815
    n_z_lookup: int
    T_tilde_min: float
    T_tilde_max: float
    # Switches
    low_k: bool
    low_k_cs_friction: bool
    low_k_fallback_zp_gw: bool

    # -----
    # Results
    # -----
    a2: FloatArr1D
    a2_lookup: FloatArr1D
    #: Bubble spacing enlargement factor $\Lambda$
    bubble_spacing_enlargement_factor: float
    #: $c_s(T_{\text{gw}})$
    cs: float
    cs2: float
    #: Nucleation $f$
    nucleation_f: float
    #: Nucleation $h_x$
    hx: float
    #: $qT_\text{lookup}$
    qT_lookup: FloatArr1D
    #: $qT_\text{gw,lookup}$
    qT_gw_lookup: FloatArr1D  # noqa: N815
    spec_den_v: FloatArr1D
    spec_den_v_lookup: FloatArr1D
    spec_den_gw: FloatArr1D
    spec_den_gw_expanded: FloatArr1D
    spec_den_gw_ssm: FloatArr1D
    spec_den_gw_int: FloatArr1D
    #: $\tilde{P}_\text{gw,low}$
    spec_den_gw_low: FloatArr1D
    T_tilde: FloatArr1D
    ubarf2: float
    z_lookup: FloatArr1D
    #: $\tilde{P}_\text{gw,ssm} / \Upsilon_\ell$, GW spectral density without the source lifetime factor
    spec_den_gw_ssm_unscaled: FloatArr1D
    #: $\mathcal{I}_v$, source contribution of the low-$k$ approximation
    Iv: float
    #: $x_{p,v}$, $x$ at the maximum of $x \tilde{P}_v(x)$, or nan if the lookup array does not cover it,
    #: see :py:func:`pttools.ssm.low_k.intersection.xp_v`
    xp_v: float

    @property
    @abc.abstractmethod
    def bubble(self) -> BaseNucleatedBubble:
        """The bubble of the spectrum.

        This is a read-only property, so that the subclasses can narrow its type, e.g. to a computed bubble.
        """

    def beta[T: (float, FloatArr)](self, H_n: T) -> T:
        r"""$\beta$, nucleation rate parameter, aka. inverse phase transition duration.

        $$\beta = \tilde{\beta} H_n$$

        :param H_n: $H_n$, Hubble parameter at nucleation
        :return: $\beta$
        :raises ValueError: if $\tilde{\beta}$ has not been given for this spectrum
        """
        if self.beta_tilde is None:
            raise ValueError("beta_tilde has not been set for this spectrum.")
        return self.beta_tilde * H_n

    @staticmethod
    def default_labels(bubble: BaseNucleatedBubble, r_star: float | None) -> tuple[str, str]:
        r"""The default labels of a spectrum, which are those of the bubble with $r_*$.

        :param bubble: the bubble
        :param r_star: $r_*$, Hubble-scaled mean bubble spacing
        :return: the labels in LaTeX and Unicode formats
        """
        return (
            bubble.label_latex.rstrip("$") + f", r_*={as_latex(r_star)}$",
            bubble.label_unicode + f", r⁎={as_unicode(r_star)}"
        )

    def covers(self, y: FloatArr1D, rtol: float = 1e-9) -> bool:
        r"""Whether the $z$ values are within the range of the $y$ array of this spectrum.

        This tells whether :py:meth:`with_params` can interpolate this spectrum to the $z$ values.

        :param y: $z = kR_*$ values
        :param rtol: relative tolerance for the endpoints of the range
        :return: True if all the values are within the range
        """
        if y.size == 0:
            return False
        return bool(y.min() >= self.y.min() * (1 - rtol) and y.max() <= self.y.max() * (1 + rtol))

    # -----
    # Properties
    # -----

    @property
    def csb2_Tn(self) -> float:
        r"""$c_{s,b}^2(T_n)$, speed of sound squared in the broken phase at the nucleation temperature.

        See :py:attr:`pttools.bubble.bubble.Bubble.csb2_Tn`.
        """
        return self.bubble.csb2_Tn

    @property
    def css2_Tn(self) -> float:
        r"""$c_{s,s}^2(T_n)$, speed of sound squared in the symmetric phase at the nucleation temperature.

        See :py:attr:`pttools.bubble.bubble.Bubble.css2_Tn`.
        """
        return self.bubble.css2_Tn

    @functools.cached_property
    @tp.override
    def delta_tau_v(self) -> float:
        r"""$\Delta \tau_\text{v}$, source duration in units of the comoving mean bubble spacing.

        $$\Delta \tau_\text{v} \equiv \frac{\Delta \eta_\text{v}}{R_*}
        = \frac{N_{\text{sh}} \eta_\text{sh}}{R_*} = \frac{N_{\text{sh}}}{\bar{U}_f},$$
        where $\tau \equiv \eta / R_*$ is the dimensionless conformal time of
        :giombi_2026:`\ ` sec. 2.5 and :giombi_2024_cs:`\ ` p. 10,
        and $\eta_\text{sh} \approx R_* / \bar{U}_f$
        (see :py:func:`pttools.ssm.scaling.H_star_eta_sh`).
        """
        return self.N_sh / self.bubble.ubarf

    @functools.cached_property
    @tp.override
    @copy_docstring_dec(eta_ratio, without_params=True)
    def eta_ratio(self) -> float:
        return eta_ratio(ubarf=self.bubble.ubarf, r_star=self.r_star, N_sh=self.N_sh, nu=self.nu_gdh2024)

    @functools.cached_property
    @copy_docstring_dec(H_star_eta_sh, without_params=True)
    def H_star_eta_sh(self) -> float:
        return H_star_eta_sh(r_star=self.r_star, ubarf=self.bubble.ubarf)

    @functools.cached_property
    @copy_docstring_dec(H_star_eta_v_old, without_params=True)
    def H_star_eta_v_old(self) -> float:
        return H_star_eta_v_old(H_star_eta_sh=self.H_star_eta_sh)

    @functools.cached_property
    def kp_sw_eta_star(self) -> float:
        r"""$k_p \eta_*$, the sound wave scale in units of the conformal time at the start of the acoustic phase.

        $$k_p = \frac{2 \pi}{R_*} \Rightarrow k_p \eta_* = (1 + \nu_\text{gdh2024}) \frac{2\pi}{r_*}$$
        :giombi_2024_cs:`\ ` p. 2.
        Here $k_p$ is the characteristic wavenumber of the sound waves, which the article calls the peak frequency.
        It is not the peak of the GW power spectrum, which is $k_\star$ in the article,
        and for the Sound Shell Model it is not the maximum of the velocity spectral density either,
        see :py:data:`pttools.ssm.low_k.const.ZP_SW`.
        """
        return (1 + self.nu_gdh2024) * 2 * np.pi / self.r_star

    @functools.cached_property
    @copy_docstring_dec(J, without_params=True)
    def J(self) -> float:
        return J(r_star=self.r_star, H_star_eta_v=self.H_star_eta_v)

    @functools.cached_property
    @tp.override
    @copy_docstring_dec(nu_gdh2024, without_params=True)
    def nu_gdh2024(self) -> float:
        if self._nu_gdh2024 is not None:
            return self._nu_gdh2024
        return self.bubble.nu_gdh2024

    @functools.cached_property
    @copy_docstring_dec(omega_barotropic_func, without_params=True)
    def omega_barotropic(self) -> float:
        if self._omega_barotropic is not None:
            return self._omega_barotropic
        return self.bubble.omega_barotropic

    @cached_property
    @tp.override
    def pow_gw(self) -> FloatArr1D:
        return self.spec_den_gw_scaling * pow_spec(z=self.y, spec_den=self.spec_den_gw)

    @functools.cached_property
    def pow_gw_expanded(self) -> FloatArr1D:
        r"""$\mathcal{P}_\text{gw,ext}$."""
        return self.spec_den_gw_scaling * pow_spec(z=self.y, spec_den=self.spec_den_gw_expanded)

    @functools.cached_property
    def pow_gw_int(self) -> FloatArr1D:
        r"""$\mathcal{P}_\text{gw,int}$."""
        return self.spec_den_gw_scaling * pow_spec(z=self.y, spec_den=self.spec_den_gw_int)

    @functools.cached_property
    def pow_gw_low(self) -> FloatArr1D:
        r"""$\mathcal{P}_\text{gw,low}$."""
        return self.spec_den_gw_scaling * pow_spec(z=self.y, spec_den=self.spec_den_gw_low)

    @functools.cached_property
    def pow_gw_rate(self) -> FloatArr1D:
        r"""$\frac{1}{H_n R_*} \frac{d\mathcal{P}_\text{gw}}{d(H_n t)}$, flat-space GW power growth rate.

        $$\frac{1}{H_n R_*} \frac{d\mathcal{P}_\text{gw}}{d(H_n t)}
        = 3 \left( \Gamma \bar{U}_f^2 \right)^2 \Sigma \frac{z^3}{2\pi^2} \frac{\tilde{P}_\text{gw}(z)}{\Upsilon}$$
        :gw_pt_ssm:`\ ` eq. 3.44, 3.45.
        This is :py:attr:`pow_gw_ssm` without the factors of the expansion of the universe and the source lifetime,
        i.e. the GW power that a stationary source produces per Hubble time in flat space.
        It can therefore be compared directly with flat-space simulations, such as those of :correia_2025:`\ `.
        The low-$k$ approximation is not included, as it describes the effects of the finite source lifetime.
        The suppression factor $\Sigma$ is included unless the suppression method is
        :py:attr:`pttools.ssm.suppression.suppression.SuppressionMethod.NONE`.
        """
        return 3 * (self.bubble.mean_adiabatic_index * self.ubarf2)**2 * self.suppression_factor \
            * pow_spec(z=self.y, spec_den=self.spec_den_gw_ssm_unscaled)

    @functools.cached_property
    def pow_gw_ssm(self) -> FloatArr1D:
        r"""$\mathcal{P}_\text{gw,ssm}$."""
        return self.spec_den_gw_scaling * pow_spec(z=self.y, spec_den=self.spec_den_gw_ssm)

    @functools.cached_property
    def pow_U_par(self) -> FloatArr1D:
        r"""$\mathcal{P}_{U_{\parallel}}(z)$, power spectrum of the enthalpy-weighted fluid velocity.

        $$\mathcal{P}_{U_{\parallel}}(z) = \bar{U}_f^2 \mathcal{P}_{\tilde{v}}(z),$$
        normalised so that $\int \frac{dz}{z} \mathcal{P}_{U_{\parallel}} = \bar{U}_f^2$ when the range of $z$
        covers the whole spectrum.
        In the Sound Shell Model the flow consists of sound waves only, and is therefore purely compressional.
        This can be compared with the spectrum of the compressional component of
        $U^i = \sqrt{w/\bar{w}} \gamma v^i$ of :correia_2025:`\ ` eq. 20.
        """
        return self.ubarf2 * self.pow_v_tilde

    @functools.cached_property
    def pow_v(self) -> FloatArr1D:
        r"""$\mathcal{P}_v$, velocity power spectrum."""
        return pow_spec(z=self.y, spec_den=self.spec_den_v)

    @functools.cached_property
    def pow_v_tilde(self) -> FloatArr1D:
        r"""$\mathcal{P}_{\tilde{v}}$."""
        return 2 * self.pow_v

    @functools.cached_property
    @copy_docstring_dec(spec_den_gw_scaling, without_params=True)
    def spec_den_gw_scaling(self) -> float:
        return spec_den_gw_scaling(
            ubarf2=self.ubarf2,
            mean_adiabatic_index=self.bubble.mean_adiabatic_index,
            r_star=self.r_star,
            nu=self.nu_gdh2024,
            dilution_of_e=self.dilution_of_e,
            suppression_factor=self.suppression_factor
        )

    @functools.cached_property
    def spec_den_v_tilde(self) -> FloatArr1D:
        r"""$\tilde{P}_{\tilde{v}}$, spectral density of the velocity field $v$.

        This includes
        $$\tilde{P}_{\tilde{v}}(q) = 2 \tilde{P}_v(q)$$
        :gw_pt_ssm:`\ ` eq. 4.18.
        """
        return 2 * self.spec_den_v

    @functools.cached_property
    @copy_docstring_dec(Suppression.suppression, without_params=True)
    def suppression_factor(self) -> float:
        return self.suppression.suppression(
            v_wall=self.bubble.v_wall,
            alpha_n=self.bubble.alpha_n,
            method=self.suppression_method
        )

    @property
    def ubarf(self) -> float:
        r"""$\bar{U}_f$, RMS fluid velocity taking into account the nucleation history.

        $$\bar{U}_f = \sqrt{\bar{U}_f^2},$$
        where $\bar{U}_f^2$ is computed by :py:meth:`compute` from $z$ and ${\lvert A \rvert}^2$
        with :py:func:`pttools.ssm.ssm.ubarf2_from_a2`.
        See :py:attr:`pttools.bubble.bubble.Bubble.ubarf` for the value computed directly from the fluid profile.
        """
        return sqrt(self.ubarf2)

    def ubarf_custom_nucleation(self, nuc_type: NucType | None = None) -> float:
        r"""$\bar{U}_f$ using $z$ and ${\lvert A \rvert}^2$ for the given nucleation type.

        See :py:meth:`ubarf2_custom_nucleation`.

        :param nuc_type: nucleation type, defaults to the nucleation type of the spectrum
        :return: $\bar{U}_f$, enthalpy-weighted RMS fluid velocity
        """
        return sqrt(self.ubarf2_custom_nucleation(nuc_type=nuc_type))

    def ubarf2_custom_nucleation(self, nuc_type: NucType | None = None) -> float:
        r"""$\bar{U}_f^2$ using $z$ and ${\lvert A \rvert}^2$.

        The arguments $z, {\lvert A \rvert}^2$ and $v_{\text{wall}}$
        are not directly dependent on the nucleation type, and therefore it's an adjustable parameter.
        The result does not depend on the bubble spacing enlargement factor $\Lambda$,
        see :py:func:`pttools.ssm.ssm.ubarf2_from_a2`.
        """
        # Todo: Think which z and A2 to use here and in compute_ssm()
        return ubarf2_from_a2(
            T_tilde=self.T_tilde, z=self.qT_lookup, A2=self.a2, v_wall=self.bubble.v_wall,
            nuc_type=self.nuc_type if nuc_type is None else nuc_type
        )

    @functools.cached_property
    @copy_docstring_dec(v_eff, without_params=True)
    def v_eff(self) -> float:
        return v_eff(self.nucleation_f, self.bubble.v_wall)

    @functools.cached_property
    @copy_docstring_dec(z_cross_approx, without_params=True)
    def z_cross_approx(self) -> float:
        return z_cross_approx(cs=self.cs, eta_ratio=self.eta_ratio, nu=self.nu_gdh2024, r_star=self.r_star)

    # -----
    # Plotting
    # -----

    def plot_v(
            self,
            fig: Figure | None = None,
            ax: Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            **kwargs: tp.Any) -> FigAndAxes:
        r"""Plot velocity power spectrum $\mathcal{P}_{\tilde{v}}(q)$."""
        from pttools.plot.spectra import plot_spectra_v  # noqa: PLC0415
        return plot_spectra_v([self], ax=ax, fig=fig, path=path, **kwargs)

    def plot_spec_den_gw(
            self,
            fig: Figure | None = None,
            ax: Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            **kwargs: tp.Any) -> FigAndAxes:
        """Plot spectral density of scaled GW power."""
        from pttools.plot.spectra import plot_spectra_spec_den_gw  # noqa: PLC0415
        return plot_spectra_spec_den_gw([self], ax=ax, fig=fig, path=path, **kwargs)

    def plot_spec_den_v(
            self,
            fig: Figure | None = None,
            ax: Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            **kwargs: tp.Any) -> FigAndAxes:
        """Plot spectral density of the velocity field $P_v(y)$."""
        from pttools.plot.spectra import plot_spectra_spec_den_v  # noqa: PLC0415
        return plot_spectra_spec_den_v([self], ax=ax, fig=fig, path=path, **kwargs)
