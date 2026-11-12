"""Base class of the bubbles that nucleate in the symmetric phase, both computed and imported."""

from __future__ import annotations

import abc
import functools
import typing as tp

import numpy as np

from pttools.bubble import props, thermo
from pttools.bubble.bubble.base import BaseBubble, NotYetSolvedError
from pttools.bubble.export import BUBBLE_FIELDS
from pttools.bubble.phase import Phase
from pttools.math.formatting import as_latex, as_unicode
import pttools.type_hints as th
from pttools.utils.docstrings import copy_docstring_dec

if tp.TYPE_CHECKING:
    from pttools.models.model import Model

if th.TYPE_CHECKING_OR_DOCS:
    from pttools.bubble.solution_type import SolutionType
    from pttools.utils.fields import Fields

__all__ = ["BaseNucleatedBubble"]


class BaseNucleatedBubble(BaseBubble, abc.ABC):
    r"""Common base class of the bubbles that nucleate in the symmetric phase, both computed and imported.

    The quantities of these bubbles, such as $\kappa$, $\omega$ and $\bar{U}_f$, are computed from the fluid profiles,
    the equation of state and the parameters of the nucleation and of the solution, e.g. $w_n$ and $v_\text{sh}$.
    Therefore, they are available both for the bubbles computed by :py:class:`pttools.bubble.bubble.Bubble`
    and for those imported with :py:class:`pttools.export.bubble.ImportedBubble`.
    This is separate from :py:class:`pttools.bubble.bubble.base.BaseBubble`,
    as these quantities are not defined for e.g. droplets.
    """

    #: The exportable fields of the bubble
    FIELDS: tp.ClassVar[Fields] = BUBBLE_FIELDS

    # -----
    # Parameters
    # -----
    alpha_n: float
    alpha_theta_bar_n: float
    sol_type: SolutionType
    Tn: float
    Psi_n: float

    # -----
    # Solution
    # -----
    alpha_plus: float
    #: $\alpha_{\bar{\theta}_+}$
    alpha_theta_bar_plus: float

    entropy_flux_p_sh: float
    r"""Incoming entropy flux at the shock
    $$\tilde{\gamma}_{+,sh} \tilde{v}_{+,sh} s_{+,sh}$$
    """

    entropy_flux_m_sh: float
    r"""Outgoing entropy flux at the shock
    $$\tilde{\gamma}_{-,sh} \tilde{v}_{-,sh} s_{-,sh}$$"""

    entropy_flux_diff_sh: float
    r"""Entropy flux difference at the shock
    $$\tilde{\gamma}_{-,sh} \tilde{v}_{-,sh} s_{-,sh} - \tilde{\gamma}_{+,sh} \tilde{v}_{+,sh} s_{+,sh}$$
    """

    sn: float
    sm_sh: float
    #: $T_{-,\text{sh}}$
    Tm_sh: float
    v_sh: float
    vm_sh: float
    vm_tilde_sh: float
    v_cj: float
    #: $w_{-,\text{sh}}$
    wm_sh: float

    # Flags
    unphysical_alpha_plus: bool

    @staticmethod
    def default_labels(model: Model, v_wall: float, alpha_n: float, with_model: bool = True) -> tuple[str, str]:
        r"""The default labels of a bubble.

        :param model: the equation of state
        :param v_wall: $v_\text{wall}$, wall speed
        :param alpha_n: $\alpha_n$, transition strength
        :param with_model: whether to include the label of the model
        :return: the labels in LaTeX and Unicode formats
        """
        # LaTeX labels are not supported in Plotly 3D plots.
        # https://github.com/plotly/plotly.js/issues/608
        model_latex = f"{model.label_latex}, " if with_model else ""
        model_unicode = f"{model.label_unicode}, " if with_model else ""
        return (
            rf"{model_latex}$v_w={as_latex(v_wall)}, \alpha_n={as_latex(alpha_n)}$",
            f"{model_unicode}v_w={as_unicode(v_wall)}, αₙ={as_unicode(alpha_n)}"
        )

    def info_str(self, prec: str = ".4f") -> str:
        """Get a string describing the key quantities of the bubble."""
        return (
            f"{self.label_unicode}: w0/wn={self.w[0] / self.wn:{prec}}, "
            f"Ubarf2={self.ubarf2:{prec}}, K={self.kinetic_energy_fraction:{prec}}, "
            f"κ={self.kappa:{prec}}, ω={self.omega:{prec}}, κ+ω={self.kappa + self.omega:{prec}}, "
            f"V-avg. trace anomaly={self.va_trace_anomaly_diff:{prec}}"
        )


    # =====
    # Quantities
    # =====

    @property
    def csb2_Tn(self) -> float:
        r"""$c_{s,b}^2(T_n)$, speed of sound squared in the broken phase at the nucleation temperature.

        $$c_{s,b}^2(T_n) = c_s^2(T_n, \phi_b)$$
        """
        return float(self.model.cs2_temp(self.Tn, Phase.BROKEN))

    @property
    def css2_Tn(self) -> float:
        r"""$c_{s,s}^2(T_n)$, speed of sound squared in the symmetric phase at the nucleation temperature.

        $$c_{s,s}^2(T_n) = c_s^2(T_n, \phi_s)$$
        """
        return float(self.model.cs2_temp(self.Tn, Phase.SYMMETRIC))

    @property
    def en(self) -> float:
        r"""Nucleation energy density $e_n = e(T_n, \phi_s)$."""
        return self.model.e(self.wn, Phase.SYMMETRIC)

    @property
    def wn(self) -> float:
        r"""Nucleation enthalpy $w_n = w(T_n, \phi_s)$.

        $$w_n \equiv w_{\text{outside}}$$
        """
        return self.w_outside

    @wn.setter
    def wn(self, wn: float) -> None:
        self.w_outside = wn

    # -----
    # At the wall
    # -----

    @property
    def vp_tilde_sh(self) -> float:
        r"""Velocity in front of the shock in the shock frame.

        The fluid ahead of the shock is still, and therefore
        $$\tilde{v}_{+,sh} = v_{sh}$$.
        """
        return self.v_sh

    @property
    def vp_vm_tilde_ratio_giese(self) -> float:
        # This docstring is copied from the model function
        r"""Giese et al. approximation for $\frac{\tilde{v}_+}{\tilde{v}_-}$, :giese_2021:`\ ` eq. 11.

        $$\frac{\tilde{v}_+}{\tilde{v}_-} \approx \frac{
        (\tilde{v}_+ \tilde{v}_- / c_{s,b}^2 - 1) + 3\alpha_{\bar{\theta}_+} }{
        (\tilde{v}_+ \tilde{v}_- / c_{s,b}^2 - 1) + 3 \tilde{v}_+ \tilde{v}_- \alpha_{\bar{\theta}_+}
        }$$
        :return: Giese et al. approximation for $\frac{\tilde{v}_+}{\tilde{v}_-}$
        """
        return self.model.vp_vm_tilde_ratio_giese(
            vp_tilde=self.vp_tilde, vm_tilde=self.vm_tilde,
            wp=self.wp, wm=self.wm
        )

    @property
    def vp_vm_tilde_ratio_giese_rel_diff(self) -> float:
        r"""Relative difference of the approximate and exact $\tilde{v}_+ / \tilde{v}_-$ ratios.

        This is the difference of the ratio of the approximate and exact
        $\tilde{v}_+, \tilde{v}_-$ ratios from unity.
        """
        return np.abs(self.vp_vm_tilde_ratio_giese / self.vp_vm_tilde_ratio - 1)

    @property
    def v_mu(self) -> float:
        r"""Maximum fluid velocity behind the bubble wall, $\mu(\xi)$."""
        # wm is the highest enthalpy inside the bubble
        cs2, _ = self.model.cs2_max(w_max=self.wm, w_min=self.w_center, phase=Phase.BROKEN)
        return props.v_max_behind(self.v_wall, np.sqrt(cs2))

    # -----
    # Averaged
    # -----

    @functools.cached_property
    @copy_docstring_dec(thermo.e_bar, without_params=True)
    def e_bar(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.e_bar(self.model, self.wn)

    @functools.cached_property
    def g_star(self) -> float:
        """Degrees of freedom $g_*$ for pressure after the bubble nucleation."""
        return self.model.gp(w=self.va_enthalpy_density, phase=Phase.BROKEN)

    @functools.cached_property
    def gs_star(self) -> float:
        """Degrees of freedom $g_{s,*}$ for entropy after the bubble nucleation."""
        return self.model.gs(w=self.va_enthalpy_density, phase=Phase.BROKEN)

    @functools.cached_property
    @copy_docstring_dec(thermo.kappa, without_params=True)
    def kappa(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kappa(self.model, self.v, self.w, self.xi, self.v_wall, delta_e_theta=self.va_trace_anomaly_diff)

    @functools.cached_property
    def kappa_giese(self) -> float:
        r"""$\kappa_\text{Giese}$, kinetic efficiency factor using the definition of Giese et al.

        $$\kappa_\text{Giese} = \frac{4 K}{3 \alpha_{\bar{\theta}_n} w_n},$$
        where $K$ is the bubble volume averaged kinetic energy density,
        :giese_2020:`\ ` eq. 12.
        """
        if not self.solved:
            raise NotYetSolvedError
        return 4 * self.kinetic_energy_density / (3 * self.alpha_theta_bar_n * self.wn)

    @functools.cached_property
    @copy_docstring_dec(thermo.mean_adiabatic_index, without_params=True)
    def mean_adiabatic_index(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.mean_adiabatic_index(self.w_bar, self.e_bar)

    @functools.cached_property
    def nu_gdh2024(self) -> float:
        r"""$\nu_\text{gdh2024}$ at the volume-averaged enthalpy density in the broken phase.

        See :func:`pttools.bubble.thermo.nu_gdh2024`.
        """
        if not self.solved:
            raise NotYetSolvedError
        return self.model.nu_gdh2024(self.va_enthalpy_density)

    @functools.cached_property
    @copy_docstring_dec(thermo.omega, without_params=True)
    def omega(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.omega(self.model, self.w, self.xi, self.v_wall, delta_e_theta=self.va_trace_anomaly_diff)

    @functools.cached_property
    def omega_barotropic(self) -> float:
        r"""$\omega$, barotropic equation of state parameter at the volume-averaged enthalpy density.

        This is evaluated in the broken phase. See :func:`pttools.bubble.thermo.omega_barotropic`.
        """
        if not self.solved:
            raise NotYetSolvedError
        return self.model.omega(self.va_enthalpy_density, Phase.BROKEN)

    @functools.cached_property
    def T_star(self) -> float:
        r"""Average temperature $T_*$ after the bubble nucleation."""
        if not self.solved:
            raise NotYetSolvedError
        return self.model.temp(w=self.va_enthalpy_density, phase=Phase.BROKEN)

    @functools.cached_property
    def ubarf(self) -> float:
        r"""Enthalpy-weighted RMS fluid velocity $\bar{U}_\text{f}$.

        In some sources such as :giombi_2024_cs:`\ `, this is denoted as $v_\text{rms}$.
        """
        return np.sqrt(self.ubarf2)

    @functools.cached_property
    @copy_docstring_dec(thermo.ubarf2, without_params=True)
    def ubarf2(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        ubarf2 = thermo.ubarf2(
            self.v, self.w, self.xi,
            v_wall=self.v_wall,
            ek_bva=self.kinetic_energy_density,
            w_bar=self.w_bar
        )
        thermo.warn_if_ubarf_relativistic(ubarf2, source=self.label_unicode)
        return ubarf2

    @functools.cached_property
    @copy_docstring_dec(thermo.w_bar, without_params=True)
    def w_bar(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.w_bar(self.w, self.xi, self.v_wall)

    # -----
    # bva = bubble volume averaged
    # -----
    @functools.cached_property
    @copy_docstring_dec(thermo.entropy_density_diff, without_params=True)
    def entropy_density_diff(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.entropy_density_diff(self.model, self.w, self.xi, self.v_wall, self.phase)

    @functools.cached_property
    def entropy_density_diff_relative(self) -> float:
        r"""Bubble volume averaged entropy density difference relative to the entropy density $s_n$ outside the bubble.

        See :attr:`entropy_density_diff`.
        """
        if not self.solved:
            raise NotYetSolvedError
        return self.entropy_density_diff / self.model.s(self.wn, Phase.SYMMETRIC)

    @functools.cached_property
    @copy_docstring_dec(thermo.kinetic_energy_density, without_params=True)
    def kinetic_energy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kinetic_energy_density(self.v, self.w, self.xi, self.v_wall)

    @functools.cached_property
    @copy_docstring_dec(thermo.kinetic_energy_fraction, without_params=True)
    def kinetic_energy_fraction(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kinetic_energy_fraction(ek_bva=self.kinetic_energy_density, eb=self.e_bar)

    @functools.cached_property
    @copy_docstring_dec(thermo.thermal_energy_density, without_params=True)
    def thermal_energy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.thermal_energy_density(v_shock=self.v_sh, eqp=self.va_thermal_energy_density)

    @functools.cached_property
    @copy_docstring_dec(thermo.thermal_energy_density_diff, without_params=True)
    def thermal_energy_density_diff(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.thermal_energy_density_diff(self.w, self.xi, self.v_wall)

    @functools.cached_property
    @copy_docstring_dec(thermo.thermal_energy_fraction, without_params=True)
    def thermal_energy_fraction(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.thermal_energy_fraction(eqp_bar=self.thermal_energy_density, eb=self.e_bar)

    @functools.cached_property
    @copy_docstring_dec(thermo.trace_anomaly_diff, without_params=True)
    def trace_anomaly(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.trace_anomaly_diff(self.model, self.w, self.xi, self.v_wall, self.phase)

    # -----
    # va = volume averaged
    # -----
    @functools.cached_property
    @copy_docstring_dec(thermo.va_enthalpy_density, without_params=True)
    def va_enthalpy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_enthalpy_density(eqp_bar=self.thermal_energy_density)

    @functools.cached_property
    @copy_docstring_dec(thermo.va_entropy_density_diff, without_params=True)
    def va_entropy_density_diff(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_entropy_density_diff(self.model, self.w, self.xi, self.v_wall, self.phase)

    @functools.cached_property
    def va_entropy_density_diff_relative(self) -> float:
        r"""Volume-averaged entropy density difference relative to the entropy density $s_n$ outside the bubble.

        See :attr:`va_entropy_density_diff`.
        """
        if not self.solved:
            raise NotYetSolvedError
        return self.va_entropy_density_diff / self.model.s(self.wn, Phase.SYMMETRIC)

    @functools.cached_property
    @copy_docstring_dec(thermo.va_kinetic_energy_fraction, without_params=True)
    def va_kinetic_energy_fraction(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_kinetic_energy_fraction(ek_va=self.va_kinetic_energy_density, eb=self.e_bar)

    @functools.cached_property
    @copy_docstring_dec(thermo.va_thermal_energy_density, without_params=True)
    def va_thermal_energy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_thermal_energy_density(
            v_shock=self.v_sh, wn=self.wn, ek=self.va_kinetic_energy_density, delta_e_theta=self.va_trace_anomaly_diff)

    @functools.cached_property
    @copy_docstring_dec(thermo.va_thermal_energy_density_diff, without_params=True)
    def va_thermal_energy_density_diff(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_thermal_energy_density_diff(self.w, self.xi)

    @functools.cached_property
    @copy_docstring_dec(thermo.va_thermal_energy_fraction, without_params=True)
    def va_thermal_energy_fraction(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_thermal_energy_fraction(eqp=self.va_thermal_energy_density, eb=self.e_bar)

    @functools.cached_property
    @copy_docstring_dec(thermo.va_trace_anomaly_diff, without_params=True)
    def va_trace_anomaly_diff(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_trace_anomaly_diff(self.model, self.w, self.xi, self.v_wall, self.phase)
