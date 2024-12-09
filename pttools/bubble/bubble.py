"""A solution of the hydrodynamic equations"""

import datetime
import functools
import logging
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble.alpha import alpha_n_max_deflagration_bag
from pttools.bubble.boundary import Phase, SolutionType
from pttools.bubble.fluid import sound_shell_generic
from pttools.bubble import const
from pttools.bubble import props
from pttools.bubble.relativity import gamma, lorentz
from pttools.bubble import thermo
from pttools.bubble import transition
from pttools.speedup.export import export_json
if tp.TYPE_CHECKING:
    from pttools.models.model import Model
    from pttools.models.const_cs import ConstCSModel

if tp.TYPE_CHECKING:
    from pttools.analysis.utils import FigAndAxes

logger = logging.getLogger(__name__)


class NotYetSolvedError(RuntimeError):
    """Error for accessing the properties of a bubble that has not been solved yet"""


class Bubble:
    """A solution of the hydrodynamic equations"""
    def __init__(
            self,
            model: "Model", v_wall: float, alpha_n: float,
            solve: bool = True,
            sol_type: SolutionType = None,
            label_latex: str = None,
            label_unicode: str = None,
            wn_guess: float = None,
            wm_guess: float = None,
            theta_bar: bool = False,
            t_end: float = const.T_END_DEFAULT,
            n_xi: int = const.N_XI_DEFAULT,
            thin_shell_t_points_min: int = const.THIN_SHELL_T_POINTS_MIN,
            use_bag_solver: bool = False,
            use_giese_solver: bool = False,
            log_success: bool = False,
            allow_invalid: bool = False,
            log_invalid: bool = True):
        if use_bag_solver and use_giese_solver:
            raise ValueError("Both bag and Giese solvers cannot be used at the same time.")

        if v_wall < 0 or v_wall > 1:
            raise ValueError(f"Invalid v_wall={v_wall}")

        if not theta_bar:
            model.validate_alpha_n(alpha_n, allow_invalid=allow_invalid, log_invalid=log_invalid)
        self.wn = model.wn(alpha_n, wn_guess, theta_bar=theta_bar)
        if theta_bar:
            self.alpha_theta_bar_n = alpha_n
            self.alpha_n = model.alpha_n_from_alpha_theta_bar_n(alpha_theta_bar_n=alpha_n, wn=self.wn)
            model.validate_alpha_n(self.alpha_n, allow_invalid=allow_invalid, log_invalid=log_invalid)
        else:
            self.alpha_n = alpha_n
            self.alpha_theta_bar_n = model.alpha_theta_bar_n_from_alpha_n(alpha_n=alpha_n, wn=self.wn)

        self.sol_type = transition.validate_solution_type(
            model,
            v_wall=v_wall, alpha_n=alpha_n, sol_type=sol_type,
            wn=self.wn, wm_guess=wm_guess
        )

        # Parameters
        self.model: Model = model
        self.v_wall = v_wall
        self.t_end = t_end
        self.n_xi = n_xi
        self.thin_shell_t_points_min = thin_shell_t_points_min
        self.log_success = log_success

        # Computed parameters
        self.Tn = model.temp(self.wn, Phase.SYMMETRIC)
        if self.Tn > model.T_crit:
            msg = f"Bubbles form only when T_nuc < T_crit. Got: T_nuc={self.Tn}, T_crit={model.T_crit}"
            if log_invalid:
                logger.error(msg)
            if not allow_invalid:
                raise ValueError(msg)

        self.Psi_n = model.Psi_n(self.wn)

        # if isinstance(model, ConstCSModel)
        if hasattr(model, "css2") and hasattr(model, "csb2"):
            model: ConstCSModel
            self.alpha_theta_bar_n = model.alpha_theta_bar_n_from_alpha_n(alpha_n)
            self.alpha_theta_bar_n_min_lte = model.alpha_theta_bar_n_min_lte(self.wn, self.sol_type, Psi_n=self.Psi_n)
            self.alpha_theta_bar_n_max_lte = model.alpha_theta_bar_n_max_lte(self.wn, self.sol_type, Psi_n=self.Psi_n)
            # Here LTE = no entropy generation
            # if log_invalid and (self.alpha_theta_bar_n_max_lte < self.alpha_theta_bar_n_min_lte
            #                     or self.alpha_theta_bar_n_max_lte < 0):
            #     logger.error(
            #         "Got invalid limits for alpha_theta_bar_n_lte: "
            #         f"min={self.alpha_theta_bar_n_min_lte}, max={self.alpha_theta_bar_n_max_lte}"
            #     )
            # if log_invalid and self.alpha_theta_bar_n < self.alpha_theta_bar_n_min_lte:
            #     logger.warning("alpha_theta_bar_n=%s < lte_min=%s", self.alpha_theta_bar_n, self.alpha_theta_bar_n_min_lte)
            # if log_invalid and self.alpha_theta_bar_n > self.alpha_theta_bar_n_max_lte:
            #     logger.warning("alpha_theta_bar_n=%s > lte_max=%s", self.alpha_theta_bar_n, self.alpha_theta_bar_n_max_lte)

        if log_invalid and self.sol_type == SolutionType.DETON and self.Psi_n < 0.75:
            logger.info(
                "This detonation may not exist, as LTE predicts a large alpha_n_hyb_max for Psi_n=%s < 0.75. "
                "Please see Ai et al. (2023), p. 15.",
                self.Psi_n
            )

        # Flags
        self.solved = False
        # Todo: clarify the differences between these
        self.solver_failed = False
        self.no_solution_found = False
        # Specific errors
        self.negative_entropy_flux = False
        self.negative_net_entropy_change = False
        self.numerical_error = False
        self.unphysical_alpha_plus = False
        self.use_bag_solver = use_bag_solver
        self.use_giese_solver = use_giese_solver

        # LaTeX labels are not supported in Plotly 3D plots.
        # https://github.com/plotly/plotly.js/issues/608
        self.label_latex = rf"{self.model.label_latex}, $v_w={v_wall}, \alpha_n={alpha_n}$" \
            if label_latex is None else label_latex
        self.label_unicode = f"{self.model.label_unicode}, v_w={v_wall}, αₙ={alpha_n}" \
            if label_unicode is None else label_unicode
        self.notes: tp.List[str] = []

        # Output arrays
        self.v: tp.Optional[np.ndarray] = None
        self.w: tp.Optional[np.ndarray] = None
        self.xi: tp.Optional[np.ndarray] = None
        self.phase: tp.Optional[np.ndarray] = None

        # Output values
        self.alpha_plus: tp.Optional[float] = None
        self.alpha_theta_bar_plus: tp.Optional[float] = None
        self.elapsed: tp.Optional[float] = None
        self.sp: tp.Optional[float] = None
        self.sm: tp.Optional[float] = None
        self.sm_sh: tp.Optional[float] = None
        self.sn: tp.Optional[float] = None
        self.vp: tp.Optional[float] = None
        self.vm: tp.Optional[float] = None
        self.vp_tilde: tp.Optional[float] = None
        self.vm_tilde: tp.Optional[float] = None
        self.v_sh: tp.Optional[float] = None
        self.vm_sh: tp.Optional[float] = None
        self.vm_tilde_sh: tp.Optional[float] = None
        self.v_cj: tp.Optional[float] = None
        self.w_center: tp.Optional[float] = None
        self.wp: tp.Optional[float] = None
        self.wm: tp.Optional[float] = None
        self.wm_sh: tp.Optional[float] = None

        if solve:
            self.solve()
        elif log_success:
            logger.info(
                "Initialized a bubble with: "
                "model=%s, v_w=%s, alpha_n=%s, T_nuc=%s, w_nuc=%s",
                self.model.label_unicode, v_wall, alpha_n, self.Tn, self.wn
            )

    def add_note(self, note: str):
        self.notes.append(note)

    def export(self, path: str = None) -> tp.Dict[str, any]:
        data = {
            "datetime": datetime.datetime.now(),
            "notes": self.notes,
            # Input parameters
            "model": self.model.export(),
            "v_wall": self.v_wall,
            "alpha_n": self.alpha_n,
            "sol_type": self.sol_type,
            "t_end": self.t_end,
            "n_xi": self.n_xi,
            "thin_shell_limit": self.thin_shell_t_points_min,
            # Solution
            "v": self.v,
            "w": self.w,
            "xi": self.xi,
            # Solution parameters
            "alpha_plus": self.alpha_plus,
            "sp": self.sp,
            "sm": self.sm,
            "sm_sh": self.sm_sh,
            "sn": self.sn,
            "Tn": self.Tn,
            "v_cj": self.v_cj,
            "vp": self.vp,
            "vm": self.vm,
            "vp_tilde": self.vp_tilde,
            "vm_tilde": self.vm_tilde,
            "v_sh": self.v_sh,
            "vm_sh": self.vm_sh,
            "vm_tilde_sh": self.vm_tilde_sh,
            "wn": self.wn,
            "wp": self.wp,
            "wm": self.wm,
            "wm_sh": self.wm_sh
        }
        if path is not None:
            export_json(data, path)
        return data

    def info_str(self, prec: str = ".4f") -> str:
        return \
            f"{self.label_unicode}: w0/wn={self.w[0] / self.wn:{prec}}, " \
            f"Ubarf2={self.ubarf2:{prec}}, K={self.kinetic_energy_fraction:{prec}}, " \
            f"κ={self.kappa:{prec}}, ω={self.omega:{prec}}, κ+ω={self.kappa + self.omega:{prec}}, " \
            f"V-avg. trace anomaly={self.va_trace_anomaly_diff:{prec}}"

    def solve(
            self,
            sum_rtol_warning: float = 1.5e-2,
            sum_rtol_error: float = 5e-2,
            error_prec: str = ".4f",
            use_bag_solver: bool = False,
            use_giese_solver: bool = False,
            log_high_alpha_n_failures: bool = True,
            log_negative_entropy: bool = True):
        if self.solved:
            msg = "Re-solving an already solved bubble! Already computed quantities will not be updated due to caching."
            logger.warning(msg)
            self.add_note(msg)

        use_bag_solver = self.use_bag_solver or use_bag_solver
        use_giese_solver = self.use_giese_solver or use_giese_solver
        if use_bag_solver and use_giese_solver:
            raise ValueError("Both bag and Giese solvers cannot be used at the same time.")

        alpha_n_max_bag = alpha_n_max_deflagration_bag(self.v_wall)
        high_alpha_n = alpha_n_max_bag - self.alpha_n < 0.05

        try:
            # Todo: make the solver errors more specific
            self.v, self.w, self.xi, self.sol_type, \
                self.vp, self.vm, self.vp_tilde, self.vm_tilde, \
                self.v_sh, self.vm_sh, self.vm_tilde_sh, \
                self.wp, self.wm, self.wm_sh, self.v_cj, self.solver_failed, self.elapsed = \
                sound_shell_generic(
                    model=self.model,
                    v_wall=self.v_wall, alpha_n=self.alpha_n, sol_type=self.sol_type,
                    wn=self.wn,
                    alpha_n_max_bag=alpha_n_max_bag,
                    high_alpha_n=high_alpha_n, t_end=self.t_end, n_xi=self.n_xi, thin_shell_limit=self.thin_shell_t_points_min,
                    use_bag_solver=use_bag_solver, use_giese_solver=use_giese_solver,
                    log_success=self.log_success, log_high_alpha_n_failures=log_high_alpha_n_failures
                )
            self.sn = self.model.s(self.wn, Phase.SYMMETRIC)
            self.sm = self.model.s(self.wm, Phase.BROKEN)
            if self.sol_type == SolutionType.DETON:
                self.sp = self.sn
                self.sm_sh = self.sm
            else:
                self.sp = self.model.s(self.wp, Phase.SYMMETRIC)
                self.sm_sh = self.model.s(
                    self.wm_sh,
                    # In detonations the shock and wall have merged
                    Phase.BROKEN if self.sol_type == SolutionType.DETON else Phase.SYMMETRIC
                )

            self.w_center = self.w[0]
            if self.solver_failed:
                # This is already reported by the individual solvers
                msg = f"Solver failed with model={self.model.label_unicode}, " \
                      f"v_wall={self.v_wall}, alpha_n={self.alpha_n}"
                # logger.error(msg)
                self.add_note(msg)
        except (IndexError, RuntimeError) as e:
            msg = f"Solver crashed with model={self.model.label_unicode}, v_wall={self.v_wall}, alpha_n={self.alpha_n}."
            logger.exception(msg, exc_info=e)
            self.add_note(msg)
            self.no_solution_found = True
            return
        self.solved = True
        self.phase = props.find_phase(self.xi, self.v_wall)

        self.alpha_plus = self.model.alpha_plus(
            self.wp, self.wm, vp_tilde=self.vp_tilde, sol_type=self.sol_type,
            error_on_invalid=False, nan_on_invalid=True, log_invalid=True
        )
        self.alpha_theta_bar_plus = self.model.alpha_theta_bar_plus(self.wp)

        # Validity checking for the solution
        if np.isnan(self.alpha_plus):
            self.alpha_plus = self.model.alpha_plus(
                self.wp, self.wm, vp_tilde=self.vp_tilde, sol_type=self.sol_type,
                error_on_invalid=False, nan_on_invalid=False, log_invalid=False
            )
            msg = f"Got invalid alpha_plus={self.alpha_plus} with " \
                  f"model={self.model.label_unicode}, v_wall={self.v_wall}, " \
                  f"alpha_n={self.alpha_n}, sol_type={self.sol_type}."
            logger.error(msg)
            self.add_note(msg)
            self.unphysical_alpha_plus = True
        if self.entropy_flux_p < 0 or self.entropy_flux_m < 0 or self.entropy_flux_diff < 0:
            msg = "Entropy fluxes should not be negative! " \
                f"Got entropy_flux_p={self.entropy_flux_p}, entropy_flux_m={self.entropy_flux_m}, " \
                f"entropy_flux_diff={self.entropy_flux_diff} with " \
                f"model={self.model.label_unicode}, v_wall={self.v_wall}, alpha_n={self.alpha_n}"
            logger.error(msg)
            self.add_note(msg)
            self.negative_entropy_flux = True
        if self.va_entropy_density_diff < 0:
            msg = "Entropy density change should not be negative! Now entropy is decreasing. " \
                  f"Got: {self.va_entropy_density_diff} with " \
                  f"model={self.model.label_unicode}, v_wall={self.v_wall}, alpha_n={self.alpha_n}"
            if log_negative_entropy:
                logger.warning(msg)
            self.add_note(msg)
            self.negative_net_entropy_change = True
        if self.va_thermal_energy_density_diff < 0:
            msg = "Thermal energy density change is negative. The bubble is therefore working as a heat engine. " \
                  f"Got: {self.va_thermal_energy_density_diff}"
            logger.warning(msg)
            self.add_note(msg)
        if not np.isclose(self.kappa + self.omega, 1, rtol=sum_rtol_warning):
            sum_err = not np.isclose(self.kappa + self.omega, 1, rtol=sum_rtol_error)
            if sum_err:
                self.numerical_error = True
            msg = f"Got κ+ω != 1. " + \
                ("Marking the solution to have a numerical error. " if sum_err else "") + \
                f"Got: κ={self.kappa:{error_prec}}, ω={self.omega:{error_prec}}, "\
                f"κ+ω={self.kappa + self.omega:{error_prec}} " \
                f"with model={self.model.label_unicode}, v_wall={self.v_wall}, alpha_n={self.alpha_n}"
            if log_high_alpha_n_failures or (not high_alpha_n) or self.sol_type == SolutionType.DETON:
                if sum_err:
                    logger.error(msg)
                else:
                    logger.warning(msg)
            self.add_note(msg)

    # ---
    # Plotting
    # ---

    def plot(self, fig: plt.Figure = None, path: str = None, **kwargs) -> plt.Figure:
        from pttools.analysis.plot_bubble import plot_bubble
        return plot_bubble(self, fig, path, **kwargs)

    def plot_v(self, fig: plt.Figure = None, ax: plt.Axes = None, path: str = None, **kwargs) -> "FigAndAxes":
        from pttools.analysis.plot_bubble import plot_bubble_v
        return plot_bubble_v(self, fig, ax, path, **kwargs)

    def plot_w(self, fig: plt.Figure = None, ax: plt.Axes = None, path: str = None, **kwargs) -> "FigAndAxes":
        from pttools.analysis.plot_bubble import plot_bubble_w
        return plot_bubble_w(self, fig, ax, path, **kwargs)

    # ---
    # Quantities
    # ---

    @property
    def vp_tilde_sh(self):
        """Velocity in front of the shock in the shock frame

        The fluid ahead of the shock is still, and therefore
        $$\tilde{v}_{+,sh} = v_{sh}$$.
        """
        return self.v_sh

    @property
    def vp_vm_tilde_ratio(self) -> float:
        return self.vp_tilde/self.vm_tilde

    @property
    def vp_vm_tilde_ratio_giese(self) -> float:
        return self.model.vp_vm_tilde_ratio_giese(vp_tilde=self.vp_tilde, vm_tilde=self.vm_tilde, wp=self.wp, wm=self.wm)

    @property
    def vp_vm_tilde_ratio_giese_rel_diff(self) -> float:
        return np.abs(self.vp_vm_tilde_ratio_giese / self.vp_vm_tilde_ratio - 1)

    @property
    def v_mu(self):
        return lorentz(xi=self.v_wall, v=np.sqrt(self.model.cs2(self.wm, Phase.BROKEN)))

    # Quantities
    def en(self) -> float:
        r"""Nucleation energy density $e_n = e(T_n, \phi_s)$"""
        return self.model.e(self.wn, Phase.SYMMETRIC)

    # -----
    # Thermodynamics
    # -----

    # bva = bubble volume averaged

    @functools.cached_property
    def entropy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.entropy_density_diff(self.model, self.w, self.xi, self.v_wall, self.phase)

    @functools.cached_property
    def entropy_density_relative(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return self.entropy_density / self.model.s(self.wn, Phase.SYMMETRIC)

    @functools.cached_property
    def entropy_flux_p(self) -> float:
        r"""Incoming entropy flux at the wall
        $$\tilde{\gamma}_+ \tilde{v}_+ s_+$$
        """
        return gamma(self.vp_tilde) * self.vp_tilde * self.sp

    @functools.cached_property
    def entropy_flux_m(self) -> float:
        r"""Outgoing entropy flux at the wall
        $$\tilde{\gamma}_- \tilde{v}_- s_-$$
        """
        return gamma(self.vm_tilde) * self.vm_tilde * self.sm

    @functools.cached_property
    def entropy_flux_diff(self) -> float:
        r"""Entropy flux difference at the wall
        $$\tilde{\gamma}_- \tilde{v}_- s_- - \tilde{\gamma}_+ \tilde{v}_+ s_+$$
        """
        return self.entropy_flux_m - self.entropy_flux_p

    @functools.cached_property
    def entropy_flux_p_sh(self) -> float:
        r"""Incoming entropy flux at the shock
        $$\tilde{\gamma}_{+,sh} \tilde{v}_{+,sh} s_{+,sh}$$
        """
        return gamma(self.vp_tilde_sh) * self.vp_tilde_sh * self.sn

    @functools.cached_property
    def entropy_flux_m_sh(self) -> float:
        r"""Outgoing entropy flux at the shock
        $$\tilde{\gamma}_{-,sh} \tilde{v}_{-,sh} s_{-,sh}$$"""
        return gamma(self.vm_tilde_sh) * self.vm_tilde_sh * self.sm_sh

    @functools.cached_property
    def entropy_flux_diff_sh(self) -> float:
        r"""Entropy flux difference at the wall
        $$\tilde{\gamma}_{-,sh} \tilde{v}_{-,sh} s_{-,sh} - \tilde{\gamma}_{+,sh} \tilde{v}_{+,sh} s_{+,sh}$$
        """
        return self.entropy_flux_m_sh - self.entropy_flux_p_sh

    @functools.cached_property
    def kinetic_energy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kinetic_energy_density(self.v, self.w, self.xi, self.v_wall)

    @functools.cached_property
    def kinetic_energy_fraction(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kinetic_energy_fraction(ek_bva=self.kinetic_energy_density, eb=self.ebar)

    @functools.cached_property
    def thermal_energy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.thermal_energy_density(v_wall=self.v_wall, eqp=self.va_thermal_energy_density)

    @functools.cached_property
    def thermal_energy_density_diff(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.thermal_energy_density_diff(self.w, self.xi, self.v_wall)

    @functools.cached_property
    def thermal_energy_fraction(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.thermal_energy_fraction(eq_bva=self.thermal_energy_density, eb=self.ebar)

    @functools.cached_property
    def trace_anomaly(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_trace_anomaly(self.model, self.w, self.xi, self.v_wall, self.phase)

    @functools.cached_property
    def ebar(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.ebar(self.model, self.wn)

    @functools.cached_property
    def kappa(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kappa(self.model, self.v, self.w, self.xi, self.v_wall, delta_e_theta=self.va_trace_anomaly_diff)

    @functools.cached_property
    def kappa_giese(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return 4 * self.kinetic_energy_density / (3 * self.alpha_theta_bar_n * self.wn)

    @functools.cached_property
    def mean_adiabatic_index(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.mean_adiabatic_index(self.wbar, self.ebar)

    @functools.cached_property
    def omega(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.omega(self.model, self.w, self.xi, self.v_wall, delta_e_theta=self.va_trace_anomaly_diff)

    @functools.cached_property
    def s(self):
        if not self.solved:
            raise NotYetSolvedError
        return self.model.s(self.w, self.phase)

    @functools.cached_property
    def ubarf2(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.ubarf2(
            self.v, self.w, self.xi,
            self.v_wall, ek_bva=self.kinetic_energy_density)

    # va = volume averaged

    @functools.cached_property
    def va_entropy_density_diff(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_entropy_density_diff(self.model, self.w, self.xi, self.v_wall, self.phase)

    @functools.cached_property
    def va_enthalpy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_enthalpy_density(eq=self.thermal_energy_density)

    @functools.cached_property
    def va_entropy_density_diff_relative(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return self.va_entropy_density_diff / self.model.s(self.wn, Phase.SYMMETRIC)

    @functools.cached_property
    def va_kinetic_energy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_kinetic_energy_density(self.v, self.w, self.xi)

    @functools.cached_property
    def va_kinetic_energy_fraction(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_kinetic_energy_fraction(ek_va=self.va_kinetic_energy_density, eb=self.ebar)

    @functools.cached_property
    def va_thermal_energy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_thermal_energy_density(
            v_shock=self.v_sh, wn=self.wn, ek=self.va_kinetic_energy_density, delta_e_theta=self.va_trace_anomaly_diff)

    @functools.cached_property
    def va_thermal_energy_density_diff(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_thermal_energy_density_diff(self.w, self.xi)

    @functools.cached_property
    def va_thermal_energy_fraction(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_thermal_energy_fraction(eq_va=self.va_thermal_energy_density, eb=self.ebar)

    @functools.cached_property
    def va_trace_anomaly_diff(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_trace_anomaly_diff(self.model, self.w, self.xi, self.v_wall, self.phase)

    @functools.cached_property
    def wbar(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.wbar(self.w, self.xi, self.v_wall, self.wn)
