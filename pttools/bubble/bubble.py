"""A solution of the hydrodynamic equations"""
import abc
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
from pttools.bubble.relativity import gamma
from pttools.bubble import thermo
from pttools.bubble import transition
import pttools.type_hints as th
from pttools.utils.docstrings import copy_docstrings
from pttools.utils.json import export_json
if tp.TYPE_CHECKING:
    from pttools.models.model import Model
    from pttools.models.const_cs import ConstCSModel
    from pttools.analysis.utils import FigAndAxes

logger = logging.getLogger(__name__)


class NotYetSolvedError(RuntimeError):
    """Error for accessing the properties of a bubble that has not been solved yet"""


class BaseBubble(abc.ABC):
    """A common base class for bubbles and droplets"""
    def __init__(
            self,
            model: "Model",
            v_wall: float,
            alpha_n: float,
            w_center: float | None = None,
            wm_guess: float | None = None,
            t_end: float = const.T_END_DEFAULT,
            n_xi: int = const.N_XI_DEFAULT):
        # Some functions such as np.vectorize tend to give 0D arrays, which may cause subtle errors later on.
        if v_wall is None or not np.isscalar(v_wall):
            raise ValueError(f"v_wall should be scalar. Did you give e.g. a 0D array instead? Got: v_wall={v_wall}")
        if isinstance(v_wall, int):
            v_wall = float(v_wall)
        if alpha_n is None or not np.isscalar(alpha_n):
            raise ValueError(f"alpha_n should be scalar. Did you give e.g. a 0D array instead? Got: alpha_n={v_wall}")
        if not (w_center is None or np.isscalar(w_center)):
            raise ValueError(
                f"w_center should be scalar. Did you give e.g. a 0D array instead? Got: w_center={w_center}"
            )
        if not (wm_guess is None or np.isscalar(wm_guess)):
            raise ValueError(
                f"wm_guess should be scalar. Did you give e.g. a 0D array instead? Got: wm_guess={wm_guess}"
            )

        # -----
        # Set parameters
        # -----
        self.model: Model = model
        self.v_wall: float = v_wall
        self.t_end: float = t_end
        self.n_xi: int = n_xi
        #: $w_\text{center}$
        self.w_center: float | None = w_center
        #: $w_{-,\text{guess}}$
        self.wm_guess: float | None = wm_guess

        # -----
        # Output arrays
        # -----
        #: Fluid velocity profile $v(\xi)$
        self.v: th.FloatArr1D | None = None
        #: Enthalpy profile $w(\xi)$
        self.w: th.FloatArr1D | None = None
        #: Self-similar droplet radius coordinates $\xi$
        self.xi: th.FloatArr1D | None = None
        #: Phase profile $\phi(\xi)$
        self.phase: th.FloatArr1D | None = None
        #: Temperature profile $T(\xi)$
        self.T: th.FloatArr1D | None = None

        # -----
        # Output values
        # -----
        self.label_latex: str
        self.label_unicode: str
        self.solution_found: bool | None = None
        self.solved: bool | None = None
        self.vm: float | None = None
        self.wm: float | None = None
        self.notes: list[str] = []
        #: $s_+$
        self.sp: float | None = None
        #: $s_-$
        self.sm: float | None = None
        #: $s_{-,\text{sh}}$
        self.sm_sh: float | None = None
        #: $s_n$
        self.sn: float | None = None
        #: $T_+$
        self.Tp: float | None = None
        #: $T_-$
        self.Tm: float | None = None
        #: $T_\text{center}$
        self.T_center: float | None = None
        #: $v_+$
        self.vp: float | None = None
        #: $v_-$
        self.vm: float | None = None
        #: $\tilde{v}_+$
        self.vp_tilde: float | None = None
        #: $\tilde{v}_-$
        self.vm_tilde: float | None = None
        #: $w_+$
        self.wp: float | None = None
        #: $w_-$
        self.wm: float | None = None

        # Flags
        self.no_solution_found: bool = False
        self.solved: bool = False

    def add_note(self, note: str):
        """Add a note to the solution"""
        self.notes.append(note)

    @abc.abstractmethod
    def solve(self):
        if self.solved:
            msg = (
                "Re-solving an already solved fluid profile! "
                "Already computed quantities will not be updated due to caching."
            )
            logger.warning(msg)
            self.add_note(msg)

    # -----
    # Plotting
    # -----

    def plot(
            self,
            fig: plt.Figure | None = None,
            path: str | None = None,
            **kwargs) -> plt.Figure:
        """Plot the velocity and enthalpy profiles of the bubble"""
        from pttools.analysis.plot_bubbles import plot_bubbles
        return plot_bubbles([self], fig, path, **kwargs)

    def plot_v(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | None = None,
            **kwargs) -> "FigAndAxes":
        """Plot the velocity profile of the bubble"""
        from pttools.analysis.plot_bubbles import plot_bubbles_v
        return plot_bubbles_v([self], fig, ax, path, **kwargs)

    def plot_w(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | None = None,
            **kwargs) -> "FigAndAxes":
        """Plot the enthalpy profile of the bubble"""
        from pttools.analysis.plot_bubbles import plot_bubbles_w
        return plot_bubbles_w([self], fig, ax, path, **kwargs)

    # -----
    # Quantities
    # -----

    @functools.cached_property
    def entropy_flux_p(self) -> float:
        r"""Incoming entropy flux at the wall
        $$\tilde{\gamma}_+ \tilde{v}_+ s_+$$
        """
        if not self.solved:
            raise NotYetSolvedError
        return gamma(self.vp_tilde) * self.vp_tilde * self.sp

    @functools.cached_property
    def entropy_flux_m(self) -> float:
        r"""Outgoing entropy flux at the wall
        $$\tilde{\gamma}_- \tilde{v}_- {s}_- $$
        """
        if not self.solved:
            raise NotYetSolvedError
        return gamma(self.vm_tilde) * self.vm_tilde * self.sm

    @functools.cached_property
    def entropy_flux_diff(self) -> float:
        r"""Entropy flux difference at the wall
        $$\tilde{\gamma}_- \tilde{v}_- {s}_- - \tilde{\gamma}_+ \tilde{v}_+ {s}_+ $$
        """
        if not self.solved:
            raise NotYetSolvedError
        return self.entropy_flux_m - self.entropy_flux_p

    @functools.cached_property
    def p(self):
        r"""Pressure $p(\xi)$"""
        if not self.solved:
            raise NotYetSolvedError
        return self.model.p(self.w, self.phase)

    @functools.cached_property
    def s(self):
        r"""Entropy density $s(\xi)$"""
        if not self.solved:
            raise NotYetSolvedError
        return self.model.s(self.w, self.phase)

    @functools.cached_property
    def va_kinetic_energy_density(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_kinetic_energy_density(self.v, self.w, self.xi)

    @property
    def vp_vm_tilde_ratio(self) -> float:
        r"""$$\frac{\tilde{v}_+}{\tilde{v}_-}$$"""
        return self.vp_tilde / self.vm_tilde


class Bubble(BaseBubble):
    """A solution of the hydrodynamic equations, aka. a bubble"""
    def __init__(
            self,
            model: "Model",
            v_wall: float,
            alpha_n: float,
            solve: bool = True,
            sol_type: SolutionType | None = None,
            label_latex: str | None = None,
            label_unicode: str | None = None,
            wn_guess: float | None = None,
            wm_guess: float | None = None,
            theta_bar: bool = False,
            t_end: float = const.T_END_DEFAULT,
            n_xi: int = const.N_XI_DEFAULT,
            thin_shell_t_points_min: int = const.THIN_SHELL_T_POINTS_MIN,
            use_bag_solver: bool = False,
            use_giese_solver: bool = False,
            log_success: bool = False,
            allow_invalid: bool = False,
            log_invalid: bool = True):
        r"""Create a solution of the hydrodynamic equations, aka. a bubble
        :param model: The equation of state object
        :param v_wall: Wall velocity $v_\text{wall}$
        :param alpha_n: Transition strength $\alpha_n$
        :param solve: Whether to solve the bubble immediately
        :param sol_type: Solution type (deflagration, hybrid, detonation).
            If None, it will be determined automatically.
        :param label_latex: LaTeX label for plots
        :param label_unicode: Unicode label for plots
        :param wn_guess: Initial guess for the enthalpy at the nucleation temperature $w_n$
        :param wm_guess: Initial guess for the enthalpy behind the wall $w_-$
        :param theta_bar: Whether the provided alpha_n is actually alpha_theta_bar_n
        :param t_end: The maximum value for the fluid shell ODE integration parameter
        :param n_xi: Number of points in the fluid velocity profile
        :param thin_shell_t_points_min: Limit of points for a shell to be so thin
            that it should be re-computed with more points
        :param use_bag_solver: Whether to use the bag model specific fluid shell solver
        :param use_giese_solver: Whether to use the Giese et al. solver for the constant sound speed model
        :param log_success: Whether to log successful solutions
        :param allow_invalid: Whether to allow invalid solutions
        :param log_invalid: Whether to log invalid solutions
        """
        super().__init__(model=model, v_wall=v_wall, alpha_n=alpha_n, wm_guess=wm_guess, t_end=t_end, n_xi=n_xi)

        # -----
        # Validate input parameters
        # -----
        if use_bag_solver and use_giese_solver:
            raise ValueError("Both bag and Giese et al. solvers cannot be used at the same time.")
        if not 0 < self.v_wall <= 1:
            raise ValueError(f"Invalid v_wall={self.v_wall}")

        # -----
        # Set and validate alpha_n and alpha_theta_bar_n
        # -----
        if isinstance(alpha_n, int):
            alpha_n = float(alpha_n)
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
            v_wall=self.v_wall, alpha_n=alpha_n, sol_type=sol_type,
            wn=self.wn, wm_guess=wm_guess
        )

        # -----
        # Set parameters
        # -----
        self.thin_shell_t_points_min = thin_shell_t_points_min
        self.log_success = log_success

        # -----
        # Compute parameters
        # -----
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
            #     logger.warning(
            #         "alpha_theta_bar_n=%s < lte_min=%s",
            #         self.alpha_theta_bar_n, self.alpha_theta_bar_n_min_lte
            #     )
            # if log_invalid and self.alpha_theta_bar_n > self.alpha_theta_bar_n_max_lte:
            #     logger.warning(
            #         "alpha_theta_bar_n=%s > lte_max=%s",
            #         self.alpha_theta_bar_n, self.alpha_theta_bar_n_max_lte
            #     )

        # Here LTE = no entropy generation
        # if log_invalid and self.sol_type == SolutionType.DETON and self.Psi_n < 0.75:
        #     logger.info(
        #         "This detonation may not exist, as LTE predicts a large alpha_n_hyb_max for Psi_n=%s < 0.75. "
        #         "Please see Ai et al. (2023), p. 15.",
        #         self.Psi_n
        #     )

        # Flags
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
        self.label_latex = rf"{self.model.label_latex}, $v_w={v_wall:.3f}, \alpha_n={alpha_n:.3f}$" \
            if label_latex is None else label_latex
        self.label_unicode = f"{self.model.label_unicode}, v_w={v_wall:3f}, αₙ={alpha_n:.3f}" \
            if label_unicode is None else label_unicode

        # -----
        # Output values
        # -----
        #: $\alpha_+$
        self.alpha_plus: float | None = None
        #: $\alpha_{\bar{\theta}_+}$
        self.alpha_theta_bar_plus: float | None = None
        self.elapsed: float | None = None
        #: $T_{-,\text{sh}}$
        self.Tm_sh: float | None = None
        #: $v_{\text{sh}}$
        self.v_sh: float | None = None
        #: $\tilde{v}_{-,\text{sh}}$
        self.vm_sh: float | None = None
        #: $\tilde{v}_{-,\text{sh}}$
        self.vm_tilde_sh: float | None = None
        #: $v_{CJ}$
        self.v_cj: float | None = None
        #: $w_{-,\text{sh}}$
        self.wm_sh: float | None = None

        if solve:
            self.solve()
        elif log_success:
            logger.info(
                "Initialized a bubble with: "
                "model=%s, v_w=%s, alpha_n=%s, T_nuc=%s, w_nuc=%s",
                self.model.label_unicode, v_wall, alpha_n, self.Tn, self.wn
            )

    def export(self, path: str | None = None) -> dict[str, tp.Any]:
        """Export the bubble data as JSON"""
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
            "T": self.T,
            # Solution parameters
            "alpha_plus": self.alpha_plus,
            "sp": self.sp,
            "sm": self.sm,
            "sm_sh": self.sm_sh,
            "sn": self.sn,
            "Tp": self.Tp,
            "Tm": self.Tm,
            "T_center": self.T_center,
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
            "wm_sh": self.wm_sh,
            "w_center": self.w_center
        }
        if path is not None:
            export_json(data, path)
        return data

    def info_str(self, prec: str = ".4f") -> str:
        """Get a string describing the key quantities of the bubble"""
        return (
            f"{self.label_unicode}: w0/wn={self.w[0] / self.wn:{prec}}, "
            f"Ubarf2={self.ubarf2:{prec}}, K={self.kinetic_energy_fraction:{prec}}, "
            f"κ={self.kappa:{prec}}, ω={self.omega:{prec}}, κ+ω={self.kappa + self.omega:{prec}}, "
            f"V-avg. trace anomaly={self.va_trace_anomaly_diff:{prec}}"
        )

    def solve(
            self,
            sum_rtol_warning: float = 1.5e-2,
            sum_rtol_error: float = 5e-2,
            error_prec: str = ".4f",
            use_bag_solver: bool = False,
            use_giese_solver: bool = False,
            log_high_alpha_n_failures: bool = True,
            log_negative_entropy: bool = True):
        """Simulate the fluid velocity profile of the bubble"""
        super().solve()

        use_bag_solver = self.use_bag_solver or use_bag_solver
        use_giese_solver = self.use_giese_solver or use_giese_solver
        if use_bag_solver and use_giese_solver:
            raise ValueError("Both bag and Giese et al. solvers cannot be used at the same time.")

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
                    high_alpha_n=high_alpha_n,
                    t_end=self.t_end,
                    n_xi=self.n_xi,
                    thin_shell_limit=self.thin_shell_t_points_min,
                    use_bag_solver=use_bag_solver,
                    use_giese_solver=use_giese_solver,
                    log_success=self.log_success,
                    log_high_alpha_n_failures=log_high_alpha_n_failures
                )
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

        # -----
        # Additional properties for the solution
        # -----

        self.solved = True
        self.alpha_plus = self.model.alpha_plus(
            self.wp, self.wm, vp_tilde=self.vp_tilde, sol_type=self.sol_type,
            error_on_invalid=False, nan_on_invalid=True, log_invalid=True
        )
        self.alpha_theta_bar_plus = self.model.alpha_theta_bar_plus(self.wp)
        self.phase = props.find_phase(self.xi, self.v_wall)

        self.sn = self.model.s(self.wn, Phase.SYMMETRIC)
        self.sm = self.model.s(self.wm, Phase.BROKEN)
        self.Tm = self.model.temp(self.wm, Phase.BROKEN)
        self.w_center = self.w[0]
        self.T_center = self.model.temp(self.w_center, Phase.BROKEN)

        # In detonations the shock and the wall have merged
        if self.sol_type == SolutionType.DETON:
            self.sp = self.sn
            self.sm_sh = self.sm
            self.Tp = self.Tn
            self.Tm_sh = self.Tm
        else:
            self.sp = self.model.s(self.wp, Phase.SYMMETRIC)
            self.sm_sh = self.model.s(self.wm_sh, Phase.SYMMETRIC)
            self.Tp = self.model.temp(self.wp, Phase.SYMMETRIC)
            self.Tm_sh = self.model.temp(self.wm_sh, Phase.SYMMETRIC)

        # -----
        # Validity checking for the solution
        # -----

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
            msg = "Got κ+ω != 1. " + \
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

    # -----
    # Quantities
    # -----

    @property
    def vp_tilde_sh(self):
        r"""Velocity in front of the shock in the shock frame

        The fluid ahead of the shock is still, and therefore
        $$\tilde{v}_{+,sh} = v_{sh}$$.
        """
        return self.v_sh

    @property
    def vp_vm_tilde_ratio_giese(self) -> float:
        # This docstring is copied from the model function
        r"""Giese et al. approximation for $\frac{\tilde{v}_+}{\tilde{v}_-}$,
        :giese_2021:`\ ` eq. 11

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
        r"""
        Relative difference of the ratio of the exact and approximate
        $\tilde{v}_+, \tilde{v}_-$ ratios from unity
        """
        return np.abs(self.vp_vm_tilde_ratio_giese / self.vp_vm_tilde_ratio - 1)

    @property
    def v_mu(self) -> float:
        r"""Maximum fluid velocity behind the bubble wall, $\mu(\xi)$"""
        # wm is the highest enthalpy inside the bubble
        cs2, _ = self.model.cs2_max(w_max=self.wm, w_min=self.w_center, phase=Phase.BROKEN)
        return props.v_max_behind(self.v_wall, np.sqrt(cs2))

    # Quantities
    def en(self) -> float:
        r"""Nucleation energy density $e_n = e(T_n, \phi_s)$"""
        return self.model.e(self.wn, Phase.SYMMETRIC)

    # -----
    # Thermodynamics
    # -----

    # bva = bubble volume averaged

    @functools.cached_property
    def entropy_density_diff(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.entropy_density_diff(self.model, self.w, self.xi, self.v_wall, self.phase)

    @functools.cached_property
    def entropy_density_diff_relative(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return self.entropy_density_diff / self.model.s(self.wn, Phase.SYMMETRIC)

    @functools.cached_property
    def entropy_flux_p_sh(self) -> float:
        r"""Incoming entropy flux at the shock
        $$\tilde{\gamma}_{+,sh} \tilde{v}_{+,sh} s_{+,sh}$$
        """
        if not self.solved:
            raise NotYetSolvedError
        return gamma(self.vp_tilde_sh) * self.vp_tilde_sh * self.sn

    @functools.cached_property
    def entropy_flux_m_sh(self) -> float:
        r"""Outgoing entropy flux at the shock
        $$\tilde{\gamma}_{-,sh} \tilde{v}_{-,sh} s_{-,sh}$$"""
        if not self.solved:
            raise NotYetSolvedError
        return gamma(self.vm_tilde_sh) * self.vm_tilde_sh * self.sm_sh

    @functools.cached_property
    def entropy_flux_diff_sh(self) -> float:
        r"""Entropy flux difference at the wall
        $$\tilde{\gamma}_{-,sh} \tilde{v}_{-,sh} s_{-,sh} - \tilde{\gamma}_{+,sh} \tilde{v}_{+,sh} s_{+,sh}$$
        """
        if not self.solved:
            raise NotYetSolvedError
        return self.entropy_flux_m_sh - self.entropy_flux_p_sh

    @functools.cached_property
    def kinetic_energy_density(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kinetic_energy_density(self.v, self.w, self.xi, self.v_wall)

    @functools.cached_property
    def kinetic_energy_fraction(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kinetic_energy_fraction(ek_bva=self.kinetic_energy_density, eb=self.ebar)

    @functools.cached_property
    def thermal_energy_density(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.thermal_energy_density(v_wall=self.v_wall, eqp=self.va_thermal_energy_density)

    @functools.cached_property
    def thermal_energy_density_diff(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.thermal_energy_density_diff(self.w, self.xi, self.v_wall)

    @functools.cached_property
    def thermal_energy_fraction(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.thermal_energy_fraction(eq_bva=self.thermal_energy_density, eb=self.ebar)

    @functools.cached_property
    def trace_anomaly(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.trace_anomaly_diff(self.model, self.w, self.xi, self.v_wall, self.phase)

    @functools.cached_property
    def ebar(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.ebar(self.model, self.wn)

    @functools.cached_property
    def kappa(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kappa(self.model, self.v, self.w, self.xi, self.v_wall, delta_e_theta=self.va_trace_anomaly_diff)

    @functools.cached_property
    def kappa_giese(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return 4 * self.kinetic_energy_density / (3 * self.alpha_theta_bar_n * self.wn)

    @functools.cached_property
    def mean_adiabatic_index(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.mean_adiabatic_index(self.wbar, self.ebar)

    @functools.cached_property
    def nu_gdh2024(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return self.model.nu_gdh2024(self.va_enthalpy_density)

    @functools.cached_property
    def omega(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.omega(self.model, self.w, self.xi, self.v_wall, delta_e_theta=self.va_trace_anomaly_diff)

    @functools.cached_property
    def omega_barotropic(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return self.model.omega(self.va_enthalpy_density, Phase.BROKEN)

    @functools.cached_property
    def ubarf(self) -> float:
        r"""Enthalpy-weighted RMS fluid velocity $\bar{U}_\text{f}$"""
        return np.sqrt(self.ubarf2)

    @functools.cached_property
    def ubarf2(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.ubarf2(
            self.v, self.w, self.xi,
            self.v_wall, ek_bva=self.kinetic_energy_density
        )

    # va = volume averaged

    @functools.cached_property
    def va_enthalpy_density(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_enthalpy_density(eq=self.thermal_energy_density)

    @functools.cached_property
    def va_entropy_density_diff(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_entropy_density_diff(self.model, self.w, self.xi, self.v_wall, self.phase)

    @functools.cached_property
    def va_entropy_density_diff_relative(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return self.va_entropy_density_diff / self.model.s(self.wn, Phase.SYMMETRIC)

    @functools.cached_property
    def va_kinetic_energy_fraction(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_kinetic_energy_fraction(ek_va=self.va_kinetic_energy_density, eb=self.ebar)

    @functools.cached_property
    def va_thermal_energy_density(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_thermal_energy_density(
            v_shock=self.v_sh, wn=self.wn, ek=self.va_kinetic_energy_density, delta_e_theta=self.va_trace_anomaly_diff)

    @functools.cached_property
    def va_thermal_energy_density_diff(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_thermal_energy_density_diff(self.w, self.xi)

    @functools.cached_property
    def va_thermal_energy_fraction(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_thermal_energy_fraction(eq_va=self.va_thermal_energy_density, eb=self.ebar)

    @functools.cached_property
    def va_trace_anomaly_diff(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.va_trace_anomaly_diff(self.model, self.w, self.xi, self.v_wall, self.phase)

    @functools.cached_property
    def wbar(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return thermo.wbar(self.w, self.xi, self.v_wall, self.wn)


copy_docstrings({
    Bubble.entropy_density_diff: thermo.entropy_density_diff,
    Bubble.kinetic_energy_density: thermo.kinetic_energy_density,
    Bubble.kinetic_energy_fraction: thermo.kinetic_energy_fraction,
    Bubble.thermal_energy_density: thermo.thermal_energy_density,
    Bubble.thermal_energy_density_diff: thermo.thermal_energy_density_diff,
    Bubble.thermal_energy_fraction: thermo.thermal_energy_fraction,
    Bubble.ebar: thermo.ebar,
    Bubble.kappa: thermo.kappa,
    Bubble.mean_adiabatic_index: thermo.mean_adiabatic_index,
    Bubble.omega: thermo.omega,
    Bubble.ubarf2: thermo.ubarf2,
    Bubble.va_enthalpy_density: thermo.va_enthalpy_density,
    Bubble.va_entropy_density_diff: thermo.va_entropy_density_diff,
    Bubble.va_kinetic_energy_density: thermo.va_kinetic_energy_density,
    Bubble.va_kinetic_energy_fraction: thermo.va_kinetic_energy_fraction,
    Bubble.va_thermal_energy_density: thermo.va_thermal_energy_density,
    Bubble.va_thermal_energy_density_diff: thermo.va_thermal_energy_density_diff,
    Bubble.va_thermal_energy_fraction: thermo.va_thermal_energy_fraction,
    Bubble.va_trace_anomaly_diff: thermo.va_trace_anomaly_diff,
    Bubble.wbar: thermo.wbar
}, without_params=True)
