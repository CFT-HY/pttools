"""A solution of the hydrodynamic equations"""

import datetime
import functools
import logging
import typing as tp

import numpy as np

from pttools.bubble.alpha import alpha_n_max_deflagration_bag
from pttools.bubble.boundary import Phase, SolutionType
from pttools.bubble.fluid import fluid_shell_generic
from pttools.bubble import const
from pttools.bubble import thermo
from pttools.bubble import transition
from pttools.speedup.export import export_json
if tp.TYPE_CHECKING:
    from pttools.models.model import Model
    from pttools.models.const_cs import ConstCSModel

logger = logging.getLogger(__name__)


class NotYetSolvedError(RuntimeError):
    """Error for accessing the properties of a bubble that has not been solved yet"""


class Bubble:
    """A solution of the hydrodynamic equations"""
    def __init__(
            self,
            model: "Model", v_wall: float, alpha_n: float,
            solve: bool = False,
            sol_type: SolutionType = None,
            label_latex: str = None,
            label_unicode: str = None,
            wn_guess: float = None,
            wm_guess: float = None,
            n_points: int = const.N_XI_DEFAULT,
            log_success: bool = False):
        if v_wall < 0 or v_wall > 1:
            raise ValueError(f"Invalid v_wall={v_wall}")
        if alpha_n < 0 or alpha_n > 1 or alpha_n < model.alpha_n_min:
            raise ValueError(f"Invalid alpha_n={alpha_n}. Minimum for the model: {model.alpha_n_min}")

        self.wn = model.w_n(alpha_n, wn_guess)
        self.sol_type = transition.validate_solution_type(
            model,
            v_wall=v_wall, alpha_n=alpha_n, sol_type=sol_type,
            wn=self.wn, wm_guess=wm_guess
        )

        # Parameters
        self.model: Model = model
        self.v_wall = v_wall
        self.alpha_n = alpha_n
        self.n_points = n_points
        self.log_success = log_success

        # Computed parameters
        self.tn = model.temp(self.wn, Phase.SYMMETRIC)
        if self.tn > model.t_crit:
            raise ValueError(f"Bubbles form only when T_nuc < T_crit. Got: T_nuc={self.tn}, T_crit={model.t_crit}")

        # if isinstance(model, ConstCSModel)
        if hasattr(model, "css2") and hasattr(model, "csb2"):
            model: ConstCSModel
            self.alpha_n_bar = model.alpha_n_bar(alpha_n)
            self.alpha_n_bar_min_lte = model.alpha_n_bar_min_lte(self.wn, self.sol_type)
            self.alpha_n_bar_max_lte = model.alpha_n_bar_max_lte(self.wn, self.sol_type)
            if self.alpha_n_bar_max_lte < self.alpha_n_bar_min_lte:
                raise RuntimeError(
                    "Got invalid limits for alpha_n_bar_lte: "
                    f"min={self.alpha_n_bar_min_lte}, max={self.alpha_n_bar_max_lte}"
                )
            if self.alpha_n_bar < self.alpha_n_bar_min_lte:
                logger.warning("alpha_n_bar=%s < lte_min=%s", self.alpha_n_bar, self.alpha_n_bar_min_lte)
            if self.alpha_n_bar > self.alpha_n_bar_max_lte:
                logger.warning("alpha_n_bar=%s > lte_max=%s", self.alpha_n_bar, self.alpha_n_bar_max_lte)

        self.psi_n = model.psi_n(self.wn)
        if self.sol_type == SolutionType.DETON and self.psi_n < 0.75:
            logger.warning(
                "This detonation should not exist, as LTE predicts a large alpha_n_hyb_max for psi_n=%s < 0.75. "
                "Please see Ai et al. (2023), p. 15.",
                self.psi_n
            )

        # Flags
        self.solved = False
        self.solver_failed = False
        self.no_solution_found = False
        self.numerical_error = False
        self.unphysical_alpha_plus = False
        self.unphysical_entropy = False

        # LaTeX labels are not supported in Plotly 3D plots.
        # https://github.com/plotly/plotly.js/issues/608
        self.label_latex = rf"{self.model.label_latex} $v_w={v_wall}, \alpha_n={alpha_n}" \
            if label_latex is None else label_latex
        self.label_unicode = f"{self.model.label_unicode}, v_w={v_wall}, αₙ={alpha_n}" \
            if label_unicode is None else label_unicode
        self.notes: tp.List[str] = []

        # Output arrays
        self.v: tp.Optional[np.ndarray] = None
        self.w: tp.Optional[np.ndarray] = None
        self.xi: tp.Optional[np.ndarray] = None

        # Output values
        self.vp: tp.Optional[float] = None
        self.vm: tp.Optional[float] = None
        self.vp_tilde: tp.Optional[float] = None
        self.vm_tilde: tp.Optional[float] = None
        self.v_sh: tp.Optional[float] = None
        self.vm_sh: tp.Optional[float] = None
        self.vm_tilde_sh: tp.Optional[float] = None
        self.v_cj: tp.Optional[float] = None
        self.wp: tp.Optional[float] = None
        self.wm: tp.Optional[float] = None
        self.wm_sh: tp.Optional[float] = None
        self.alpha_plus: tp.Optional[float] = None
        self.elapsed: tp.Optional[float] = None

        self.gw_power_spectrum = None

        if solve:
            self.solve()
        elif log_success:
            logger.info(
                "Initialized a bubble with: "
                "model=%s, v_w=%s, alpha_n=%s, T_nuc=%s, w_nuc=%s",
                self.model.label_unicode, v_wall, alpha_n, self.tn, self.wn
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
            # Solution
            "v": self.v,
            "w": self.w,
            "xi": self.xi,
            # Solution parameters
            "tn": self.tn,
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
            "alpha_plus": self.alpha_plus,
            "v_cj": self.v_cj
        }
        if path is not None:
            export_json(data, path)
        return data

    def info_str(self, prec: str = ".4f") -> str:
        return \
            f"{self.label_unicode}: w0/wn={self.w[0] / self.wn:{prec}}, " \
            f"Ubarf2={self.ubarf2:{prec}}, K={self.kinetic_energy_fraction:{prec}}, " \
            f"κ={self.kappa:{prec}}, ω={self.omega:{prec}}, κ+ω={self.kappa + self.omega:{prec}}, " \
            f"trace anomaly={self.trace_anomaly:{prec}}"

    def solve(
            self,
            sum_rtol_warning: float = 1.5e-2,
            sum_rtol_error: float = 5e-2,
            error_prec: str = ".4f",
            use_bag_solver: bool = False,
            log_high_alpha_n_failures: bool = True,
            log_negative_entropy: bool = True):
        if self.solved:
            msg = "Re-solving an already solved bubble! Already computed quantities will not be updated due to caching."
            logger.warning(msg)
            self.add_note(msg)

        alpha_n_max_bag = alpha_n_max_deflagration_bag(self.v_wall)
        high_alpha_n = alpha_n_max_bag - self.alpha_n < 0.05

        try:
            # Todo: make the solver errors more specific
            self.v, self.w, self.xi, self.sol_type, \
                self.vp, self.vm, self.vp_tilde, self.vm_tilde, \
                self.v_sh, self.vm_sh, self.vm_tilde_sh, \
                self.wp, self.wm, self.wm_sh, self.v_cj, self.solver_failed, self.elapsed = \
                fluid_shell_generic(
                    model=self.model,
                    v_wall=self.v_wall, alpha_n=self.alpha_n, sol_type=self.sol_type,
                    wn=self.wn,
                    alpha_n_max_bag=alpha_n_max_bag,
                    high_alpha_n=high_alpha_n, n_xi=self.n_points,
                    use_bag_solver=use_bag_solver,
                    log_success=self.log_success, log_high_alpha_n_failures=log_high_alpha_n_failures
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
        self.solved = True

        # Validity checking for the solution
        self.alpha_plus = self.model.alpha_plus(self.wp, self.wm)
        if self.alpha_plus >= 1/3 and self.sol_type != SolutionType.DETON:
            msg = "Got alpha_plus > 1/3 with " \
                  f"model={self.model.label_unicode}, v_wall={self.v_wall}, " \
                  f"alpha_n={self.alpha_n}, sol_type={self.sol_type}. " \
                  f"This is unphysical! Got: {self.alpha_plus}"
            logger.error(msg)
            self.add_note(msg)
            self.unphysical_alpha_plus = True
        if self.entropy_density < 0:
            msg = "Entropy density should not be negative! Now entropy is decreasing. " \
                  f"Got: {self.entropy_density} with " \
                  f"model={self.model.label_unicode}, v_wall={self.v_wall}, alpha_n={self.alpha_n}"
            if log_negative_entropy:
                logger.warning(msg)
            self.add_note(msg)
            self.unphysical_entropy = True
        if self.thermal_energy_density < 0:
            msg = "Thermal energy density is negative. The bubble is therefore working as a heat engine. " \
                  f"Got: {self.thermal_energy_density}"
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

    def spectrum(self):
        raise NotImplementedError

    @property
    def vp_tilde_sh(self):
        """Velocity in front of the shock in the shock frame

        The fluid ahead of the shock is still, and therefore
        $$\tilde{v}_{+,sh} = v_{sh}$$.
        """
        return self.v_sh

    # -----
    # Thermodynamics
    # -----

    @functools.cached_property
    def ebar(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.ebar(self.model, self.wn)

    @functools.cached_property
    def entropy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.entropy_density(self.model, self.w, self.xi, self.v_wall)

    @functools.cached_property
    def entropy_density_relative(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return self.entropy_density / self.model.s(self.wn, Phase.SYMMETRIC)

    @functools.cached_property
    def kappa(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kappa(self.model, self.v, self.w, self.xi, self.v_wall, delta_e_theta=self.trace_anomaly)

    @functools.cached_property
    def kinetic_energy_fraction(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kinetic_energy_fraction(
            self.model, self.v, self.w, self.xi,
            self.v_wall, ek=self.kinetic_energy_density)

    @functools.cached_property
    def kinetic_energy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.kinetic_energy_density(self.v, self.w, self.xi, self.v_wall)

    @functools.cached_property
    def mean_adiabatic_index(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.mean_adiabatic_index(self.wbar, self.ebar)

    @functools.cached_property
    def omega(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.omega(self.model, self.w, self.xi, self.v_wall, self.trace_anomaly)

    @functools.cached_property
    def thermal_energy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.thermal_energy_density(self.w, self.xi, self.v_wall)

    @functools.cached_property
    def trace_anomaly(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.trace_anomaly(self.model, self.w, self.xi, self.v_wall)

    @functools.cached_property
    def ubarf2(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.ubarf2(
            self.v, self.w, self.xi,
            self.v_wall, ek=self.kinetic_energy_density, wn=self.wn)  # wb=self.wbar

    @functools.cached_property
    def wbar(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.wbar(self.w, self.xi, self.v_wall, self.wn)
