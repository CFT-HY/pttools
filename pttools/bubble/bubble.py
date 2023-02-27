"""A solution of the hydrodynamic equations"""

import datetime
import functools
import logging
import typing as tp

import numpy as np

from pttools.bubble.boundary import Phase, SolutionType
from pttools.bubble.fluid import fluid_shell_generic
from pttools.bubble import const
from pttools.bubble import thermo
from pttools.bubble import transition
from pttools.speedup.export import export_json
if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


class NotYetSolvedError(RuntimeError):
    """Error for accessing the properties of a bubble that has not been solved yet"""


class Bubble:
    """A solution of the hydrodynamic equations"""
    def __init__(
            self,
            model: "Model", v_wall: float, alpha_n: float,
            sol_type: SolutionType = None,
            label_latex: str = None,
            label_unicode: str = None,
            wn_guess: float = 1,
            wm_guess: float = 2,
            n_points: int = const.N_XI_DEFAULT):
        if v_wall < 0 or v_wall > 1:
            raise ValueError(f"Invalid v_wall={v_wall}")
        if alpha_n < 0 or alpha_n > 1 or alpha_n < model.alpha_n_min:
            raise ValueError(f"Invalid alpha_n={alpha_n}. Minimum for the model: {model.alpha_n_min}")
        self.sol_type = transition.validate_solution_type(
            model,
            v_wall=v_wall, alpha_n=alpha_n, sol_type=sol_type,
            wn_guess=wn_guess, wm_guess=wm_guess
        )

        # Parameters
        self.model: Model = model
        self.v_wall = v_wall
        self.alpha_n = alpha_n
        self.n_points = n_points

        # Computed parameters
        self.wn = model.w_n(alpha_n)
        self.tn = model.temp(self.wn, Phase.SYMMETRIC)
        if self.tn > model.t_crit:
            raise ValueError(f"Bubbles form only when T_nuc < T_crit. Got: T_nuc={self.tn}, T_crit={model.t_crit}")

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

        # Output data
        self.v: tp.Optional[np.ndarray] = None
        self.w: tp.Optional[np.ndarray] = None
        self.xi: tp.Optional[np.ndarray] = None
        self.wp: tp.Optional[float] = None
        self.wm: tp.Optional[float] = None
        self.alpha_plus: tp.Optional[float] = None

        self.gw_power_spectrum = None

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
            "model": self.model.export(),
            "v_wall": self.v_wall,
            "alpha_n": self.alpha_n,
            "sol_type": self.sol_type,
            "v": self.v,
            "w": self.w,
            "xi": self.xi,
            "notes": self.notes
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

    def solve(self, sum_rtol_warning: float = 1e-2, sum_rtol_error: float = 5e-2, error_prec: str = ".4f"):
        if self.solved:
            msg = "Re-solving an already solved bubble! Already computed quantities will not be updated due to caching."
            logger.warning(msg)
            self.add_note(msg)
        try:
            # Todo: make this more specific
            self.v, self.w, self.xi, self.sol_type, self.wp, self.wm, self.solver_failed = fluid_shell_generic(
                model=self.model,
                v_wall=self.v_wall, alpha_n=self.alpha_n, sol_type=self.sol_type, n_xi=self.n_points)
            if self.solver_failed:
                msg = f"Solver failed with model={self.model}, v_wall={self.v_wall}, alpha_n={self.alpha_n}"
                logger.error(msg)
                self.add_note(msg)
        except (IndexError, RuntimeError) as e:
            msg = f"Solver crashed with model={self.model}, v_wall={self.v_wall}, alpha_n={self.alpha_n}."
            logger.exception(msg, exc_info=e)
            self.add_note(msg)
            self.no_solution_found = True
            return
        self.solved = True

        self.alpha_plus = self.model.alpha_plus(self.wp, self.wm)
        if self.alpha_plus >= 1/3:
            msg = f"Got alpha_plus > 1/3 with model={self.model}, v_wall={self.v_wall}, alpha_n={self.alpha_n}. "\
                  f"This is unphysical! Got: {self.alpha_plus}"
            logger.error(msg)
            self.add_note(msg)
            self.unphysical_alpha_plus = True
        if self.entropy_density < 0:
            msg = f"Entropy density should not be negative! Now entropy is decreasing. Got: {self.entropy_density}"
            logger.error(msg)
            self.add_note(msg)
            self.unphysical_entropy = True
        if self.thermal_energy_density < 0:
            msg = "Thermal energy density is negative. The bubble is therefore working as a heat engine. "\
                  f"Got: {self.thermal_energy_density}"
            logger.warning(msg)
            self.add_note(msg)
        if not np.isclose(self.kappa + self.omega, 1, rtol=sum_rtol_warning):
            sum_err = not np.isclose(self.kappa + self.omega, 1, rtol=sum_rtol_error)
            if sum_err:
                self.numerical_error = True
            msg = "κ+ω != 1. " + \
                ("Marking the solution to have a numerical error." if sum_err else "") + \
                " Got: " \
                f"κ={self.kappa:{error_prec}}, ω={self.omega:{error_prec}}, κ+ω={self.kappa + self.omega:{error_prec}}"
            logger.error(msg)
            self.add_note(msg)

    def spectrum(self):
        raise NotImplementedError

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
