import datetime
import functools
import logging
import typing as tp

import numpy as np

from pttools.bubble.boundary import Phase, SolutionType
from pttools.bubble.fluid import fluid_shell_generic
from pttools.bubble import thermo
from pttools.bubble import transition
from pttools.speedup.export import export_json
if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


class NotYetSolvedError(RuntimeError):
    """Error for accessing the properties of a bubble that has not been solved yet"""


class Bubble:
    """A solution of the hydrodynamic equations

    (Defined in the meeting with Hindmarsh on 2022-07-06.)
    """
    def __init__(
            self,
            model: "Model", v_wall: float, alpha_n: float,
            sol_type: SolutionType = None,
            label_latex: str = None,
            label_unicode: str = None,
            wn_guess: float = 1,
            wm_guess: float = 2):
        if v_wall < 0 or v_wall > 1:
            raise ValueError(f"Invalid v_wall={v_wall}")
        if alpha_n < 0 or alpha_n > 1 or alpha_n < model.alpha_n_min:
            raise ValueError(f"Invalid alpha_n={alpha_n}. Minimum for the model: {model.alpha_n_min}")
        sol_type = transition.validate_solution_type(
            model,
            v_wall=v_wall, alpha_n=alpha_n, sol_type=sol_type,
            wn_guess=wn_guess, wm_guess=wm_guess
        )

        self.model: Model = model
        self.v_wall = v_wall
        self.alpha_n = alpha_n
        self.sol_type = sol_type
        self.solved = False
        # The labels are defined without LaTeX, as it's not supported in Plotly 3D plots.
        # https://github.com/plotly/plotly.js/issues/608
        self.label_latex = rf"{self.model.label_latex} $v_w={v_wall}, \alpha_n={alpha_n}" \
            if label_latex is None else label_latex
        self.label_unicode = f"{self.model.label_unicode}, v_w={v_wall}, αₙ={alpha_n}" \
            if label_unicode is None else label_unicode
        self.notes: tp.List[str] = []

        self.wn = model.w_n(alpha_n)
        self.tn = model.temp(self.wn, Phase.SYMMETRIC)
        if self.tn > model.t_crit:
            raise ValueError(f"Bubbles form only when T_nuc < T_crit. Got: T_nuc={self.tn}, T_crit={model.t_crit}")

        self.v: tp.Optional[np.ndarray] = None
        self.w: tp.Optional[np.ndarray] = None
        self.xi: tp.Optional[np.ndarray] = None

        self.gw_power_spectrum = None

        logger.info(
            "Initialized a bubble with: "
            f"model={self.model.label_unicode}, v_w={v_wall}, alpha_n={alpha_n}, T_nuc={self.tn}, w_nuc={self.wn}")

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

    def solve(self):
        if self.solved:
            logger.warning("Re-solving an already solved model!")
        self.v, self.w, self.xi = fluid_shell_generic(
            model=self.model,
            v_wall=self.v_wall, alpha_n=self.alpha_n, sol_type=self.sol_type)
        self.solved = True

        if self.entropy_density < 0:
            logger.error(
                "Entropy density should not be negative! Now entropy is decreasing. Got: %s",
                self.entropy_density
            )
        if self.thermal_energy_density < 0:
            logger.error(
                "Thermal energy density is negative. The bubble is therefore working as a heat engine. Got: %s",
                self.thermal_energy_density
            )

    def spectrum(self):
        raise NotImplementedError

    # -----
    # Thermodynamics
    # -----

    @functools.cached_property
    def ebar(self) -> float:
        return thermo.ebar(self.model, self.wn)

    @functools.cached_property
    def entropy_density(self) -> float:
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
            self.v_wall, ek=self.kinetic_energy_density, wb=self.wbar, wn=self.wn)

    @functools.cached_property
    def wbar(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return thermo.wbar(self.w, self.xi, self.v_wall, self.wn)
