import datetime
import logging
import typing as tp

import numpy as np

from pttools.bubble.boundary import Phase, SolutionType
from pttools.bubble.fluid import fluid_shell_generic
from pttools.speedup.export import export_json
if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


class Bubble:
    """A solution of the hydrodynamic equations

    (Defined in the meeting with Hindmarsh on 2022-07-06.)
    """
    def __init__(
            self,
            model: "Model", v_wall: float, alpha_n: float,
            sol_type: SolutionType = None,
            label: str = None):
        if v_wall < 0 or v_wall > 1:
            raise ValueError(f"Invalid v_wall={v_wall}")
        if alpha_n < 0 or alpha_n > 1 or alpha_n < model.alpha_n_min:
            raise ValueError(f"Invalid alpha_n={alpha_n}. Minimum for the model: {model.alpha_n_min}")
        if sol_type is None:
            # Todo: determine solution type
            pass

        self.model: Model = model
        self.v_wall = v_wall
        self.alpha_n = alpha_n
        self.sol_type = sol_type
        self.solved = False
        self.label = rf"{self.model.label}, $v_w={v_wall}, \alpha_n={alpha_n}$" if label is None else label
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
            f"model={self.model}, v_w={v_wall}, alpha_n={alpha_n}, T_nuc={self.tn}, w_nuc={self.wn}")

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

    def solve(self):
        if self.solved:
            logger.warning("Re-solving an already solved model!")
        self.v, self.w, self.xi = fluid_shell_generic(
            model=self.model,
            v_wall=self.v_wall, alpha_n=self.alpha_n, sol_type=self.sol_type)

    def spectrum(self):
        raise NotImplementedError
