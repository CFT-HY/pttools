"""Base class of the bubbles and droplets that are computed by solving the fluid equations."""

from __future__ import annotations

import abc
import logging
import os
import typing as tp
import uuid

import numpy as np

from pttools.bubble import const
from pttools.bubble.bubble.base import BaseBubble
from pttools.speedup import NAN_ARR
import pttools.type_hints as th
from pttools.utils.cache import Cacheable
from pttools.utils.fields import FieldSpec, Preset
from pttools.utils.json import export_json
from pttools.utils.validation import ensure_floats

if tp.TYPE_CHECKING:
    from pttools.models.model import Model

__all__ = ["ComputedBubble"]

logger: logging.Logger = logging.getLogger(__name__)


class ComputedBubble(Cacheable, BaseBubble, abc.ABC):
    """Common base class of the bubbles and droplets that are computed by solving the fluid equations.

    The bubbles can be cached with :py:meth:`pttools.utils.cache.Cacheable.cached`.
    """

    def __init__(
            self,
            model: Model,
            v_wall: float,
            w_center: float | None = None,
            w_outside: float | None = None,
            wm_guess: float | None = None,
            t_end: float = const.DEFAULT_T_END,
            n_xi: int = const.DEFAULT_N_XI,
            label_latex: str = "UNSET",
            label_unicode: str = "UNSET"):
        r"""Set the parameters of the bubble.

        :param model: The equation of state object
        :param v_wall: Wall velocity $v_\text{wall}$
        :param w_center: Enthalpy at the center of the bubble $w_\text{center}$, if known
        :param w_outside: Enthalpy far away from the bubble $w_\text{outside}$, if known
        :param wm_guess: Initial guess for the enthalpy behind the wall $w_-$
        :param t_end: The maximum value for the fluid shell ODE integration parameter
        :param n_xi: Number of points in the fluid velocity profile
        :param label_latex: LaTeX label for plots
        :param label_unicode: Unicode label for plots
        """
        v_wall, w_center, w_outside, wm_guess = ensure_floats(
            {"v_wall": v_wall, "w_center": w_center, "w_outside": w_outside, "wm_guess": wm_guess},
            allow_none=True
        )

        self.id: str = uuid.uuid4().hex

        # -----
        # Set parameters
        # -----
        self.model: Model = model
        self.v_wall: float = v_wall
        self.t_end: float = t_end
        self.n_xi: int = n_xi
        self.w_center: float = np.nan if w_center is None else w_center
        self.w_outside: float = np.nan if w_outside is None else w_outside
        self.wm_guess: float | None = wm_guess

        # -----
        # Output arrays
        # -----
        self.v: th.FloatArr1D = NAN_ARR
        self.w: th.FloatArr1D = NAN_ARR
        self.xi: th.FloatArr1D = NAN_ARR
        self.phase: th.FloatArr1D = NAN_ARR

        # -----
        # Output values
        # -----
        self.label_latex: str = label_latex
        self.label_unicode: str = label_unicode
        self.notes: list[str] = []
        self.solving_duration: float = np.nan

        self.entropy_flux_p: float = np.nan
        self.entropy_flux_m: float = np.nan
        self.entropy_flux_diff: float = np.nan
        self.sp: float = np.nan
        self.sm: float = np.nan
        self.Tp: float = np.nan
        self.Tm: float = np.nan
        self.T_center: float = np.nan
        self.vp: float = np.nan
        self.vp_tilde: float = np.nan
        self.vm: float = np.nan
        self.vm_tilde: float = np.nan
        self.wp: float = np.nan
        self.wm: float = np.nan

        # Flags
        self.failed: bool = False
        self.solved: bool = False
        self.solving_attempted: bool = False
        # Specific errors
        self.invalid_junction: bool = False
        self.negative_entropy_flux: bool = False
        self.negative_net_entropy_change: bool = False
        self.numerical_error: bool = False
        self.solver_crashed: bool = False
        self.solver_failed: bool = False

    def add_note(self, note: str) -> None:
        """Add a note to the solution."""
        self.notes.append(note)

    def export(
            self,
            path: str | os.PathLike[str] | None = None,
            fields: FieldSpec = Preset.FULL,
            model_fields: FieldSpec = Preset.FULL) -> dict[str, tp.Any]:
        """Export the bubble data to a dictionary, and optionally save it as a JSON file.

        :param path: path of the JSON file
        :param fields: the fields of the bubble to export, see :py:data:`pttools.utils.fields.FieldSpec`
        :param model_fields: the fields of the model to export
        :return: the exported data, where the model data is under the key ``"model"``
        """
        data = {
            # extract() is used instead of export(), since user-created model classes may override export().
            "model": self.model.extract(model_fields),
            **self.extract(fields)
        }
        if path is not None:
            export_json(data, path)
        return data

    @abc.abstractmethod
    def solve(self) -> None:
        """Solve the fluid velocity profile of the bubble.

        Subclasses should call this with ``super().solve()`` before solving.
        This base implementation marks the solving as attempted,
        and warns and adds a note if the bubble has already been solved,
        as the cached quantities will not be updated.
        """
        if self.solving_attempted:
            msg = (
                "Re-solving a bubble! "
                "Already computed quantities will not be updated due to caching."
            )
            logger.warning(msg)
            self.add_note(msg)
        self.solving_attempted = True
