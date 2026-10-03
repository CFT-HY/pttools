"""A solution of the hydrodynamic equations."""

import abc
import functools
import logging
import os
import typing as tp
import uuid

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble import const
from pttools.bubble.export import BASE_BUBBLE_FIELDS
from pttools.bubble.thermo import va_kinetic_energy_density
from pttools.speedup import NAN_ARR
import pttools.type_hints as th
from pttools.utils.docstrings import copy_docstring_dec
from pttools.utils.fields import Extractable, Fields, FieldSpec, Preset
from pttools.utils.json import export_json
from pttools.utils.validation import ensure_floats

if tp.TYPE_CHECKING:
    from pttools.analysis.utils import FigAndAxes
    from pttools.models.model import Model

logger: logging.Logger = logging.getLogger(__name__)


class BaseBubble(Extractable, abc.ABC):
    """A common base class for bubbles and droplets."""

    #: The exportable fields of the bubble
    FIELDS: tp.ClassVar[Fields] = BASE_BUBBLE_FIELDS

    def __init__(
            self,
            model: "Model",
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

        #: Unique identifier of the bubble, which is used for deduplication when exporting.
        #: This is preserved when the bubble is pickled, e.g. when sent to another process.
        self.id: str = uuid.uuid4().hex

        # -----
        # Set parameters
        # -----
        #: Equation of state
        self.model: Model = model
        self.v_wall: float = v_wall
        self.t_end: float = t_end
        self.n_xi: int = n_xi
        self.w_center: float = np.nan if w_center is None else w_center
        #: $w_\text{outside}$ (far away)
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
        r"""Incoming entropy flux at the wall
        $$\tilde{\gamma}_+ \tilde{v}_+ {s}_+$$
        """

        self.entropy_flux_m: float = np.nan
        r"""Outgoing entropy flux at the wall
        $$\tilde{\gamma}_- \tilde{v}_- {s}_- $$
        """

        self.entropy_flux_diff: float = np.nan
        r"""Entropy flux difference at the wall
        $$\tilde{\gamma}_- \tilde{v}_- {s}_- - \tilde{\gamma}_+ \tilde{v}_+ {s}_+ $$
        """

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
        #: Whether the solver provided a solution (not necessarily a valid one)
        self.solved: bool = False
        #: Whether the solving has been attempted
        self.solving_attempted: bool = False
        # Specific errors
        #: Whether the junction conditions were not solved correctly
        self.invalid_junction: bool = False
        #: Whether there is a negative entropy flux across a junction
        self.negative_entropy_flux: bool = False
        #: Whether there is a total negative net enropy change in the system
        self.negative_net_entropy_change: bool = False
        #: Whether there is a numerical error, e.g. $\kappa + \omega \neq 1$
        self.numerical_error: bool = False
        #: Whether the solver crashed without returning output
        self.solver_crashed: bool = False
        #: Whether the solver failed but returned output
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

    # =====
    # Plotting
    # =====

    def plot(
            self,
            fig: plt.Figure | None = None,
            path: str | os.PathLike[str] | None = None,
            full_range: bool = False,
            **kwargs: tp.Any) -> plt.Figure:
        """Plot the velocity and enthalpy profiles of the bubble."""
        from pttools.analysis.plot_bubbles import plot_bubbles  # noqa: PLC0415
        return plot_bubbles([self], fig, path, full_range=full_range, **kwargs)

    def plot_v(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            full_range: bool = False,
            **kwargs: tp.Any) -> "FigAndAxes":
        """Plot the velocity profile of the bubble."""
        from pttools.analysis.plot_bubbles import plot_bubbles_v  # noqa: PLC0415
        return plot_bubbles_v([self], fig, ax, path, full_range=full_range, **kwargs)

    def plot_w(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            full_range: bool = False,
            **kwargs: tp.Any) -> "FigAndAxes":
        """Plot the enthalpy profile of the bubble."""
        from pttools.analysis.plot_bubbles import plot_bubbles_w  # noqa: PLC0415
        return plot_bubbles_w([self], fig, ax, path, full_range=full_range, **kwargs)

    # =====
    # Quantities
    # =====

    @functools.cached_property
    def e(self) -> th.FloatArr1D:
        r"""Energy density $e(\xi)$."""
        if not self.solved:
            raise NotYetSolvedError
        return self.model.e(self.w, self.phase)

    @functools.cached_property
    def p(self) -> th.FloatArr1D:
        r"""Pressure $p(\xi)$."""
        if not self.solved:
            raise NotYetSolvedError
        return self.model.p(self.w, self.phase)

    @functools.cached_property
    def s(self) -> th.FloatArr1D:
        r"""Entropy density $s(\xi)$."""
        if not self.solved:
            raise NotYetSolvedError
        return self.model.s(self.w, self.phase)

    @functools.cached_property
    def T(self) -> th.FloatArr1D:
        r"""Temperature profile $T(\xi)$."""
        return self.model.temp(w=self.w, phase=self.phase)

    @functools.cached_property
    @copy_docstring_dec(va_kinetic_energy_density, without_params=True)
    def va_kinetic_energy_density(self) -> float:
        if not self.solved:
            raise NotYetSolvedError
        return va_kinetic_energy_density(self.v, self.w, self.xi)

    @property
    def vp_vm_tilde_ratio(self) -> float:
        r"""Ratio of the fluid velocities at the wall.

        $$\frac{\tilde{v}_+}{\tilde{v}_-}$$
        """
        if not self.solved:
            raise NotYetSolvedError
        return self.vp_tilde / self.vm_tilde


class NotYetSolvedError(RuntimeError):
    """Error for accessing the properties of a bubble that has not been solved yet."""
