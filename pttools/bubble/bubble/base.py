"""A solution of the hydrodynamic equations"""

import abc
import datetime
import functools
import logging
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble import const
from pttools.bubble.thermo import va_kinetic_energy_density
from pttools.speedup import NAN_ARR
import pttools.type_hints as th
from pttools.utils.json import export_json
from pttools.utils.validation import ensure_floats
if tp.TYPE_CHECKING:
    from pttools.models.model import Model
    from pttools.analysis.utils import FigAndAxes

logger = logging.getLogger(__name__)


class BaseBubble(abc.ABC):
    """A common base class for bubbles and droplets"""
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
        v_wall, w_center, w_outside, wm_guess = ensure_floats(
            {"v_wall": v_wall, "w_center": w_center, "w_outside": w_outside, "wm_guess": wm_guess},
            allow_none=True
        )

        # -----
        # Set parameters
        # -----
        #: Equation of state
        self.model: Model = model
        #: Wall speed $v_\text{wall}$
        self.v_wall: float = v_wall
        #: Fluid shell integration cut-off $t_\text{end}$
        self.t_end: float = t_end
        #: Number of $\xi$ points, $n_\xi$
        self.n_xi: int = n_xi
        #: $w_\text{center}$
        self.w_center: float = np.nan if w_center is None else w_center
        #: $w_\text{outside}$ (far away)
        self.w_outside: float = np.nan if w_outside is None else w_outside
        #: $w_{-,\text{guess}}$
        self.wm_guess: float | None = wm_guess

        # -----
        # Output arrays
        # -----
        #: Fluid velocity profile $v(\xi)$
        self.v: th.FloatArr1D = NAN_ARR
        #: Enthalpy profile $w(\xi)$
        self.w: th.FloatArr1D = NAN_ARR
        #: Self-similar droplet radius coordinates $\xi$
        self.xi: th.FloatArr1D = NAN_ARR
        #: Phase profile $\phi(\xi)$
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

        #: $s_+$
        self.sp: float = np.nan
        #: $s_-$
        self.sm: float = np.nan
        #: $T_+$
        self.Tp: float = np.nan
        #: $T_-$
        self.Tm: float = np.nan
        #: $T_\text{center}$
        self.T_center: float = np.nan
        #: $v_+$
        self.vp: float = np.nan
        #: $\tilde{v}_+$
        self.vp_tilde: float = np.nan
        #: $v_-$
        self.vm: float = np.nan
        #: $\tilde{v}_-$
        self.vm_tilde: float = np.nan
        #: $w_+$
        self.wp: float = np.nan
        #: $w_-$
        self.wm: float = np.nan

        # Flags
        #: Whether the solution has errors
        self.failed = False
        #: Whether the solver provided a solution (not necessarily a valid one)
        self.solved = False
        #: Whether the solving has been attempted
        self.solving_attempted = False
        # Specific errors
        #: Whether the junction conditions were not solved correctly
        self.invalid_junction = False
        #: Whether there is a negative entropy flux across a junction
        self.negative_entropy_flux = False
        #: Whether there is a total negative net enropy change in the system
        self.negative_net_entropy_change = False
        #: Whether there is a numerical error, e.g. $\kappa + \omega \neq 1$
        self.numerical_error = False
        #: Whether the solver crashed without returning output
        self.solver_crashed = False
        #: Whether the solver failed but returned output
        self.solver_failed = False

    def add_note(self, note: str) -> None:
        """Add a note to the solution"""
        self.notes.append(note)

    def export(self, path: str | None = None) -> dict[str, tp.Any]:
        """Export the bubble data"""
        data = {
            "datetime": datetime.datetime.now(),
            "solving_duration": self.solving_duration,
            "notes": self.notes,
            # Input parameters
            "model": self.model.export(),
            "v_wall": self.v_wall,
            "t_end": self.t_end,
            "n_xi": self.n_xi,
            # Solution
            "v": self.v,
            "w": self.w,
            "xi": self.xi,
            "T": self.T,
            # Solution parameters
            "sp": self.sp,
            "sm": self.sm,
            "Tp": self.Tp,
            "Tm": self.Tm,
            "T_center": self.T_center,
            "vp": self.vp,
            "vm": self.vm,
            "vp_tilde": self.vp_tilde,
            "vm_tilde": self.vm_tilde,
            "wp": self.wp,
            "wm": self.wm,
            "w_center": self.w_center
        }
        if path is not None:
            export_json(data, path)
        return data

    @abc.abstractmethod
    def solve(self) -> None:
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
            path: str | None = None,
            full_range: bool = False,
            **kwargs) -> plt.Figure:
        """Plot the velocity and enthalpy profiles of the bubble"""
        from pttools.analysis.plot_bubbles import plot_bubbles
        return plot_bubbles([self], fig, path, full_range=full_range, **kwargs)

    def plot_v(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | None = None,
            full_range: bool = False,
            **kwargs) -> "FigAndAxes":
        """Plot the velocity profile of the bubble"""
        from pttools.analysis.plot_bubbles import plot_bubbles_v
        return plot_bubbles_v([self], fig, ax, path, full_range=full_range, **kwargs)

    def plot_w(
            self,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            path: str | None = None,
            full_range: bool = False,
            **kwargs) -> "FigAndAxes":
        """Plot the enthalpy profile of the bubble"""
        from pttools.analysis.plot_bubbles import plot_bubbles_w
        return plot_bubbles_w([self], fig, ax, path, full_range=full_range, **kwargs)

    # =====
    # Quantities
    # =====

    @functools.cached_property
    def e(self):
        r"""Energy density $e(\xi)$"""
        if not self.solved:
            raise NotYetSolvedError
        return self.model.e(self.w, self.phase)

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
    def T(self):
        r"""Temperature profile $T(\xi)$"""
        return self.model.temp(w=self.w, phase=self.phase)

    @functools.cached_property
    def va_kinetic_energy_density(self) -> float:  # pylint: disable=missing-function-docstring
        if not self.solved:
            raise NotYetSolvedError
        return va_kinetic_energy_density(self.v, self.w, self.xi)

    @property
    def vp_vm_tilde_ratio(self) -> float:
        r"""$$\frac{\tilde{v}_+}{\tilde{v}_-}$$"""
        if not self.solved:
            raise NotYetSolvedError
        return self.vp_tilde / self.vm_tilde


class NotYetSolvedError(RuntimeError):
    """Error for accessing the properties of a bubble that has not been solved yet"""
