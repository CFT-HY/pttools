"""A solution of the hydrodynamic equations"""

import abc
import functools
import logging
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble import const
from pttools.bubble.relativity import gamma
from pttools.bubble.thermo import va_kinetic_energy_density
import pttools.type_hints as th
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
            wm_guess: float | None = None,
            t_end: float = const.DEFAULT_T_END,
            n_xi: int = const.DEFAULT_N_XI):
        # Some functions such as np.vectorize tend to give 0D arrays, which may cause subtle errors later on.
        if v_wall is None or not np.isscalar(v_wall):
            raise ValueError(f"v_wall should be scalar. Did you give e.g. a 0D array instead? Got: v_wall={v_wall}")
        if isinstance(v_wall, int):
            v_wall = float(v_wall)
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

    def add_note(self, note: str) -> None:
        """Add a note to the solution"""
        self.notes.append(note)

    @abc.abstractmethod
    def solve(self) -> None:
        if self.solved:
            msg = (
                "Re-solving an already solved fluid profile! "
                "Already computed quantities will not be updated due to caching."
            )
            logger.warning(msg)
            self.add_note(msg)

    # =====
    # Plotting
    # =====

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
        return va_kinetic_energy_density(self.v, self.w, self.xi)

    @property
    def vp_vm_tilde_ratio(self) -> float:
        r"""$$\frac{\tilde{v}_+}{\tilde{v}_-}$$"""
        return self.vp_tilde / self.vm_tilde


class NotYetSolvedError(RuntimeError):
    """Error for accessing the properties of a bubble that has not been solved yet"""
