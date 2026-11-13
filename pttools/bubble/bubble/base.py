"""Base classes of the solutions of the hydrodynamic equations."""

from __future__ import annotations

import abc
import functools
import os
import typing as tp

import numpy as np

from pttools.bubble.export import BASE_BUBBLE_FIELDS
from pttools.bubble.thermo import va_kinetic_energy_density
import pttools.type_hints as th
from pttools.utils.docstrings import copy_docstring_dec
from pttools.utils.fields import Extractable, Fields

if tp.TYPE_CHECKING:
    from pttools.models.model import Model
    from pttools.plot.utils import FigAndAxes

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["BaseBubble", "NotYetSolvedError"]


class BaseBubble(Extractable, abc.ABC):
    """Common base class of the bubbles and droplets, both computed and imported.

    This defines the attributes of the solutions and the quantities that are computed from them,
    so that the plotting and analysis functions can take any bubble.
    The attributes are set either by the solver of :py:class:`~pttools.bubble.bubble.computed.ComputedBubble`,
    or from the data of a file by :py:class:`pttools.export.bubble.ImportedBubble`.
    """

    #: The exportable fields of the bubble
    FIELDS: tp.ClassVar[Fields] = BASE_BUBBLE_FIELDS

    #: Unique identifier of the bubble, which is used for deduplication when exporting.
    #: This is preserved when the bubble is pickled, e.g. when sent to another process.
    id: str
    #: Equation of state
    model: Model

    # -----
    # Parameters
    # -----
    v_wall: float
    t_end: float
    n_xi: int
    w_center: float
    #: $w_\text{outside}$ (far away)
    w_outside: float
    wm_guess: float | None

    # -----
    # Profiles
    # -----
    v: th.FloatArr1D
    w: th.FloatArr1D
    xi: th.FloatArr1D
    phase: th.FloatArr1D

    # -----
    # Values
    # -----
    label_latex: str
    label_unicode: str
    notes: list[str]
    solving_duration: float

    entropy_flux_p: float
    r"""Incoming entropy flux at the wall
    $$\tilde{\gamma}_+ \tilde{v}_+ {s}_+$$
    """

    entropy_flux_m: float
    r"""Outgoing entropy flux at the wall
    $$\tilde{\gamma}_- \tilde{v}_- {s}_- $$
    """

    entropy_flux_diff: float
    r"""Entropy flux difference at the wall
    $$\tilde{\gamma}_- \tilde{v}_- {s}_- - \tilde{\gamma}_+ \tilde{v}_+ {s}_+ $$
    """

    sp: float
    sm: float
    Tp: float
    Tm: float
    T_center: float
    vp: float
    vp_tilde: float
    vm: float
    vm_tilde: float
    wp: float
    wm: float

    # -----
    # Flags
    # -----
    failed: bool
    #: Whether the solver provided a solution (not necessarily a valid one)
    solved: bool
    #: Whether the solving has been attempted
    solving_attempted: bool
    # Specific errors
    # These are documented by the descriptions of their fields in pttools.bubble.export.
    invalid_junction: bool
    negative_entropy_flux: bool
    negative_net_entropy_change: bool
    non_monotonic_xi: bool
    numerical_error: bool
    solver_crashed: bool
    solver_failed: bool

    # =====
    # Plotting
    # =====

    def plot(
            self,
            fig: Figure | None = None,
            path: str | os.PathLike[str] | None = None,
            full_range: bool = False,
            **kwargs: tp.Any) -> Figure:
        """Plot the velocity and enthalpy profiles of the bubble."""
        from pttools.plot.bubbles import plot_bubbles  # noqa: PLC0415
        return plot_bubbles([self], fig, path, full_range=full_range, **kwargs)

    def plot_v(
            self,
            fig: Figure | None = None,
            ax: Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            full_range: bool = False,
            **kwargs: tp.Any) -> FigAndAxes:
        """Plot the velocity profile of the bubble."""
        from pttools.plot.bubbles import plot_bubbles_v  # noqa: PLC0415
        return plot_bubbles_v([self], fig, ax, path, full_range=full_range, **kwargs)

    def plot_w(
            self,
            fig: Figure | None = None,
            ax: Axes | None = None,
            path: str | os.PathLike[str] | None = None,
            full_range: bool = False,
            **kwargs: tp.Any) -> FigAndAxes:
        """Plot the enthalpy profile of the bubble."""
        from pttools.plot.bubbles import plot_bubbles_w  # noqa: PLC0415
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

    @property
    def has_profile(self) -> bool:
        """Whether the bubble has a finite fluid velocity profile, from which e.g. the spectra can be computed.

        The bubble may have a profile even if the solution has failed, e.g. due to the validations of the solution.
        """
        return self.solved and not self.solver_crashed and bool(np.all(np.isfinite(self.v)))

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
