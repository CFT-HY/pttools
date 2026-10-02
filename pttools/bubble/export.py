r"""Exportable fields of the bubbles.

The fields of the :py:attr:`~pttools.utils.fields.Preset.INIT` preset are named after the constructor parameters
of :py:class:`pttools.bubble.bubble.Bubble`, so that a bubble can be recreated from them.
"""

import datetime
import typing as tp

from pttools.bubble.phase import Phase
from pttools.bubble.solution_type import SolutionType
from pttools.utils.fields import Field, Fields, FieldShape, FieldType, Preset, decode_optional

if tp.TYPE_CHECKING:
    from pttools.bubble.bubble import Bubble

__all__ = [
    "BASE_BUBBLE_FIELDS",
    "BUBBLE_FIELDS",
    "PROFILE_AXIS",
    "csb2_Tn",
    "css2_Tn",
]

#: Name of the axis of the fluid profiles $v(\xi)$, $w(\xi)$ etc.
PROFILE_AXIS: str = "xi"

_MINIMAL = {Preset.MINIMAL, Preset.FULL}
_MINIMAL_INIT = {Preset.MINIMAL, Preset.FULL, Preset.INIT}
_FULL = {Preset.FULL}
_FULL_INIT = {Preset.FULL, Preset.INIT}


def _now(_obj: tp.Any) -> datetime.datetime:
    return datetime.datetime.now()


def css2_Tn(bubble: "Bubble") -> float:
    r"""$c_{s,s}^2(T_n)$, speed of sound squared in the symmetric phase at the nucleation temperature."""
    return float(bubble.model.cs2_temp(bubble.Tn, Phase.SYMMETRIC))


def csb2_Tn(bubble: "Bubble") -> float:
    r"""$c_{s,b}^2(T_n)$, speed of sound squared in the broken phase at the nucleation temperature."""
    return float(bubble.model.cs2_temp(bubble.Tn, Phase.BROKEN))


def _profile(name: str, description: str, presets: tp.Iterable[Preset]) -> Field:
    return Field(name, shape=FieldShape.RAGGED, axis=PROFILE_AXIS, presets=frozenset(presets), description=description)


#: Fields of :py:class:`pttools.bubble.bubble.BaseBubble`
BASE_BUBBLE_FIELDS: Fields = Fields(
    Field("datetime", getter=_now, type=FieldType.STR, presets=_FULL, description="time of the export"),
    Field("solving_duration", presets=_FULL, description="time taken by the solver in seconds"),
    Field("notes", type=FieldType.STR, presets=_FULL, description="notes about the solution, one per line"),
    # Input parameters
    Field("v_wall", presets=_MINIMAL_INIT, description=r"$v_\text{wall}$, wall speed"),
    Field("t_end", presets=_FULL_INIT, description=r"$t_\text{end}$, fluid shell integration cut-off"),
    Field("n_xi", type=FieldType.INT, presets=_FULL_INIT, description=r"$n_\xi$, number of $\xi$ points"),
    Field(
        "wm_guess", presets={Preset.INIT}, decode=decode_optional,
        description=r"$w_{-,\text{guess}}$, initial guess for the enthalpy behind the wall"),
    # Solution
    _profile("v", r"$v(\xi)$, fluid velocity profile", _MINIMAL),
    _profile("w", r"$w(\xi)$, enthalpy profile", _MINIMAL),
    _profile("xi", r"$\xi$, self-similar radius coordinates", _MINIMAL),
    _profile("T", r"$T(\xi)$, temperature profile", _FULL),
    _profile("e", r"$e(\xi)$, energy density profile", ()),
    _profile("p", r"$p(\xi)$, pressure profile", ()),
    _profile("s", r"$s(\xi)$, entropy density profile", ()),
    _profile("phase", r"$\phi(\xi)$, phase profile", ()),
    # Solution parameters
    Field("sp", presets=_FULL, description="$s_+$, entropy density in front of the wall"),
    Field("sm", presets=_FULL, description="$s_-$, entropy density behind the wall"),
    Field("Tp", presets=_FULL, description="$T_+$, temperature in front of the wall"),
    Field("Tm", presets=_FULL, description="$T_-$, temperature behind the wall"),
    Field("T_center", presets=_FULL, description=r"$T_\text{center}$, temperature at the center of the bubble"),
    Field("vp", presets=_FULL, description="$v_+$, fluid velocity in front of the wall"),
    Field("vm", presets=_FULL, description="$v_-$, fluid velocity behind the wall"),
    Field(
        "vp_tilde", presets=_FULL,
        description=r"$\tilde{v}_+$, fluid velocity in front of the wall in the wall frame"),
    Field(
        "vm_tilde", presets=_FULL,
        description=r"$\tilde{v}_-$, fluid velocity behind the wall in the wall frame"),
    Field("wp", presets=_FULL, description="$w_+$, enthalpy in front of the wall"),
    Field("wm", presets=_FULL, description="$w_-$, enthalpy behind the wall"),
    Field("w_center", presets=_FULL, description=r"$w_\text{center}$, enthalpy at the center of the bubble"),
    # Flags
    Field("failed", type=FieldType.BOOL, presets={Preset.MINIMAL}, description="whether the solution has errors"),
)

#: Fields of :py:class:`pttools.bubble.bubble.Bubble`
BUBBLE_FIELDS: Fields = Fields(
    BASE_BUBBLE_FIELDS,
    # Input parameters
    Field("alpha_n", presets=_MINIMAL_INIT, description=r"$\alpha_n$, transition strength"),
    Field(
        "sol_type", type=FieldType.STR, presets=_MINIMAL_INIT, decode=SolutionType,
        description="solution type"),
    Field(
        "thin_shell_limit", getter="thin_shell_t_points_min", type=FieldType.INT, presets=_FULL,
        description="limit of points for a shell to be so thin that it should be re-computed with more points"),
    Field(
        "thin_shell_t_points_min", type=FieldType.INT, presets={Preset.INIT},
        description="limit of points for a shell to be so thin that it should be re-computed with more points"),
    Field(
        "use_bag_solver", type=FieldType.BOOL, presets={Preset.INIT},
        description="whether the bag model specific fluid shell solver is used"),
    Field(
        "use_giese_solver", type=FieldType.BOOL, presets={Preset.INIT},
        description="whether the Giese et al. solver is used"),
    # Solution parameters
    Field("alpha_plus", presets=_FULL, description=r"$\alpha_+$, transition strength in front of the wall"),
    Field("sm_sh", presets=_FULL, description=r"$s_{-,\text{sh}}$, entropy density behind the shock"),
    Field("sn", presets=_FULL, description="$s_n$, entropy density at the nucleation temperature"),
    Field("Tn", presets=_FULL, description="$T_n$, nucleation temperature"),
    Field("v_cj", presets=_FULL, description=r"$v_\text{CJ}$, Chapman-Jouguet speed"),
    Field("v_sh", presets=_FULL, description=r"$v_\text{sh}$, shock speed"),
    Field("vm_sh", presets=_FULL, description=r"$v_{-,\text{sh}}$, fluid velocity behind the shock"),
    Field(
        "vm_tilde_sh", presets=_FULL,
        description=r"$\tilde{v}_{-,\text{sh}}$, fluid velocity behind the shock in the shock frame"),
    Field("wn", presets=_FULL, description="$w_n$, enthalpy at the nucleation temperature"),
    # Computed values
    Field("mean_adiabatic_index", presets=_FULL, description=r"$\Gamma$, mean adiabatic index"),
    Field(
        "css2_Tn", getter=css2_Tn, presets={Preset.MINIMAL},
        description="$c_{s,s}^2(T_n)$, speed of sound squared in the symmetric phase at the nucleation temperature"),
    Field(
        "csb2_Tn", getter=csb2_Tn, presets={Preset.MINIMAL},
        description="$c_{s,b}^2(T_n)$, speed of sound squared in the broken phase at the nucleation temperature"),
    Field(
        "alpha_theta_bar_n",
        description=r"$\alpha_{\bar{\theta}_n}$, transition strength based on the trace anomaly"),
    Field("kappa", description=r"$\kappa$, kinetic energy fraction"),
    Field("omega", description=r"$\omega$, thermal energy fraction"),
    Field("kinetic_energy_fraction", description="$K$, kinetic energy fraction"),
    Field("ubarf2", description=r"$\bar{U}_f^2$, enthalpy-weighted mean square fluid velocity"),
    Field("Psi_n", description=r"$\Psi_n$, inverse enthalpy ratio at the nucleation temperature"),
)
