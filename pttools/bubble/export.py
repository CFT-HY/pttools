r"""Exportable fields of the bubbles.

The fields of the :py:attr:`~pttools.utils.fields.Preset.INIT` preset are named after the constructor parameters
of :py:class:`pttools.bubble.bubble.Bubble`, so that a bubble can be recreated from them.
"""

from collections.abc import Set

from pttools.bubble.solution_type import SolutionType
from pttools.utils.fields import (
    PRESETS_ALL,
    PRESETS_FULL,
    PRESETS_FULL_INIT,
    PRESETS_INIT,
    PRESETS_MINIMAL,
    PRESETS_MINIMAL_FULL,
    Field,
    Fields,
    FieldShape,
    FieldType,
    Preset,
    decode_optional,
)
from pttools.utils.time import now

__all__ = [
    "BASE_BUBBLE_FIELDS",
    "BUBBLE_FIELDS",
    "PROFILE_AXIS",
]

#: Name of the axis of the fluid profiles $v(\xi)$, $w(\xi)$ etc.
PROFILE_AXIS: str = "xi"



def _profile(name: str, description: str, presets: Set[Preset] = frozenset()) -> Field:
    """Create a field for a fluid profile, which is a ragged array along the axis :py:data:`PROFILE_AXIS`.

    :param name: name of the field
    :param description: description of the field
    :param presets: the presets that include the field
    :return: the field
    """
    return Field(name, shape=FieldShape.RAGGED, axis=PROFILE_AXIS, presets=presets, description=description)


#: Fields of :py:class:`pttools.bubble.bubble.BaseBubble`
BASE_BUBBLE_FIELDS: Fields = Fields(
    Field("datetime", getter=now, type=FieldType.STR, presets=PRESETS_FULL, description="time of the export"),
    Field("solving_duration", presets=PRESETS_FULL, description="time taken by the solver in seconds"),
    Field("notes", type=FieldType.STR, presets=PRESETS_FULL, description="notes about the solution, one per line"),
    # Input parameters
    Field("v_wall", presets=PRESETS_ALL, description=r"$v_\text{wall}$, wall speed"),
    Field("t_end", presets=PRESETS_FULL_INIT, description=r"$t_\text{end}$, fluid shell integration cut-off"),
    Field("n_xi", type=FieldType.INT, presets=PRESETS_FULL_INIT, description=r"$n_\xi$, number of $\xi$ points"),
    Field(
        "wm_guess", presets=PRESETS_INIT, decode=decode_optional,
        description=r"$w_{-,\text{guess}}$, initial guess for the enthalpy behind the wall"),
    # Solution
    _profile("v", r"$v(\xi)$, fluid velocity profile", PRESETS_MINIMAL_FULL),
    _profile("w", r"$w(\xi)$, enthalpy profile", PRESETS_MINIMAL_FULL),
    _profile("xi", r"$\xi$, self-similar radius coordinates", PRESETS_MINIMAL_FULL),
    _profile("T", r"$T(\xi)$, temperature profile", PRESETS_FULL),
    _profile("e", r"$e(\xi)$, energy density profile"),
    _profile("p", r"$p(\xi)$, pressure profile"),
    _profile("s", r"$s(\xi)$, entropy density profile"),
    _profile("phase", r"$\phi(\xi)$, phase profile"),
    # Solution parameters
    Field("sp", presets=PRESETS_FULL, description="$s_+$, entropy density in front of the wall"),
    Field("sm", presets=PRESETS_FULL, description="$s_-$, entropy density behind the wall"),
    Field("Tp", presets=PRESETS_FULL, description="$T_+$, temperature in front of the wall"),
    Field("Tm", presets=PRESETS_FULL, description="$T_-$, temperature behind the wall"),
    Field("T_center", presets=PRESETS_FULL, description=r"$T_\text{center}$, temperature at the center of the bubble"),
    Field("vp", presets=PRESETS_FULL, description="$v_+$, fluid velocity in front of the wall"),
    Field("vm", presets=PRESETS_FULL, description="$v_-$, fluid velocity behind the wall"),
    Field(
        "vp_tilde", presets=PRESETS_FULL,
        description=r"$\tilde{v}_+$, fluid velocity in front of the wall in the wall frame"),
    Field(
        "vm_tilde", presets=PRESETS_FULL,
        description=r"$\tilde{v}_-$, fluid velocity behind the wall in the wall frame"),
    Field("wp", presets=PRESETS_FULL, description="$w_+$, enthalpy in front of the wall"),
    Field("wm", presets=PRESETS_FULL, description="$w_-$, enthalpy behind the wall"),
    Field("w_center", presets=PRESETS_FULL, description=r"$w_\text{center}$, enthalpy at the center of the bubble"),
    # Flags
    Field("failed", type=FieldType.BOOL, presets=PRESETS_MINIMAL, description="whether the solution has errors"),
)

#: Fields of :py:class:`pttools.bubble.bubble.Bubble`
BUBBLE_FIELDS: Fields = Fields(
    BASE_BUBBLE_FIELDS,
    # Input parameters
    Field("alpha_n", presets=PRESETS_ALL, description=r"$\alpha_n$, transition strength"),
    Field(
        "sol_type", type=FieldType.STR, presets=PRESETS_ALL, decode=SolutionType,
        description="solution type"),
    Field(
        "thin_shell_limit", getter="thin_shell_t_points_min", type=FieldType.INT, presets=PRESETS_FULL,
        description="limit of points for a shell to be so thin that it should be re-computed with more points"),
    Field(
        "thin_shell_t_points_min", type=FieldType.INT, presets=PRESETS_INIT,
        description="limit of points for a shell to be so thin that it should be re-computed with more points"),
    Field(
        "use_bag_solver", type=FieldType.BOOL, presets=PRESETS_INIT,
        description="whether the bag model specific fluid shell solver is used"),
    Field(
        "use_giese_solver", type=FieldType.BOOL, presets=PRESETS_INIT,
        description="whether the Giese et al. solver is used"),
    # Solution parameters
    Field("alpha_plus", presets=PRESETS_FULL, description=r"$\alpha_+$, transition strength in front of the wall"),
    Field("sm_sh", presets=PRESETS_FULL, description=r"$s_{-,\text{sh}}$, entropy density behind the shock"),
    Field("sn", presets=PRESETS_FULL, description="$s_n$, entropy density at the nucleation temperature"),
    Field("Tn", presets=PRESETS_FULL, description="$T_n$, nucleation temperature"),
    Field("v_cj", presets=PRESETS_FULL, description=r"$v_\text{CJ}$, Chapman-Jouguet speed"),
    Field("v_sh", presets=PRESETS_FULL, description=r"$v_\text{sh}$, shock speed"),
    Field("vm_sh", presets=PRESETS_FULL, description=r"$v_{-,\text{sh}}$, fluid velocity behind the shock"),
    Field(
        "vm_tilde_sh", presets=PRESETS_FULL,
        description=r"$\tilde{v}_{-,\text{sh}}$, fluid velocity behind the shock in the shock frame"),
    Field("wn", presets=PRESETS_FULL, description="$w_n$, enthalpy at the nucleation temperature"),
    # Computed values
    Field("mean_adiabatic_index", presets=PRESETS_FULL, description=r"$\Gamma$, mean adiabatic index"),
    Field(
        "css2_Tn", presets=PRESETS_MINIMAL,
        description="$c_{s,s}^2(T_n)$, speed of sound squared in the symmetric phase at the nucleation temperature"),
    Field(
        "csb2_Tn", presets=PRESETS_MINIMAL,
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
